import gym
import torch as th
from torch import nn
import numpy as np
from typing import Union, Tuple
from torch.distributions import Categorical
import time
from stable_baselines3.common.callbacks import ConvertCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pls.observation_nets.observation_nets import Observation_Net_Carracing
# import matplotlib.pyplot as plt
from collections import deque
from gym.spaces import Box

from stable_baselines3.common.type_aliases import (
    GymObs,
    GymStepReturn,
    Schedule,
)

from pls.deepproblog.light import DeepProbLogLayer, DeepProbLogLayer_Approx
from pls.dpl_policies.carracing.util import get_ground_truth_of_grass
from os import path
import pickle
import torch.nn.functional as F

class Carracing_Encoder(nn.Module):
    def __init__(self, n_stacked_images):
        super(Carracing_Encoder, self).__init__()

    def forward(self, x):
        return x


class Carracing_Callback(ConvertCallback):
    def __init__(self, callback):
        super(Carracing_Callback, self).__init__(callback)
    def on_step(self) -> bool:
        logger = self.locals["self"].logger
        mass = self.locals["mass"]
        action_lookup = self.locals["action_lookup"]
        object_detect_probs = self.locals["object_detect_probs"]
        base_policy = self.locals["base_policy"]
        policy = self.locals["self"].policy
        for act in range(self.locals["self"].action_space.n):
            logger.record(
                f"policy/shielded {action_lookup[act]}",
                float(mass.probs[0][act]),
            )
        if object_detect_probs.get("alpha") is not None:
            logger.record(
                f"safety/alpha",
                float(object_detect_probs.get("alpha")),
            )
        abs_safe_next_shielded = policy.get_step_safety(
            mass.probs,
            object_detect_probs["ground_truth_grass"],
        )
        abs_safe_next_base = policy.get_step_safety(
            base_policy,
            object_detect_probs["ground_truth_grass"],
        )
        rel_safe_next_shielded = policy.get_step_safety(
            mass.probs,
            object_detect_probs["grass"]
        )
        rel_safe_next_base = policy.get_step_safety(
            base_policy,
            object_detect_probs["grass"]
        )

        self.locals["abs_safeties_shielded"].append(abs_safe_next_shielded)
        self.locals["abs_safeties_base"].append(abs_safe_next_base)
        self.locals["rel_safeties_shielded"].append(rel_safe_next_shielded)
        self.locals["rel_safeties_base"].append(rel_safe_next_base)

        if object_detect_probs.get("alpha") is not None:
            self.locals["alphas"].append(object_detect_probs["alpha"])
        if object_detect_probs.get("num_rejected_samples") is not None:
            self.locals["nums_rejected_samples"].append(object_detect_probs["num_rejected_samples"])
        # if is in a risky situation
        if th.any(object_detect_probs["ground_truth_grass"], dim=1):
            self.locals["n_risky_states"].append(1)
        else:
            self.locals["n_risky_states"].append(0)


class Carracing_Monitor(Monitor):
    def __init__(self, *args, vio_len, **kwargs):
        self.vio_len = vio_len
        super(Carracing_Monitor, self).__init__(*args, **kwargs)

    def reset(self, **kwargs) -> GymObs:
        # self.violate_constraint = False # TODO
        # self.violate_constraint_dequeue = deque(maxlen=self.vio_len)
        self.cont_in_grass_len = 0
        self.max_cont_in_grass_len = 0

        self.total_violate_len = 0
        self.cont_violate_len = 0
        self.max_cont_violate_len = 0
        output = super(Carracing_Monitor, self).reset(**kwargs)
        return output


    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

        symbolic_state = get_ground_truth_of_grass(th.from_numpy(observation.copy()).unsqueeze(0))

        is_in_grass = th.all(symbolic_state)
        if is_in_grass == True:
            self.cont_in_grass_len += 1
        else:
            self.cont_in_grass_len = 0
        self.max_cont_in_grass_len = max(self.cont_in_grass_len, self.max_cont_in_grass_len)


        left_grass_only = th.logical_and(symbolic_state[0,1].bool(), ~symbolic_state[0,2].bool())
        right_grass_only = th.logical_and(~symbolic_state[0,1].bool(), symbolic_state[0,2].bool())
        violate_cont = th.logical_or(left_grass_only, right_grass_only)
        if violate_cont == True:
            self.cont_violate_len += 1
            self.total_violate_len += 1
        else:
            self.cont_violate_len = 0
        self.max_cont_violate_len = max(self.cont_violate_len, self.max_cont_violate_len)

        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)

            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
                "last_r": reward,
                "violate_constraint": self.max_cont_in_grass_len > self.vio_len,
                # "violate_constraint": violate_constraint,
                "is_success": info["is_success"],
                "max_cont_in_grass_len": self.max_cont_in_grass_len,
                "max_cont_violate_len": self.max_cont_violate_len,
                "total_violate_len": self.total_violate_len
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info


class CustomCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    A CNN based architecture the for PPO features extractor.
    The architecture is a standard multi layer CNN with ReLU activations.

    :param observation_space: Metadata about the observation space to operate over. Assumes shape represents HWC.
    :param features_dim: The number of features to extract from the observations.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNNFeaturesExtractor, self).__init__(observation_space, features_dim)
        # TODO: automatically compute features_dim (now it must be manually provided according to the observation size)
        self._features_dim = features_dim # This is the size of the output
        n_stacked_images, _, _ = observation_space.shape
        self.conv1 = nn.Conv2d(in_channels=n_stacked_images, out_channels=8, kernel_size=5, stride=2, padding=1) # (8, 15, 15)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=1) # (16, 7, 7)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1) # (32, 3, 3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1) # (64, 1, 1)


    """
    Forward pass through the model.

    :param observations: BCHW tensor representing the states to extract features from.

    Returns:
        Tensor of shape (B,features_dim) representing a compressed view of the input image.
        Intended to be used for policy and
    """

    def forward(self, x: th.Tensor) -> th.Tensor:
        # convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x.view(-1, self._features_dim)


class Carracing_DPLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            image_encoder: Carracing_Encoder = None,
            shielding_params = None,
            net_input_dim = 1,
            folder = None,
            **kwargs
    ):
        observation_space = Box(
            low=-1,
            high=1,
            shape=(
                observation_space.shape[0], net_input_dim, net_input_dim
            )
        )
        super(Carracing_DPLActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule,
                                                             features_extractor_class=CustomCNNFeaturesExtractor,
                                                             **kwargs)
        ###############################
        self.image_encoder = image_encoder
        self.n_grass_locs = shielding_params["n_grass_locs"]
        self.alpha = shielding_params["alpha"]
        self.differentiable_shield = shielding_params["differentiable_shield"]
        self.net_input_dim = net_input_dim
        self.n_actions = 5
        self.folder = path.join(path.dirname(__file__), "../../..", folder)
        self.program_path = path.join(self.folder, "../../../data", shielding_params["program_type"]+".pl")

        if self.program_path:
            # pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments_safety/carracing/data/carracing_grass3.pl")
            # self.program_path = pp
            with open(self.program_path) as f:
                self.program = f.read()


        if self.alpha == 0: # Baseline
            pass
        elif self.differentiable_shield: # PLS
            self.use_learned_observations = shielding_params["use_learned_observations"]
            self.train_observations = shielding_params["train_observations"] if self.use_learned_observations else None
            self.noisy_observations = shielding_params["noisy_observations"] if self.use_learned_observations else None
            self.observation_type = shielding_params["observation_type"] if self.use_learned_observations else None
        else: # VSRL
            self.vsrl_use_renormalization = shielding_params["vsrl_use_renormalization"]
            if not self.vsrl_use_renormalization:
                self.max_num_rejected_samples = shielding_params["max_num_rejected_samples"]
            self.use_learned_observations = shielding_params["use_learned_observations"]
            self.train_observations = shielding_params["train_observations"] if self.use_learned_observations else None
            self.noisy_observations = shielding_params["noisy_observations"] if self.use_learned_observations else None
            self.observation_type = shielding_params["observation_type"] if self.use_learned_observations else None


        if self.alpha == 0: # NO shielding
            pass
        else: # HARD shielding and SOFT shielding
            # self.queries = ["safe_action(do_nothing)", "safe_action(accelerate)",
            #                 "safe_action(brake)", "safe_action(turn_left)",
            #                 "safe_action(turn_right)"][: self.n_actions]
            # input_struct = {
            #     "grass": [i for i in range(self.n_grass_locs)],
            #     "action": [i for i in range(self.n_grass_locs, self.n_grass_locs + self.n_actions)]
            # }
            # query_struct = {"safe_action": {"do_nothing": 0, "accelerate": 1, "brake": 2, "turn_left": 3, "turn_right": 4}}
            # pp = path.join(self.folder, "../../../data", "dpl_layer.p")
            # # pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments_safety/carracing/data/dpl_layer.p")
            # self.dpl_layer = self.get_layer(
            #     pp, program=self.program, queries=self.queries, evidences=["safe_next"],
            #     input_struct=input_struct, query_struct=query_struct
            # )
            if self.use_learned_observations:
                use_cuda = False
                device = th.device("cuda" if use_cuda else "cpu")
                self.observation_model = Observation_Net_Carracing(input_size=self.net_input_dim*self.net_input_dim, output_size=3).to(device)
                pp = path.join(self.folder, "../../data", self.observation_type)
                # pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments_safety/carracing/map1/data", self.observation_type)
                self.observation_model.load_state_dict(th.load(pp))

        debug_queries = ["safe_next"]
        debug_query_struct = {"safe_next": 0}
        debug_input_struct = {
            "grass": [i for i in range(self.n_grass_locs)],
            "action": [i for i in range(self.n_grass_locs, self.n_grass_locs + self.n_actions)]
        }
        pp = path.join(self.folder, "../../../data", "query_safety_layer.p")
        # pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments_safety/carracing/data/query_safety_layer.p")
        self.query_safety_layer = self.get_layer(
            pp, program=self.program, queries=debug_queries, evidences=[],
            input_struct=debug_input_struct,query_struct=debug_query_struct
        )

        self._build(lr_schedule)

    def get_layer(self, cache_path, program, queries, evidences, input_struct, query_struct):

        if path.exists(cache_path):
            try:
                return pickle.load(open(cache_path, "rb"))
            except:
                pass
                print("Nooooo")
                layer = DeepProbLogLayer_Approx(
                    program=program, queries=queries, evidences=evidences,
                    input_struct=input_struct, query_struct=query_struct
                )
                return layer

        layer = DeepProbLogLayer_Approx(
            program=program, queries=queries, evidences=evidences,
            input_struct=input_struct, query_struct=query_struct
        )
        pickle.dump(layer, open(cache_path, "wb"))
        return layer

    def get_step_safety(self, policy_distribution, grass_probs):
        with th.no_grad():
            abs_safe_next = self.query_safety_layer(
                x={
                    "grass": grass_probs,
                    "action": policy_distribution,
                }
            )
            return abs_safe_next["safe_next"]

    def forward(self, x, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        obs = self.image_encoder(x)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        base_actions = distribution.distribution.probs

        with th.no_grad():
            ground_truth_grass = get_ground_truth_of_grass(input=x)

        if self.alpha != 0 and self.use_learned_observations:
            if self.train_observations:
                if self.noisy_observations:
                    # use the first frame
                    grasses = self.observation_model.sigmoid(self.observation_model(x[:,0:1,:,:])[:, :3])
                else:
                    grasses = self.observation_model.sigmoid(self.observation_model(x[:,0:1,:,:])[:, :3])
                    grasses = (grasses > 0.5).float()
            else:
                with th.no_grad():
                    if self.noisy_observations:
                        grasses = self.observation_model.sigmoid(self.observation_model(x[:,0:1,:,:])[:, :3])
                    else:
                        grasses = self.observation_model.sigmoid(self.observation_model(x[:,0:1,:,:])[:, :3])
                        grasses = (grasses > 0.5).float()

        else:
            grasses = ground_truth_grass

        object_detect_probs = {
            "ground_truth_grass": ground_truth_grass,
            "grass": grasses
        }

        if self.alpha == 0: # PPO
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            object_detect_probs["alpha"] = 0
            results = self.query_safety_layer(
                x={
                    "grass": grasses,
                    "action": base_actions,
                }
            )
            policy_safety = results["safe_next"]
            object_detect_probs["policy_safety"] = policy_safety
            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions])

        if not self.differentiable_shield:  # VSRL
            if not self.vsrl_use_renormalization:
                if self.alpha != 0:
                    raise NotImplemented
                #====== VSRL rejection sampling =========
                num_rejected_samples = 0
                with th.no_grad():
                    while True:
                        actions = distribution.get_actions(deterministic=deterministic) # sample an action
                        # check if the action is safe
                        vsrl_actions_encoding = th.eye(self.n_actions)[actions][:, 1:]
                        actions_are_unsafe = th.logical_and(vsrl_actions_encoding, grasses)
                        if not th.any(actions_are_unsafe) or num_rejected_samples > self.max_num_rejected_samples:
                            break
                        else: # sample another action
                            num_rejected_samples += 1
                log_prob = distribution.log_prob(actions)
                object_detect_probs["num_rejected_samples"] = num_rejected_samples
                object_detect_probs["alpha"] = 1
                return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions])
            else:
                # ====== VSRL with mask =========
                with th.no_grad():
                    num_rejected_samples = 0
                    acc = th.ones((grasses.size()[0], 1)) # extra dimension for action "stay"
                    safety_left = ~th.logical_and(grasses.bool()[:, 1:2], ~grasses.bool()[:, 2:3])
                    safety_right = ~th.logical_and(~grasses.bool()[:, 1:2], grasses.bool()[:, 2:3])
                    safety_front = ~th.logical_or(~safety_left, ~safety_right)

                    mask = th.cat((acc, safety_front, acc, safety_left,safety_right), 1).bool()
                    masked_distr = distribution.distribution.probs * mask
                    safe_normalization_const = th.sum(masked_distr, dim=1,  keepdim=True)
                    safeast_actions = masked_distr / safe_normalization_const

                    alpha = self.alpha
                    actions = alpha * safeast_actions + (1 - alpha) * base_actions
                    try:
                        mass = Categorical(probs=actions)
                    except:
                        mass = Categorical(probs=actions)
                    if not deterministic:
                        actions = mass.sample()
                    else:
                        actions = th.argmax(mass.probs, dim=1)

                log_prob = distribution.log_prob(actions)
                object_detect_probs["num_rejected_samples"] = num_rejected_samples
                object_detect_probs["alpha"] = alpha

                results = self.query_safety_layer(
                    x={
                        "grass": grasses,
                        "action": base_actions,
                    }
                )
                policy_safety = results["safe_next"]
                object_detect_probs["policy_safety"] = policy_safety

                return (actions, values, log_prob, mass, [object_detect_probs, base_actions])

        if self.differentiable_shield: # PLS
            results = self.query_safety_layer(
                x={
                    "grass": grasses,
                    "action": base_actions,
                }
            )
            policy_safety = results["safe_next"]
            object_detect_probs["policy_safety"] = policy_safety

            acc = th.ones((grasses.size()[0], 1)) # extra dimension for action "stay"
            safety_left = 1 - (grasses[:, 1:2] * (1- grasses[:, 2:]))
            safety_right = 1 - ((1-grasses[:, 1:2]) * grasses[:, 2:])
            safety_front = 1 - ((1-safety_left)+(1-safety_right))


            safety_a = th.cat((acc, safety_front, acc, safety_left, safety_right), 1)
            safeast_actions = safety_a*base_actions/policy_safety

            alpha = self.alpha
            actions = alpha * safeast_actions + (1 - alpha) * base_actions

            # TODO: This is incorrect? This should be the shielded policy distribution (of type CategoricalDistribution)
            mass = Categorical(probs=actions)
            if not deterministic:
                actions = mass.sample()
            else:
                actions = th.argmax(mass.probs, dim=1)

            log_prob = mass.log_prob(actions)
            object_detect_probs["alpha"] = alpha

        return (actions, values, log_prob, mass, [object_detect_probs, base_actions])

    def evaluate_actions(
            self, x: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        if not self.differentiable_shield:
            obs = self.image_encoder(x)
            features = self.extract_features(obs)
            latent_pi, latent_vf = self.mlp_extractor(features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)
            base_actions = distribution.distribution.probs

            with th.no_grad():
                ground_truth_grass = get_ground_truth_of_grass(input=x)

            if self.alpha != 0 and self.use_learned_observations:
                if self.train_observations:
                    if self.noisy_observations:
                        grasses = self.observation_model.sigmoid(self.observation_model(x[:,0:1,:,:])[:, :3])
                    else:
                        grasses = self.observation_model.sigmoid(self.observation_model(x[:,0:1,:,:])[:, :3])
                        grasses = (grasses > 0.5).float()
                else:
                    with th.no_grad():
                        if self.noisy_observations:
                            grasses = self.observation_model.sigmoid(self.observation_model(x[:,0:1,:,:])[:, :3])
                        else:
                            grasses = self.observation_model.sigmoid(self.observation_model(x[:,0:1,:,:])[:, :3])
                            grasses = (grasses > 0.5).float()

            else:
                grasses = ground_truth_grass

            results = self.query_safety_layer(
                x={
                    "grass": grasses,
                    "action": base_actions,
                }
            )
            policy_safety = results["safe_next"]

            return values, log_prob, distribution.entropy(), policy_safety

        _, values, _, mass, [object_detect_probs, _] = self.forward(x)
        log_prob = mass.log_prob(actions)
        policy_safety = object_detect_probs["policy_safety"]
        return values, log_prob, mass.entropy(), policy_safety

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        with th.no_grad():
            _actions, values, log_prob, mass, _ = self.forward(observation, deterministic)
            return _actions