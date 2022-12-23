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
from random import random

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
        shielded_policy = self.locals["shielded_policy"]
        action_lookup = self.locals["action_lookup"]
        object_detect_probs = self.locals["object_detect_probs"]
        base_policy = self.locals["base_policy"]
        policy = self.locals["self"].policy
        for act in range(self.locals["self"].action_space.n):
            logger.record(
                f"policy/shielded {action_lookup[act]}",
                float(shielded_policy[0][act]),
            )
        if object_detect_probs.get("alpha") is not None:
            logger.record(
                f"safety/alpha",
                float(object_detect_probs.get("alpha")),
            )
        abs_safe_next_shielded = policy.get_step_safety(
            shielded_policy,
            object_detect_probs["ground_truth_grass"],
        )
        abs_safe_next_base = policy.get_step_safety(
            base_policy,
            object_detect_probs["ground_truth_grass"],
        )
        rel_safe_next_shielded = policy.get_step_safety(
            shielded_policy,
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
    def __init__(self, *args, vio_len, stochasticity, **kwargs):
        self.vio_len = vio_len
        super(Carracing_Monitor, self).__init__(*args, **kwargs)
        self.stochasticity = stochasticity

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
                "violate_constraint": self.max_cont_in_grass_len > self.vio_len or info["out_of_field"],
                # "violate_constraint": violate_constraint,
                "is_success": info["is_success"],
                "max_cont_in_grass_len": self.max_cont_in_grass_len,
                "max_cont_violate_len": self.max_cont_violate_len,
                "total_violate_len": self.total_violate_len,
                "out_of_field": info["out_of_field"]
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

        if not self.differentiable_shield and self.alpha > 0:
            self.vsrl_eps = shielding_params["vsrl_eps"] if "vsrl_eps" in shielding_params else 0
        if self.program_path:
            # pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments_safety/carracing/data/carracing_grass4.pl")
            # self.program_path = pp
            with open(self.program_path) as f:
                self.program = f.read()


        self.use_learned_observations = shielding_params["use_learned_observations"]
        if not self.use_learned_observations: # Baseline
            pass
        elif self.use_learned_observations:
            self.train_observations = shielding_params["train_observations"] if self.use_learned_observations else None
            self.noisy_observations = shielding_params["noisy_observations"] if self.use_learned_observations else None
            self.observation_type = shielding_params["observation_type"] if self.use_learned_observations else None
            use_cuda = False
            device = th.device("cuda" if use_cuda else "cpu")
            self.observation_model = Observation_Net_Carracing(input_size=self.net_input_dim*self.net_input_dim, output_size=3).to(device)
            pp = path.join(self.folder, "../../data", self.observation_type)
            # pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments5/goal_finding_sto/small2/data", self.observation_type)
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
            return pickle.load(open(cache_path, "rb"))

        layer = DeepProbLogLayer_Approx(
            program=program, queries=queries, evidences=evidences,
            input_struct=input_struct, query_struct=query_struct
        )
        pickle.dump(layer, open(cache_path, "wb"))
        return layer

    def get_step_safety(self, policy_distribution, grass_probs):
        with th.no_grad():
            return self.get_policy_safety(grass_probs, policy_distribution)

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


        if self.alpha == 0:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            object_detect_probs["alpha"] = 0
            policy_safety = self.get_policy_safety(grasses, base_actions)
            object_detect_probs["policy_safety"] = policy_safety

            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions, base_actions])

        if not self.differentiable_shield:  # VSRL
            # ====== VSRL with mask =========
            with th.no_grad():
                num_rejected_samples = 0

                rn = random()
                if rn >= self.vsrl_eps:
                    action_safeties = self.get_action_safeties(grasses)
                    mask = action_safeties > 0.5
                else:
                    mask = th.ones(( grasses.size(0), 5))
                safeast_actions = base_actions * mask / th.sum(base_actions * mask, dim=1,  keepdim=True)

                alpha = self.alpha
                actions = alpha * safeast_actions + (1 - alpha) * base_actions

                shielded_policy = Categorical(probs=actions)
                if not deterministic:
                    actions = shielded_policy.sample()
                else:
                    actions = th.argmax(shielded_policy.probs, dim=1)

            log_prob = distribution.log_prob(actions)
            object_detect_probs["num_rejected_samples"] = num_rejected_samples
            object_detect_probs["alpha"] = alpha

            policy_safety = self.get_policy_safety(grasses, base_actions)
            object_detect_probs["policy_safety"] = policy_safety

            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions, shielded_policy.probs])

        if self.differentiable_shield: # PLS
            policy_safety = self.get_policy_safety(grasses, base_actions)
            object_detect_probs["policy_safety"] = policy_safety

            action_safeties = self.get_action_safeties(grasses)
            safeast_actions = action_safeties * base_actions / policy_safety

            assert(safeast_actions.max() <=  1.00001), f"{safeast_actions} violates MAX"
            assert(safeast_actions.min() >= -0.00001), f"{safeast_actions} violates MIN"

            alpha = self.alpha
            actions = alpha * safeast_actions + (1 - alpha) * base_actions

            # TODO: This is incorrect? This should be the shielded policy distribution (of type CategoricalDistribution)
            shielded_policy = Categorical(probs=actions)
            if not deterministic:
                actions = shielded_policy.sample()
            else:
                actions = th.argmax(shielded_policy.probs, dim=1)

            log_prob = shielded_policy.log_prob(actions)
            object_detect_probs["alpha"] = alpha

        return (actions, values, log_prob, shielded_policy, [object_detect_probs, base_actions, shielded_policy.probs])



    def get_policy_safety(self, grasses, base_actions):
        results = self.query_safety_layer(
            x={
                "grass": grasses,
                "action": base_actions,
            }
        )
        policy_safety = results["safe_next"]
        return policy_safety

    def get_action_safeties(self, grasses):
        all_actions = th.eye(5).unsqueeze(1)
        action_safeties = []
        for action in all_actions:
            base_actions = th.repeat_interleave(action, grasses.size(0), dim=0)
            results = self.query_safety_layer(
                x={
                    "grass": grasses,
                    "action": base_actions,
                }
            )
            action_safety = results["safe_next"]
            action_safeties.append(action_safety)
        action_safeties = th.cat(action_safeties, dim=1)
        return action_safeties

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

        _, values, _, mass, [object_detect_probs, _, _] = self.forward(x)
        log_prob = mass.log_prob(actions)
        policy_safety = object_detect_probs["policy_safety"]
        return values, log_prob, mass.entropy(), policy_safety

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        with th.no_grad():
            _actions, values, log_prob, mass, _ = self.forward(observation, deterministic)
            return _actions