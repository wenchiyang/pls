import gym
import torch as th
from torch import nn
import numpy as np
from typing import Union, Tuple, Dict, Optional
from torch.distributions import Categorical
import time
from stable_baselines3.common.callbacks import ConvertCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import (
    GymObs,
    GymStepReturn,
    Schedule,
)

from pls.deepproblog.light import DeepProbLogLayer, DeepProbLogLayer_Approx
from pls.dpl_policies.sokoban.util import get_ground_truth_of_box, get_ground_truth_of_corners, stuck
from os import path
import pickle
from gym.spaces import Box

from pls.observation_nets.observation_nets import Observation_net

WALL_COLOR = th.tensor([0], dtype=th.float32)
FLOOR_COLOR = th.tensor([1 / 6], dtype=th.float32)
BOX_TARGET_COLOR = th.tensor([2 / 6], dtype=th.float32)
BOX_ON_TARGET_COLOR = th.tensor([3 / 6], dtype=th.float32)
BOX_COLOR = th.tensor([4 / 6], dtype=th.float32)

PLAYER_COLOR = th.tensor([5 / 6], dtype=th.float32)
PLAYER_ON_TARGET_COLOR = th.tensor([1], dtype=th.float32)

PLAYER_COLORS = th.tensor(([5 / 6], [1]))
BOX_COLORS = th.tensor(([3 / 6], [4 / 6]))
OBSTABLE_COLORS = th.tensor(([0], [3 / 6], [4 / 6]))


NEIGHBORS_RELATIVE_LOCS_BOX = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),

]  # DO NOT CHANGE THE ORDER: up, down, left, right,
NEIGHBORS_RELATIVE_LOCS_CORNER = [
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
]  # DO NOT CHANGE THE ORDER

def box_stuck_in_corner(tinygrid):
    s = stuck(
        input=tinygrid,
        box_color=BOX_COLOR,
        obsacle_colors=OBSTABLE_COLORS
    )
    return s

class Sokoban_Encoder(nn.Module):
    def __init__(self):
        super(Sokoban_Encoder, self).__init__()

    def forward(self, x):
        xx = th.flatten(x, 1)
        return xx


class Sokoban_Callback(ConvertCallback):
    def __init__(self, callback):
        super(Sokoban_Callback, self).__init__(callback)
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
            object_detect_probs["ground_truth_box"],
            object_detect_probs["ground_truth_corner"],
        )
        abs_safe_next_base = policy.get_step_safety(
            base_policy,
            object_detect_probs["ground_truth_box"],
            object_detect_probs["ground_truth_corner"],
        )
        rel_safe_next_shielded = policy.get_step_safety(
            mass.probs,
            object_detect_probs["box"],
            object_detect_probs["corner"],
        )
        rel_safe_next_base = policy.get_step_safety(
            base_policy,
            object_detect_probs["box"],
            object_detect_probs["corner"],
        )

        self.locals["abs_safeties_shielded"].append(abs_safe_next_shielded)
        self.locals["abs_safeties_base"].append(abs_safe_next_base)
        self.locals["rel_safeties_shielded"].append(rel_safe_next_shielded)
        self.locals["rel_safeties_base"].append(rel_safe_next_base)

        if object_detect_probs.get("alpha") is not None:
            self.locals["alphas"].append(object_detect_probs["alpha"])
        if object_detect_probs.get("num_rejected_samples") is not None:
            self.locals["nums_rejected_samples"].append(object_detect_probs["num_rejected_samples"])
        # if in a risky situation
        if th.any(th.logical_and(object_detect_probs["ground_truth_box"], object_detect_probs["ground_truth_corner"])):
            self.locals["n_risky_states"].append(1)
        else:
            self.locals["n_risky_states"].append(0)

class Sokoban_Monitor(Monitor):
    def __init__(self, *args, **kwargs):
        super(Sokoban_Monitor, self).__init__(*args, **kwargs)

    def reset(self, **kwargs) -> GymObs:
        return super(Sokoban_Monitor, self).reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)

            # violate_constraint = box_stuck_in_corner(tinygrid)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
                "last_r": reward,
                "violate_constraint": info["box_at_corner"],
                "is_success": info["all_boxes_on_target"]
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
            # info["is_success"] = True  # Idk what this should be
        self.total_steps += 1
        return observation, reward, done, info




class Sokoban_DPLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        image_encoder: Sokoban_Encoder = None,
        shielding_params = None,
        net_input_dim = 1,
        folder = None,
        **kwargs
    ):
        observation_space = Box(
            low=-1,
            high=1,
            shape=(
                net_input_dim, net_input_dim
            )
        )
        super(Sokoban_DPLActorCriticPolicy, self).__init__(observation_space,action_space, lr_schedule, **kwargs)
        ###############################

        self.image_encoder = image_encoder
        self.n_box_locs = shielding_params["n_box_locs"]
        self.n_corner_locs = shielding_params["n_corner_locs"]
        self.alpha = shielding_params["alpha"]
        self.differentiable_shield = shielding_params["differentiable_shield"]
        self.net_input_dim = net_input_dim
        self.n_actions = 5
        self.tinygrid_dim = shielding_params["tinygrid_dim"]
        self.folder = path.join(path.dirname(__file__), "../../..", folder)
        self.program_path = path.join(self.folder, "../../../data", shielding_params["program_type"]+".pl")


        if self.program_path:
            self.program_path = path.join("/Users/wenchi/PycharmProjects/pls/experiments_trials3/sokoban/data/sokoban_corner2.pl")
            with open(self.program_path) as f:
                self.program = f.read()

        # if self.detect_boxes:
        #     self.box_layer = nn.Sequential(
        #         nn.Linear(self.input_size, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, self.n_box_locs),
        #         nn.Sigmoid(),  # TODO : add a flag
        #     )
        #
        # if self.detect_corners:
        #     self.corner_layer = nn.Sequential(
        #         nn.Linear(self.input_size, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, self.n_corner_locs),
        #         nn.Sigmoid(),
        #     )

        if self.alpha == 0: # Baseline
            pass
        elif self.differentiable_shield: # PLS
            self.use_learned_observations = shielding_params["use_learned_observations"]
            self.noisy_observations = shielding_params["noisy_observations"] if self.use_learned_observations else None
            self.observation_type = shielding_params["observation_type"] if self.use_learned_observations else None
        else: # VSRL
            self.vsrl_use_renormalization = shielding_params["vsrl_use_renormalization"]
            if not self.vsrl_use_renormalization:
                self.max_num_rejected_samples = shielding_params["max_num_rejected_samples"]
            self.use_learned_observations = shielding_params["use_learned_observations"]
            self.noisy_observations = shielding_params["noisy_observations"] if self.use_learned_observations else None
            self.observation_type = shielding_params["observation_type"] if self.use_learned_observations else None



        if self.alpha == 0: # NO shielding
            pass
        else: # HARD shielding and SOFT shielding
            # IMPORTANT: THE ORDER OF QUERIES IS THE ORDER OF THE OUTPUT
            self.queries = ["safe_action(no_op)", "safe_action(push_up)", "safe_action(push_down)",
                            "safe_action(push_left)", "safe_action(push_right)", "safe_action(move_up)",
                            "safe_action(move_down)", "safe_action(move_left)", "safe_action(move_right)"
                           ][: self.n_actions]
            input_struct = {
                "box": [i for i in range(self.n_box_locs)],
                "corner": [i for i in range(self.n_box_locs, self.n_box_locs + self.n_corner_locs)],
                "action": [i for i in range(self.n_box_locs + self.n_corner_locs,
                                            self.n_box_locs + self.n_corner_locs + self.n_actions)],
            }
            query_struct = {"safe_action": { "no_op": 0, "push_up": 1,
                                             "push_down": 2, "push_left": 3, "push_right": 4}}
            pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments_trials3/sokoban/data/dpl_layer.p")
            self.dpl_layer = self.get_layer(
                pp, #path.join(self.folder, "../../../data", "dpl_layer.p"),
                program=self.program, queries=self.queries, evidences=["safe_next"],
                input_struct=input_struct, query_struct=query_struct
            )
            if self.use_learned_observations:
                # TODO
                observation_model_path = path.join(self.folder, "../../data", self.observation_type)
                observation_model_path = path.join(
                    "/experiments_trials3/sokoban/2box5map_gray3/data/observation_model_10000_examples.pt")
                use_cuda = False
                device = th.device("cuda" if use_cuda else "cpu")
                self.observation_model = Observation_net(input_size=self.input_size*self.input_size, output_size=8).to(device)
                self.observation_model.load_state_dict(th.load(observation_model_path))

        debug_queries = ["safe_next"]
        debug_query_struct = {"safe_next": 0}
        debug_input_struct = {
            "box": [i for i in range(self.n_box_locs)],
            "corner": [i for i in range(self.n_box_locs, self.n_box_locs + self.n_corner_locs)],
            "action": [i for i in range(self.n_box_locs + self.n_corner_locs,
                                        self.n_box_locs + self.n_corner_locs + self.n_actions)]
        }
        pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments_trials3/sokoban/data/query_safety_layer.p")
        self.query_safety_layer = self.get_layer(
            pp, #path.join(self.folder, "../../../data", "query_safety_layer.p"),
            program=self.program, queries=debug_queries, evidences=[],
            input_struct=debug_input_struct, query_struct=debug_query_struct)

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

    def get_step_safety(self, policy_distribution, box_probs, corner_probs):
        with th.no_grad():
            abs_safe_next = self.query_safety_layer(
                x={
                    "box": box_probs,
                    "corner": corner_probs,
                    "action": policy_distribution,
                }
            )
            return abs_safe_next["safe_next"]


    def forward(self, x, tinygrid, deterministic: bool = False):
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
            ground_truth_box = get_ground_truth_of_box(
                input=tinygrid, agent_colors=PLAYER_COLORS, box_colors=BOX_COLORS,
            ) if tinygrid is not None else None
            ground_truth_corner = get_ground_truth_of_corners(
                input=tinygrid, agent_colors=PLAYER_COLORS, obsacle_colors=OBSTABLE_COLORS, floor_color=FLOOR_COLOR,
            ) if tinygrid is not None else None

        if self.alpha != 0 and self.use_learned_observations:
            # TODO
            output = self.observation_model(x)
            if self.noisy_observations:
                boxes_and_corners = self.observation_model.sigmoid(output)
                boxes = boxes_and_corners[:, :self.n_box_locs]
                corners = boxes_and_corners[:, self.n_box_locs:]
            else:
                boxes_and_corners = (self.observation_model.sigmoid(output) > 0.5).float()
                boxes = boxes_and_corners[:, :self.n_box_locs]
                corners = boxes_and_corners[:, self.n_box_locs:]
        else:
            boxes = ground_truth_box
            corners = ground_truth_corner

        object_detect_probs = {
            "ground_truth_box": ground_truth_box,
            "ground_truth_corner": ground_truth_corner,
            "box": boxes,
            "corner": corners
        }

        if self.alpha == 0: # PPO
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            object_detect_probs["alpha"] = 0
            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions])

        if not self.differentiable_shield: # VSRL
            if not self.vsrl_use_renormalization:
                if self.alpha != 0:
                    raise NotImplemented
                #====== VSRL rejection sampling =========
                num_rejected_samples = 0
                with th.no_grad():
                    while True:
                        actions = distribution.get_actions(deterministic=deterministic) # sample an action
                        # check if the action is safe
                        vsrl_actions_encoding = th.eye(self.n_actions)[actions][:, 1:] # TODO
                        risky_actions = th.logical_and(object_detect_probs["ground_truth_box"], object_detect_probs["ground_truth_corner"])
                        actions_are_unsafe = th.logical_and(vsrl_actions_encoding, risky_actions)
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
                    risky_actions = th.logical_and(object_detect_probs["ground_truth_box"], object_detect_probs["ground_truth_corner"])
                    acc = th.ones((risky_actions.size()[0], 1)) # extra dimension for action "stay"
                    mask = th.cat((acc, ~risky_actions.bool()), 1).bool()
                    masked_distr = distribution.distribution.probs * mask
                    safe_normalization_const = th.sum(masked_distr, dim=1)
                    safeast_actions = masked_distr / safe_normalization_const

                    alpha = self.alpha
                    actions = alpha * safeast_actions + (1 - alpha) * base_actions

                    mass = Categorical(probs=actions)
                    if not deterministic:
                        actions = mass.sample()
                    else:
                        actions = th.argmax(mass.probs, dim=1)
                log_prob = distribution.log_prob(actions)
                object_detect_probs["num_rejected_samples"] = num_rejected_samples
                object_detect_probs["alpha"] = alpha
                return (actions, values, log_prob, mass, [object_detect_probs, base_actions])

        if self.differentiable_shield:
            results = self.dpl_layer(
                x={
                    "box": boxes,
                    "corner": corners,
                    "action": base_actions,
                }
            )
            safeast_actions = results["safe_action"]

            alpha = self.alpha
            actions = alpha * safeast_actions + (1 - alpha) * base_actions

            # TODO: This is incorrect? This should be the shielded policy distribution (of type CategoricalDistribution)
            mass = Categorical(probs=actions)
            if not deterministic:
                actions = mass.sample()
            else:
                actions = th.argmax(mass.probs,dim=1)

            log_prob = mass.log_prob(actions)
            object_detect_probs["alpha"] = alpha
            return (actions, values, log_prob, mass, [object_detect_probs, base_actions])


    def evaluate_actions(
        self, obs: th.Tensor, tinygrid: th.Tensor, actions: th.Tensor
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
            obs = self.image_encoder(obs)
            features = self.extract_features(obs)
            latent_pi, latent_vf = self.mlp_extractor(features)
            distribution = self._get_action_dist_from_latent(latent_pi)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)
            return values, log_prob, distribution.entropy()

        _, values, _, mass, _ = self.forward(obs, tinygrid=tinygrid)
        log_prob = mass.log_prob(actions)
        return values, log_prob, mass.entropy()

    def _predict(self, observation: th.Tensor, deterministic: bool = False, tinygrid = None) -> th.Tensor:
        with th.no_grad():
            _actions, _, _, _, _  = self.forward(observation, tinygrid, deterministic)
            return _actions

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, tinygrid=state, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]

        return actions
