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

from pls.deepproblog.light import DeepProbLogLayer_Approx
from pls.dpl_policies.sokoban.util import get_ground_truth_of_box, get_ground_truth_of_corners
from os import path
import pickle
from gym.spaces import Box

from pls.observation_nets.observation_nets import Observation_Net_Sokoban
from random import random


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
            object_detect_probs["ground_truth_box"],
            object_detect_probs["ground_truth_corner"],
        )
        abs_safe_next_base = policy.get_step_safety(
            base_policy,
            object_detect_probs["ground_truth_box"],
            object_detect_probs["ground_truth_corner"],
        )
        rel_safe_next_shielded = policy.get_step_safety(
            shielded_policy,
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
        # if is in a risky situation
        if th.any(th.logical_and(object_detect_probs["ground_truth_box"], object_detect_probs["ground_truth_corner"])):
            self.locals["n_risky_states"].append(1)
        else:
            self.locals["n_risky_states"].append(0)

class Sokoban_Monitor(Monitor):
    def __init__(self, *args, stochasticity, **kwargs):
        super(Sokoban_Monitor, self).__init__(*args, **kwargs)
        self.stochasticity = stochasticity

    def reset(self, **kwargs) -> GymObs:
        return super(Sokoban_Monitor, self).reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        rdm = random()
        if rdm >= 2*self.stochasticity:
            action = [0, 1, 2, 3, 4][action]
        elif rdm >= self.stochasticity:
            action = [0, 3, 3, 1, 1][action]
        else:
            action = [0, 4, 4, 2, 2][action]

        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)

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

        if not self.differentiable_shield and self.alpha > 0:
            self.vsrl_eps = shielding_params["vsrl_eps"] if "vsrl_eps" in shielding_params else 0
        if self.program_path:
            # pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments_trials3/sokoban/data/sokoban_corner2.pl")
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
            self.observation_model = Observation_Net_Sokoban(input_size=self.net_input_dim * self.net_input_dim, output_size=8).to(device)
            pp = path.join(self.folder, "../../data", self.observation_type)
            # pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments5/goal_finding_sto/small2/data", self.observation_type)
            self.observation_model.load_state_dict(th.load(pp))

        debug_queries = ["safe_next"]
        debug_query_struct = {"safe_next": 0}
        debug_input_struct = {
            "box": [i for i in range(self.n_box_locs)],
            "corner": [i for i in range(self.n_box_locs, self.n_box_locs + self.n_corner_locs)],
            "action": [i for i in range(self.n_box_locs + self.n_corner_locs,
                                        self.n_box_locs + self.n_corner_locs + self.n_actions)]
        }
        pp = path.join(self.folder, "../../../data", "query_safety_layer.p")
        # pp = path.join("/Users/wenchi/PycharmProjects/pls/experiments5/goal_finding_sto/data/query_safety_layer.p")
        self.query_safety_layer = self.get_layer(
            pp, program=self.program, queries=debug_queries, evidences=[],
            input_struct=debug_input_struct, query_struct=debug_query_struct
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

    def get_step_safety(self, policy_distribution, boxes, corners):
        with th.no_grad():
            return self.get_policy_safety(boxes, corners, policy_distribution)


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


        if self.use_learned_observations:
            if self.train_observations:
                if self.noisy_observations:
                    boxes_and_corners = self.observation_model.sigmoid(self.observation_model(x.unsqueeze(1))[:, :8])
                else:
                    boxes_and_corners = self.observation_model.sigmoid(self.observation_model(x.unsqueeze(1))[:, :8])
                    boxes_and_corners = (boxes_and_corners > 0.5).float()
            else:
                with th.no_grad():
                    if self.noisy_observations:
                        boxes_and_corners = self.observation_model.sigmoid(self.observation_model(x.unsqueeze(1))[:, :8])
                    else:
                        boxes_and_corners = self.observation_model.sigmoid(self.observation_model(x.unsqueeze(1))[:, :8])
                        boxes_and_corners = (boxes_and_corners > 0.5).float()
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

        if self.alpha == 0:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            object_detect_probs["alpha"] = 0
            policy_safety = self.get_policy_safety(boxes, corners, base_actions)
            object_detect_probs["policy_safety"] = policy_safety

            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions, base_actions])

        if not self.differentiable_shield: # VSRL
            with th.no_grad():
                num_rejected_samples = 0

                rn = random()
                if rn >= self.vsrl_eps:
                    action_safeties = self.get_action_safeties(boxes, corners)
                    mask = action_safeties > 0.5
                else:
                    mask = th.ones(( boxes.size(0), 5))
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

            policy_safety = self.get_policy_safety(boxes, corners, base_actions)
            object_detect_probs["policy_safety"] = policy_safety

            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions, shielded_policy.probs])

        if self.differentiable_shield: # PLS
            policy_safety = self.get_policy_safety(boxes, corners, base_actions)
            object_detect_probs["policy_safety"] = policy_safety

            action_safeties = self.get_action_safeties(boxes, corners)
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

    def get_policy_safety(self, boxes, corners, base_actions):
        results = self.query_safety_layer(
            x={
                "box": boxes,
                "corner": corners,
                "action": base_actions,
            }
        )
        policy_safety = results["safe_next"]
        return policy_safety

    def get_action_safeties(self, boxes, corners):
        all_actions = th.eye(5).unsqueeze(1)
        action_safeties = []
        for action in all_actions:
            base_actions = th.repeat_interleave(action, boxes.size(0), dim=0)
            results = self.query_safety_layer(
                x={
                    "box": boxes,
                    "corner": corners,
                    "action": base_actions,
                }
            )
            action_safety = results["safe_next"]
            action_safeties.append(action_safety)
        action_safeties = th.cat(action_safeties, dim=1)
        return action_safeties

    def evaluate_actions(
        self, x: th.Tensor, tinygrid: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        _, values, _, mass, [object_detect_probs, _, _] = self.forward(x, tinygrid=tinygrid)
        log_prob = mass.log_prob(actions)
        policy_safety = object_detect_probs["policy_safety"]
        return values, log_prob, mass.entropy(), policy_safety

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
