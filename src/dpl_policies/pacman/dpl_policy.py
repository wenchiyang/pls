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
from stable_baselines3.common.type_aliases import (
    GymObs,
    GymStepReturn,
    Schedule,
)


from src.deepproblog.light import DeepProbLogLayer, DeepProbLogLayer_Approx
from .util import get_ground_wall

WALL_COLOR = 0.25
GHOST_COLOR = 0.5
PACMAN_COLOR = 0.75
FOOD_COLOR = 1


class Pacman_Encoder(nn.Module):
    def __init__(self, input_size, n_actions, shielding_settings, program_path):
        super(Pacman_Encoder, self).__init__()
        self.input_size = input_size
        self.shield = shielding_settings["shield"]
        self.detect_ghosts = shielding_settings["detect_ghosts"]
        self.detect_walls = shielding_settings["detect_walls"]
        self.n_ghost_locs = shielding_settings["n_ghost_locs"]
        self.n_wall_locs = shielding_settings["n_wall_locs"]
        self.n_actions = n_actions
        self.program_path = program_path

    def forward(self, x):
        xx = th.flatten(x, 1)
        return xx

class Pacman_Callback(ConvertCallback):
    def __init__(self, callback):
        super(Pacman_Callback, self).__init__(callback)

class Pacman_Monitor(Monitor):
    def __init__(self, *args, program_path, **kwargs):
        super(Pacman_Monitor, self).__init__(*args, **kwargs)

    def reset(self, **kwargs) -> GymObs:
        return super(Pacman_Monitor, self).reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
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
                # "abs_safety_shielded": ep_abs_safety_shielded,
                # "abs_safety_base": ep_abs_safety_base
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
            info["is_success"] = True  # Idk what this should be
        self.total_steps += 1
        return observation, reward, done, info

class Pacman_DPLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            image_encoder: Pacman_Encoder = None,
            **kwargs
    ):
        super(Pacman_DPLActorCriticPolicy, self).__init__(observation_space,action_space, lr_schedule, **kwargs)
        ###############################

        self.image_encoder = image_encoder
        self.input_size = self.image_encoder.input_size
        self.shield = self.image_encoder.shield

        self.detect_ghosts = self.image_encoder.detect_ghosts
        self.detect_walls = self.image_encoder.detect_walls

        self.n_ghost_locs = self.image_encoder.n_ghost_locs
        self.n_wall_locs = self.image_encoder.n_wall_locs

        self.n_actions = self.image_encoder.n_actions
        self.program_path = self.image_encoder.program_path

        with open(self.program_path) as f:
            self.program = f.read()

        if self.detect_ghosts:
            self.ghost_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.n_ghost_locs),
                nn.Sigmoid(),  # TODO : add a flag
            )

        if self.detect_walls:
            self.wall_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.n_wall_locs),
                nn.Sigmoid(),
            )

        if self.shield:
            self.evidences = ["safe_next"]
            # IMPORTANT: THE ORDER OF QUERIES IS THE ORDER OF THE OUTPUT
            self.queries = [
                "safe_action(stay)",
                "safe_action(up)",
                "safe_action(down)",
                "safe_action(left)",
                "safe_action(right)",
            ][: self.n_actions]
            self.queries += [
                "ghost(up)",
                "ghost(down)",
                "ghost(left)",
                "ghost(right)",
                "wall(up)",
                "wall(down)",
                "wall(left)",
                "wall(right)"
            ]
            # self.dpl_layer = DeepProbLogLayer(
            #     program=self.program, queries=self.queries, evidences=self.evidences
            # )
            input_struct = {
                "ghost": [i for i in range(self.n_ghost_locs)],
                "wall": [i for i in
                         range(self.n_ghost_locs, self.n_ghost_locs + self.n_wall_locs)],
                "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
                                            self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
            query_struct = {
                "ghost": [i for i in range(self.n_ghost_locs)],
                "wall": [i for i in
                           range(self.n_ghost_locs, self.n_ghost_locs + self.n_wall_locs)],
                "safe_action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
                                                 self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
            self.dpl_layer = DeepProbLogLayer_Approx(
                program=self.program, queries=self.queries, evidences=self.evidences,
                input_struct=input_struct, query_struct=query_struct
            )

        debug_queries = ["safe_next"]
        # self.query_safety_layer = DeepProbLogLayer(
        #     program=self.program, queries=debug_queries
        # )
        input_struct = {
            "ghost": [i for i in range(self.n_ghost_locs)],
            "wall": [i for i in
                     range(self.n_ghost_locs, self.n_ghost_locs + self.n_wall_locs)],
            "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
                                        self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
        query_struct = {"safe_next": [i for i in range(1)]}
        self.query_safety_layer = DeepProbLogLayer_Approx(
            program=self.program, queries=debug_queries,
            input_struct=input_struct, query_struct=query_struct
        )
        self._build(lr_schedule)

    def logging(self, mass, object_detect_probs, base_policy, action_lookup, logger):
        for act in range(self.action_space.n):
            logger.record(
                f"policy/shielded {action_lookup[act]}",
                float(mass.probs[0][act]),
            )
        for direction in [0, 1, 2, 3]:  # TODO: order matters: use a map
            if "prob_ghost_prior" in object_detect_probs:
                logger.record(
                    f"prob/prob_ghost_prior_{direction}",
                    float(object_detect_probs["prob_ghost_prior"][0][direction]),
                )
            if "prob_wall_prior" in object_detect_probs:
                logger.record(
                    f"prob/prob_wall_prior_{direction}",
                    float(object_detect_probs["prob_wall_prior"][0][direction]),
                )
            if "prob_ghost_posterior" in object_detect_probs:
                logger.record(
                    f"prob/prob_ghost_posterior_{direction}",
                    float(object_detect_probs["prob_ghost_posterior"][0][direction]),
                )
                error_ghost_posterior = (
                    object_detect_probs["ground_truth_ghost"]
                    - object_detect_probs["prob_ghost_posterior"]
                ).abs()
                logger.record(
                    f"error/error_ghost_posterior_{direction}",
                    float(error_ghost_posterior[0][direction]),
                )
            if "prob_wall_posterior" in object_detect_probs:
                logger.record(
                    f"prob/prob_wall_posterior_{direction}",
                    float(object_detect_probs["prob_wall_posterior"][0][direction]),
                )
                error_wall_posterior = (
                    object_detect_probs["ground_truth_wall"]
                    - object_detect_probs["prob_wall_posterior"]
                ).abs()
                logger.record(
                    f"error/error_wall_posterior_{direction}",
                    float(error_wall_posterior[0][direction]),
                )

    def get_step_safety(self, policy_distribution, ghost_probs, wall_probs):
        with th.no_grad():
            abs_safe_next = self.query_safety_layer(
                x={
                    "ghost": ghost_probs,
                    "wall": wall_probs,
                    "action": policy_distribution,
                }
            )
            return abs_safe_next["safe_next"]

    def evaluate_safety_shielded(self, obs: th.Tensor):
        with th.no_grad():
            _, _, _, mass, (object_detect_probs, base_policy) = self.forward(obs)
            return self.get_step_safety(
                mass.probs,
                object_detect_probs["ground_truth_ghost"],
                object_detect_probs["ground_truth_wall"],
            )

    def forward(self, x, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        obs = self.image_encoder(x)
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(
            latent_pi, latent_sde=latent_sde
        )
        # NO Shielding
        if not self.shield:
            return self.no_shielding(distribution, values, x, deterministic)

        # SOFT Shielding
        elif self.shield and self.detect_ghosts and self.detect_walls:
            return self.soft_shielding(distribution, values, obs, x)

        # HARD Shielding
        elif self.shield and not self.detect_ghosts and not self.detect_walls:
            return self.hard_shielding(distribution, values, obs, x)

    def no_shielding(self, distribution, values, x, deterministic):
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        with th.no_grad():
            ground_truth_ghost = get_ground_wall(x, PACMAN_COLOR, GHOST_COLOR)
            ground_truth_wall = get_ground_wall(x, PACMAN_COLOR, WALL_COLOR)

            base_actions = distribution.distribution.probs

            object_detect_probs = {
                "ground_truth_ghost": ground_truth_ghost,
                "ground_truth_wall": ground_truth_wall
            }

        return (
            actions,
            values,
            log_prob,
            distribution.distribution,
            [object_detect_probs, base_actions],
        )

    def soft_shielding(self, distribution, values, obs, x):
        ghosts = self.ghost_layer(obs)
        walls = self.wall_layer(obs)

        base_actions = distribution.distribution.probs
        results = self.dpl_layer(
            x={
                "ghost": ghosts,
                "wall": walls,
                "action": base_actions,
            }
        )
        # For when we set safe_next as a query
        # safe_next = results["safe_next"]
        # safe_actions = results["safe_action"] / safe_next
        # # When safe_next is zero (meaning there are no safe actions), use base_actions
        # actions = th.where(abs(safe_next) < 1e-6, base_actions, safe_actions)

        # For when we set safe_action=true as evidence
        actions = results["safe_action"]

        mass = Categorical(probs=actions)
        actions = mass.sample()
        log_prob = mass.log_prob(actions)

        with th.no_grad():
            ground_truth_ghost = get_ground_wall(x, PACMAN_COLOR, GHOST_COLOR)
            ground_truth_wall = get_ground_wall(x, PACMAN_COLOR, WALL_COLOR)

            object_detect_probs = {
                "prob_ghost_prior": ghosts,
                "prob_wall_prior": walls,
                "prob_ghost_posterior": results["ghost"],
                "prob_wall_posterior": results["wall"],
                "ground_truth_ghost": ground_truth_ghost,
                "ground_truth_wall": ground_truth_wall,
            }

        return (actions, values, log_prob, mass, [object_detect_probs, base_actions])

    def hard_shielding(self, distribution, values, obs, x):
        with th.no_grad():
            ground_truth_ghost = get_ground_wall(x, PACMAN_COLOR, GHOST_COLOR)
            ground_truth_wall = get_ground_wall(x, PACMAN_COLOR, WALL_COLOR)

            ghosts = ground_truth_ghost
            walls = ground_truth_wall

        base_actions = distribution.distribution.probs
        results = self.dpl_layer(
            x={
                "ghost": ghosts,
                "wall": walls,
                "action": base_actions,
            }
        )
        # For when we set safe_next as a query
        # safe_next = results["safe_next"]
        # safe_actions = results["safe_action"] / safe_next
        # # When safe_next is zero (meaning there are no safe actions), use base_actions
        # actions = th.where(abs(safe_next) < 1e-6, base_actions, safe_actions)

        # For when we set safe_action=true as evidence
        actions = results["safe_action"]

        mass = Categorical(probs=actions)
        # if random.random() < 0.5:
        actions = mass.sample()
        log_prob = mass.log_prob(actions)
        # else:
        #     actions = distribution.distribution.sample()
        #     log_prob = distribution.distribution.log_prob(actions)
        with th.no_grad():
            object_detect_probs = {
                "prob_ghost_prior": ghosts,
                "prob_wall_prior": walls,
                "prob_ghost_posterior": results["ghost"],
                "prob_wall_posterior": results["wall"],
                "ground_truth_ghost": ground_truth_ghost,
                "ground_truth_wall": ground_truth_wall,
            }

        return (actions, values, log_prob, mass, [object_detect_probs, base_actions])

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        _actions, values, log_prob, mass, _ = self.forward(obs)

        log_prob = mass.log_prob(actions)
        return values, log_prob, mass.entropy()