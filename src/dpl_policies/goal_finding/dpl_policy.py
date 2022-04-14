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
from os import path
import pickle

from deepproblog.light import DeepProbLogLayer, DeepProbLogLayer_Approx
from .util import get_ground_wall

WALL_COLOR = 0.25
GHOST_COLOR = 0.5
PACMAN_COLOR = 0.75
FOOD_COLOR = 1


class GoalFinding_Encoder(nn.Module):
    def __init__(self, input_size, n_actions, shielding_settings, program_path, debug_program_path, folder):
        super(GoalFinding_Encoder, self).__init__()
        self.input_size = input_size
        self.n_ghost_locs = shielding_settings["n_ghost_locs"]
        self.n_actions = n_actions
        self.program_path = program_path
        self.debug_program_path = debug_program_path
        self.folder = folder
        self.sensor_noise = shielding_settings["sensor_noise"]


    def forward(self, x):
        xx = th.flatten(x, 1)
        # xx = xx + (self.sensor_noise**0.5)*th.randn(xx.shape)
        return xx

class GoalFinding_Callback(ConvertCallback):
    def __init__(self, callback):
        super(GoalFinding_Callback, self).__init__(callback)

class GoalFinding_Monitor(Monitor):
    def __init__(self, *args, **kwargs):
        super(GoalFinding_Monitor, self).__init__(*args, **kwargs)

    def reset(self, **kwargs) -> GymObs:
        return super(GoalFinding_Monitor, self).reset(**kwargs)

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
                "violate_constraint": reward < -1,
                "is_success": info["is_success"]
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

class GoalFinding_DPLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            image_encoder: GoalFinding_Encoder = None,
            alpha = 0.5,
            differentiable_shield = True,
            **kwargs
    ):
        super(GoalFinding_DPLActorCriticPolicy, self).__init__(observation_space,action_space, lr_schedule, **kwargs)
        ###############################

        self.image_encoder = image_encoder
        self.input_size = self.image_encoder.input_size

        self.n_ghost_locs = self.image_encoder.n_ghost_locs

        self.n_actions = self.image_encoder.n_actions
        self.program_path = self.image_encoder.program_path
        self.debug_program_path = self.image_encoder.debug_program_path
        self.folder = self.image_encoder.folder
        self.sensor_noise = self.image_encoder.sensor_noise
        self.alpha = alpha
        self.differentiable_shield = differentiable_shield

        with open(self.program_path) as f:
            self.program = f.read()
        with open(self.debug_program_path) as f:
            self.debug_program = f.read()

        # if self.detect_ghosts:
        #     self.ghost_layer = nn.Sequential(
        #         nn.Linear(self.input_size, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, self.n_ghost_locs),
        #         nn.Sigmoid(),  # TODO : add a flag
        #     )
        #
        # if self.detect_walls:
        #     self.wall_layer = nn.Sequential(
        #         nn.Linear(self.input_size, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, self.n_wall_locs),
        #         nn.Sigmoid(),
        #     )

        # if self.shield:
        #     self.evidences = ["safe_next"]
        #     # IMPORTANT: THE ORDER OF QUERIES IS THE ORDER OF THE OUTPUT
        #     self.queries = [
        #         "safe_action(stay)",
        #         "safe_action(up)",
        #         "safe_action(down)",
        #         "safe_action(left)",
        #         "safe_action(right)",
        #     ][: self.n_actions]
        #
        #     if self.detect_ghosts and self.detect_walls:
        #         self.queries += [
        #             "ghost(up)",
        #             "ghost(down)",
        #             "ghost(left)",
        #             "ghost(right)",
        #             "wall(up)",
        #             "wall(down)",
        #             "wall(left)",
        #             "wall(right)"
        #         ]
        #         input_struct = {
        #             "ghost": [i for i in range(self.n_ghost_locs)],
        #             "wall": [i for i in
        #                      range(self.n_ghost_locs, self.n_ghost_locs + self.n_wall_locs)],
        #             "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                         self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
        #         query_struct = {
        #             "ghost": [i for i in range(self.n_ghost_locs)],
        #             "wall": [i for i in
        #                      range(self.n_ghost_locs, self.n_ghost_locs + self.n_wall_locs)],
        #             "safe_action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                              self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
        #         self.dpl_layer = DeepProbLogLayer_Approx(
        #             program=self.program, queries=self.queries, evidences=self.evidences,
        #             input_struct=input_struct, query_struct=query_struct
        #         )
        #     elif self.detect_ghosts:
        #         self.queries += [
        #             "ghost(up)",
        #             "ghost(down)",
        #             "ghost(left)",
        #             "ghost(right)",
        #         ]
        #
        #         input_struct = {
        #             "ghost": [i for i in range(self.n_ghost_locs)],
        #             "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                         self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
        #         query_struct = {
        #             "ghost": [i for i in range(self.n_ghost_locs)],
        #             "safe_action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                              self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
        #         self.dpl_layer = DeepProbLogLayer_Approx(
        #             program=self.program, queries=self.queries,  evidences=self.evidences,
        #             input_struct=input_struct, query_struct=query_struct
        #         )
        #     else:
        #         if hasattr(self, 'free_actions') and self.free_actions:
        #             ## TODO more hard
        #             input_struct = {
        #                 "ghost": [i for i in range(self.n_ghost_locs)],
        #                 "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                             self.n_ghost_locs + self.n_wall_locs + self.n_actions)],
        #                 "free_action": [i for i in range(self.n_ghost_locs + self.n_wall_locs + self.n_actions,
        #                                                  self.n_ghost_locs + self.n_wall_locs + 2 * self.n_actions)],
        #             }
        #             query_struct = {
        #                 "safe_action": [i for i in range(self.n_actions)]}
        #             self.dpl_layer = DeepProbLogLayer_Approx(
        #                 program=self.program, queries=self.queries,  # evidences=self.evidences,
        #                 input_struct=input_struct, query_struct=query_struct
        #             )
        #         else:
        #             input_struct = {
        #                 "ghost": [i for i in range(self.n_ghost_locs)],
        #                 "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                             self.n_ghost_locs + self.n_wall_locs + self.n_actions)]
        #             }
        #             query_struct = {
        #                 "safe_action": [i for i in range(self.n_actions)]}
        #             self.dpl_layer = DeepProbLogLayer_Approx(
        #                 program=self.program, queries=self.queries, evidences=self.evidences,
        #                 input_struct=input_struct, query_struct=query_struct
        #             )
        #             # input_struct = {
        #             #     "ghost": [i for i in range(self.n_ghost_locs)],
        #             #     "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #             #                                 self.n_ghost_locs + self.n_wall_locs + self.n_actions)],
        #             #     "free_action": [i for i in range(self.n_ghost_locs + self.n_wall_locs + self.n_actions,
        #             #                                      self.n_ghost_locs + self.n_wall_locs + 2 * self.n_actions)],
        #             # }
        #             # query_struct = {
        #             #     "safe_action": [i for i in range(self.n_actions)]}
        #             # self.dpl_layer = DeepProbLogLayer_Approx(
        #             #     program=self.program, queries=self.queries,  # evidences=self.evidences,
        #             #     input_struct=input_struct, query_struct=query_struct
        #             # )

        # # DEBUG
        # debug_queries = ["safe_next"]
        # if self.shield:
        #     if self.detect_ghosts and self.detect_walls:
        #         input_struct = {
        #             "ghost": [i for i in range(self.n_ghost_locs)],
        #             "wall": [i for i in
        #                      range(self.n_ghost_locs, self.n_ghost_locs + self.n_wall_locs)],
        #             "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                         self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
        #     elif self.detect_ghosts:
        #         input_struct = {
        #             "ghost": [i for i in range(self.n_ghost_locs)],
        #             "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                         self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
        #     else:
        #         if hasattr(self, 'free_actions') and self.free_actions:
        #             input_struct = {
        #                 "ghost": [i for i in range(self.n_ghost_locs)],
        #                 "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                             self.n_ghost_locs + self.n_wall_locs + self.n_actions)],
        #                 "free_action": [i for i in range(self.n_ghost_locs + self.n_wall_locs + self.n_actions,
        #                                             self.n_ghost_locs + self.n_wall_locs + 2*self.n_actions)],
        #             }
        #         else:
        #             input_struct = {
        #                 "ghost": [i for i in range(self.n_ghost_locs)],
        #                 "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                             self.n_ghost_locs + self.n_wall_locs + self.n_actions)]
        #             }
        #             # input_struct = {
        #             #     "ghost": [i for i in range(self.n_ghost_locs)],
        #             #     "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #             #                                 self.n_ghost_locs + self.n_wall_locs + self.n_actions)],
        #             #     "free_action": [i for i in range(self.n_ghost_locs + self.n_wall_locs + self.n_actions,
        #             #                                      self.n_ghost_locs + self.n_wall_locs + 2 * self.n_actions)],
        #             # }
        #
        # else:
        #     input_struct = {
        #         "ghost": [i for i in range(self.n_ghost_locs)],
        #         "action": [i for i in range(self.n_ghost_locs + self.n_wall_locs,
        #                                     self.n_ghost_locs + self.n_wall_locs + self.n_actions)]}
        #
        #
        # query_struct = {"safe_next": [i for i in range(1)]}
        # self.query_safety_layer = DeepProbLogLayer_Approx(
        #     program=self.program, queries=debug_queries,
        #     input_struct=input_struct, query_struct=query_struct
        # )


        ##### SOFT SHILDENG WITH GROUND TRUTH ####
        self.queries = [
                   "safe_action(stay)",
                   "safe_action(up)",
                   "safe_action(down)",
                   "safe_action(left)",
                   "safe_action(right)",
               ][: self.n_actions]

        if self.alpha == 0:
            # NO shielding
            pass
        else:
            # HARD shielding and SOFT shielding
            input_struct = {
                "ghost": [i for i in range(self.n_ghost_locs)],
                "action": [i for i in range(self.n_ghost_locs,
                                            self.n_ghost_locs + self.n_actions)]
            }
            query_struct = {"safe_action": [i for i in range(self.n_actions)]}
            cache_path = path.join(self.folder, "../../../.cache", "dpl_layer.p")
            self.dpl_layer = self.get_layer(
                cache_path,
                program=self.debug_program, queries=self.queries, evidences=["safe_next"],
                input_struct=input_struct, query_struct=query_struct
            )

        if self.alpha == "learned":
            self.alpha_net = nn.Sequential(
                    nn.Linear(self.input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid(),
                )

        debug_queries = ["safe_next"]
        query_struct = {"safe_next": [i for i in range(1)]}
        debug_input_struct = {
            "ghost": [i for i in range(self.n_ghost_locs)],
            "action": [i for i in range(self.n_ghost_locs,
                                        self.n_ghost_locs + self.n_actions)]}

        cache_path = path.join(self.folder, "../../../.cache", "query_safety_layer.p")
        self.query_safety_layer = self.get_layer(
            cache_path,
            program=self.debug_program, queries=debug_queries, evidences=[],
            input_struct=debug_input_struct, query_struct=query_struct
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

    def logging_per_step(self, mass, object_detect_probs, base_policy, action_lookup, logger):
        for act in range(self.action_space.n):
            logger.record(
                f"policy/shielded {action_lookup[act]}",
                float(mass.probs[0][act]),
            )
        if object_detect_probs.get("alpha") is not None:
            logger.record(
                f"safety/alpha",
                float(object_detect_probs.get("alpha")),
            )

    def logging_per_episode(self, mass, object_detect_probs, base_policy, action_lookup):
        abs_safe_next_shielded = self.get_step_safety(
            mass.probs,
            object_detect_probs["ground_truth_ghost"],
        )
        abs_safe_next_base = self.get_step_safety(
            base_policy,
            object_detect_probs["ground_truth_ghost"],
        )
        return abs_safe_next_shielded, abs_safe_next_base

    def get_step_safety(self, policy_distribution, ghost_probs):
        with th.no_grad():
            abs_safe_next = self.query_safety_layer(
                x={
                    "ghost": ghost_probs,
                    "action": policy_distribution
                }
            )
            return abs_safe_next["safe_next"]

    def evaluate_safety_shielded(self, obs: th.Tensor):
        with th.no_grad():
            _, _, _, mass, (object_detect_probs, base_policy) = self.forward(obs)
            if self.shield and not self.detect_walls:
                return self.get_step_safety(
                    mass.probs,
                    object_detect_probs["ground_truth_ghost"],
                    # object_detect_probs["ground_truth_wall"],
                )
            else:
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

        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        base_actions = distribution.distribution.probs

        with th.no_grad():
            ground_truth_ghost = get_ground_wall(x, PACMAN_COLOR, GHOST_COLOR)
            ghosts = ground_truth_ghost + (self.sensor_noise)*th.randn(ground_truth_ghost.shape)
            ghosts = th.clamp(ghosts, min=0, max=1)
            object_detect_probs = {
                "ground_truth_ghost": ground_truth_ghost,
            }
            # if not th.all(ground_truth_ghost == 0):
            #     k=1

        if self.alpha == 0:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            object_detect_probs["alpha"] = 0
            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions])

        if not self.differentiable_shield and self.alpha == 1:
            num_rejected_samples = 0
            while True:
                actions = distribution.get_actions(deterministic=deterministic)
                with th.no_grad():
                    results = self.query_safety_layer(
                        x={
                            "ghost": ghosts,
                            "action": th.eye(self.n_actions)[actions],
                        }
                    )
                safe_next = results["safe_next"]
                if not th.any(safe_next.isclose(th.zeros(actions.shape))) or num_rejected_samples > 100000:
                    break
                else:
                    num_rejected_samples += 1
            log_prob = distribution.log_prob(actions)
            object_detect_probs["num_rejected_samples"] = num_rejected_samples
            object_detect_probs["alpha"] = 1
            return (actions, values, log_prob, distribution.distribution, [object_detect_probs, base_actions])

        if self.differentiable_shield:
            results = self.dpl_layer(
                x={
                    "ghost": ghosts,
                    "action": base_actions,
                }
            )

            if self.alpha == "one_minus_safety":
                safety = self.get_step_safety(base_actions, ghosts)
                alpha = (1 - safety)
            elif self.alpha == "learned":
                alpha = self.alpha_net(obs)
                object_detect_probs["alpha"] = alpha
            else:
                alpha = self.alpha
        else:
            with th.no_grad():
                results = self.dpl_layer(
                    x={
                        "ghost": ghosts,
                        "action": base_actions,
                    }
                )
            # Combine safest policy and base_policy
            if self.alpha == "one_minus_safety":
                safety = self.get_step_safety(base_actions, boxes, corners)
                alpha = (1 - safety)
            elif self.alpha == "learned":
                raise NotImplemented
            else:
                alpha = self.alpha

        object_detect_probs["alpha"] = alpha
        safeast_actions = results["safe_action"]
        actions = alpha * safeast_actions + (1 - alpha) * base_actions


        mass = Categorical(probs=actions)
        if not deterministic:
            actions = mass.sample()
        else:
            actions = th.argmax(mass.probs,dim=1)
        log_prob = mass.log_prob(actions)

        return (actions, values, log_prob, mass, [object_detect_probs, base_actions])


    def no_shielding(self, distribution, values, x, deterministic):
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        with th.no_grad():
            ground_truth_ghost = get_ground_wall(x, PACMAN_COLOR, GHOST_COLOR)
            # ground_truth_wall = get_ground_wall(x, PACMAN_COLOR, WALL_COLOR)

            base_actions = distribution.distribution.probs

            object_detect_probs = {
                "ground_truth_ghost": ground_truth_ghost,
                "ground_truth_wall": None, #ground_truth_wall
            }

        return (
            actions,
            values,
            log_prob,
            distribution.distribution,
            [object_detect_probs, base_actions],
        )

    def soft_shielding(self, distribution, values, obs, x, deterministic):
        with th.no_grad():
            ground_truth_ghost = get_ground_wall(x, PACMAN_COLOR, GHOST_COLOR)

        ghosts = self.ghost_layer(obs) if self.detect_ghosts else ground_truth_ghost

        base_actions = distribution.distribution.probs

        results = self.dpl_layer(
            x={
                "ghost": ghosts,
                "action": base_actions,
                "free_action": base_actions,
            }
        )

        actions = results["safe_action"]

        mass = Categorical(probs=actions)
        if not deterministic:
            actions = mass.sample()
        else:
            actions = th.argmax(mass.probs,dim=1)
        log_prob = mass.log_prob(actions)

        with th.no_grad():
            if self.detect_walls:
                object_detect_probs = {
                    "prob_ghost_prior": ghosts,
                    "prob_wall_prior": None, #walls,
                    "prob_ghost_posterior": ghosts,
                    "prob_wall_posterior": None, #results["wall"],
                    "ground_truth_ghost": ground_truth_ghost,
                    "ground_truth_wall": None, #ground_truth_wall,
                }
            else:
                object_detect_probs = {
                    "prob_ghost_prior": ghosts,
                    "prob_wall_prior": None,
                    "prob_ghost_posterior": ghosts,
                    "prob_wall_posterior": None,
                    "ground_truth_ghost": ground_truth_ghost,
                    "ground_truth_wall": None,
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
        if not self.differentiable_shield and self.alpha == 1:
            latent_pi, latent_vf, latent_sde = self._get_latent(obs)
            distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
            log_prob = distribution.log_prob(actions)
            values = self.value_net(latent_vf)
            return values, log_prob, distribution.entropy()


        _, values, log_prob, mass, _ = self.forward(obs)
        return values, log_prob, mass.entropy()

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        with th.no_grad():
            _actions, values, log_prob, mass, _  = self.forward(observation, deterministic)
            return _actions
