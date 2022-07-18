import torch as th
from typing import Optional
import gym
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback
)
from stable_baselines3 import A2C
import time
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

class GoalFinding_DPLA2C(A2C):
    def __init__(self, *args, **kwargs):
        super(GoalFinding_DPLA2C, self).__init__(*args, **kwargs)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "A2C":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/ep_rew_mean",
                        safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    )
                    self.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    )
                    self.logger.record(
                        "rollout/ep_last_r_mean",
                        safe_mean(
                            [ep_info["last_r"] for ep_info in self.ep_info_buffer]
                        ),
                    )
                    self.logger.record(
                        "safety/ep_abs_safety_shielded",
                        safe_mean(
                            [
                                ep_info["abs_safety_shielded"]
                                for ep_info in self.ep_info_buffer
                            ]
                        ),
                    )
                    self.logger.record(
                        "safety/ep_abs_safety_base",
                        safe_mean(
                            [
                                ep_info["abs_safety_base"]
                                for ep_info in self.ep_info_buffer
                            ]
                        ),
                    )
                    if self.ep_info_buffer[0].get("rel_safety_shielded") is not None:
                        self.logger.record(
                            "safety/ep_rel_safety_shielded",
                            safe_mean(
                                [
                                    ep_info["rel_safety_shielded"]
                                    for ep_info in self.ep_info_buffer
                                ]
                            ),
                        )
                        self.logger.record(
                            "safety/ep_rel_safety_base",
                            safe_mean(
                                [
                                    ep_info["rel_safety_base"]
                                    for ep_info in self.ep_info_buffer
                                ]
                            ),
                        )
                self.logger.record("time/fps", fps)
                self.logger.record(
                    "time/time_elapsed",
                    int(time.time() - self.start_time),
                    exclude="tensorboard",
                )
                self.logger.record(
                    "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                )
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        self.abs_safeties_shielded = []
        self.abs_safeties_base = []
        self.rel_safeties_shielded = []
        self.rel_safeties_base = []

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                (
                    actions,
                    values,
                    log_probs,
                    mass,
                    (object_detect_probs, base_policy),
                ) = self.policy.forward(obs_tensor)
                action_lookup = env.envs[0].get_action_lookup()

                self.policy.logging(
                    mass, object_detect_probs, base_policy, action_lookup, self.logger
                )
                abs_safe_next_shielded = self.policy.get_step_safety(
                    mass.probs,
                    object_detect_probs["ground_truth_ghost"],
                    object_detect_probs["ground_truth_wall"],
                )
                abs_safe_next_base = self.policy.get_step_safety(
                    base_policy,
                    object_detect_probs["ground_truth_ghost"],
                    object_detect_probs["ground_truth_wall"],
                )
                self.abs_safeties_shielded.append(abs_safe_next_shielded)
                self.abs_safeties_base.append(abs_safe_next_base)

                rel_safe_next_shielded = None
                rel_safe_next_base = None
                if object_detect_probs.get("prob_ghost_prior") is not None:
                    rel_safe_next_shielded = self.policy.get_step_safety(
                        mass.probs,
                        object_detect_probs["prob_ghost_prior"],
                        object_detect_probs["prob_wall_prior"],
                    )
                    rel_safe_next_base = self.policy.get_step_safety(
                        base_policy,
                        object_detect_probs["prob_ghost_prior"],
                        object_detect_probs["prob_wall_prior"],
                    )
                    self.rel_safeties_shielded.append(rel_safe_next_shielded)
                    self.rel_safeties_base.append(rel_safe_next_base)

                    error_ghost_posterior = (
                            object_detect_probs["ground_truth_ghost"]
                            - object_detect_probs["prob_ghost_posterior"]
                    ).abs()
                    avg_error_ghost_posterior = float(sum(error_ghost_posterior[0])) / len(error_ghost_posterior[0])
                    self.logger.record(f"error/avg_error_ghost_posterior", avg_error_ghost_posterior)

                    error_wall_posterior = (
                            object_detect_probs["ground_truth_wall"]
                            - object_detect_probs["prob_wall_posterior"]
                    ).abs()
                    avg_error_wall_posterior = float(sum(error_wall_posterior[0])) / len(error_wall_posterior[0])
                    self.logger.record(f"error/avg_error_wall_posterior", avg_error_wall_posterior)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            for e in env.envs:
                if e.env.render_or_not:
                    e.env.render()

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            if dones:
                ep_len = infos[0]["episode"]["l"]
                ep_abs_safety_shielded = float(min(self.abs_safeties_shielded))
                ep_abs_safety_base = float(min(self.abs_safeties_base))
                infos[0]["episode"]["abs_safety_shielded"] = ep_abs_safety_shielded
                infos[0]["episode"]["abs_safety_base"] = ep_abs_safety_base
                if self.rel_safeties_shielded:
                    ep_rel_safety_shielded = float(min(self.rel_safeties_shielded))
                    ep_rel_safety_base = float(min(self.rel_safeties_base))
                    infos[0]["episode"]["rel_safety_shielded"] = ep_rel_safety_shielded
                    infos[0]["episode"]["rel_safety_base"] = ep_rel_safety_base
                self.abs_safeties_shielded = []
                self.abs_safeties_base = []
                self.rel_safeties_shielded = []
                self.rel_safeties_base = []

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _, _, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
