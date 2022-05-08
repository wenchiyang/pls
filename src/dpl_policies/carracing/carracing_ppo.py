import torch as th
from typing import Optional
import gym

# from tqdm import tqdm

from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback
)
from stable_baselines3 import PPO
import time
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from .util import safe_max, safe_min


class DPL_RolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        self.distribution = None
        super(DPL_RolloutBuffer, self).__init__(*args, **kwargs)

    def reset(self):
        self.distribution = np.zeros((self.buffer_size, self.n_envs, self.action_space.n), dtype=np.float32)
        super(DPL_RolloutBuffer, self).reset()


class Carracing_DPLPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(Carracing_DPLPPO, self).__init__(*args, **kwargs)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "PPO":
        iteration = 0
        self.n_deaths = 0
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
        # step_progress_bar = tqdm(total=total_timesteps // self.n_steps, desc="learn")

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            # Display training infos # TODO: put the following in a callback
            if log_interval is not None and iteration % log_interval == 0:
                self.logger.record("safety/n_deaths", self.n_deaths)
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
                        "rollout/success_rate",
                        safe_mean(
                            [ep_info["is_success"] for ep_info in self.ep_info_buffer]
                        ),
                    )
                    self.logger.record(
                        "rollout/#violations",
                        safe_mean(
                            [ep_info["violate_constraint"] for ep_info in self.ep_info_buffer]
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
                    self.logger.record(
                        "safety/ep_abs_safety_impr",
                        safe_mean(
                            [
                                ep_info["abs_safety_shielded"] - ep_info["abs_safety_base"]
                                for ep_info in self.ep_info_buffer
                            ]
                        ),
                    )
                    self.logger.record(
                        "safety/n_risky_states",
                        safe_mean(
                            [
                                ep_info["n_risky_states"]
                                for ep_info in self.ep_info_buffer
                            ]
                        ),
                    )
                    if self.ep_info_buffer[0].get("alpha_min") is not None:
                        self.logger.record(
                            "safety/alpha_min",
                            safe_min(
                                [
                                    ep_info["alpha_min"]
                                    for ep_info in self.ep_info_buffer
                                ]
                            ),
                        )
                        self.logger.record(
                            "safety/alpha_max",
                            safe_max(
                                [
                                    ep_info["alpha_max"]
                                    for ep_info in self.ep_info_buffer
                                ]
                            ),
                        )
                    if self.ep_info_buffer[0].get("num_rejected_samples_max") is not None:
                        self.logger.record(
                            "safety/num_rejected_samples_max",
                            safe_max(
                                [
                                    ep_info["num_rejected_samples_max"]
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
            # step_progress_bar.update(1)
        callback.on_training_end()
        # step_progress_bar.close()
        return self

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
            render_interval: int = 10
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
        :param render_interval: The number of elapsed steps between rendered frames.
        Higher for faster processing.
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        alphas = []
        nums_rejected_samples = []
        abs_safeties_shielded = []  # TODO: can be put in call back
        abs_safeties_base = []
        n_risky_states = 0
        # progress_bar = tqdm(total=n_rollout_steps)
        # progress_bar.set_description("collect_rollouts")
        while n_steps < n_rollout_steps:
            # progress_bar.update(1)
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

                ##### LOG #####
                action_lookup = env.envs[0].get_action_lookup()

                self.policy.logging_per_step(
                    mass, object_detect_probs, base_policy, action_lookup, self.logger
                )
                abs_safe_next_shielded, abs_safe_next_base = self.policy.logging_per_episode(
                    mass, object_detect_probs, base_policy, action_lookup
                )
                if object_detect_probs.get("alpha") is not None:
                    alphas.append(object_detect_probs["alpha"])
                if object_detect_probs.get("num_rejected_samples") is not None:
                    nums_rejected_samples.append(object_detect_probs["num_rejected_samples"])

                # if in a risky situation
                if th.any(object_detect_probs["ground_truth_grass"]):
                    n_risky_states += 1
                    # risky_action = 1 + th.logical_and(object_detect_probs["ground_truth_box"], object_detect_probs["ground_truth_corner"]).squeeze().nonzero()
                    # if th.any(actions == risky_action):
                    #     self.n_deaths += 1

                abs_safeties_shielded.append(abs_safe_next_shielded)
                abs_safeties_base.append(abs_safe_next_base)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            if n_steps % render_interval == 0:
                # if rewards: progress_bar.set_postfix({"reward": str(rewards)})
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
                ep_abs_safety_shielded = float(min(abs_safeties_shielded))
                ep_abs_safety_base = float(min(abs_safeties_base))
                infos[0]["episode"]["abs_safety_shielded"] = ep_abs_safety_shielded
                infos[0]["episode"]["abs_safety_base"] = ep_abs_safety_base
                infos[0]["episode"]["n_risky_states"] = n_risky_states

                if object_detect_probs.get("alpha") is not None:
                    alpha_min = float(min(alphas))
                    alpha_max = float(max(alphas))
                    infos[0]["episode"]["alpha_min"] = alpha_min
                    infos[0]["episode"]["alpha_max"] = alpha_max
                    alphas = []
                if object_detect_probs.get("num_rejected_samples") is not None:
                    num_rejected_samples_max = float(max(nums_rejected_samples))
                    infos[0]["episode"]["num_rejected_samples_max"] = num_rejected_samples_max
                    nums_rejected_samples = []

                abs_safeties_shielded = []
                abs_safeties_base = []
                n_risky_states = 0

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
        # progress_bar.close()
        return True

