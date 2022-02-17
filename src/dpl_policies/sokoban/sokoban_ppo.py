import torch as th
from typing import Optional
import gym

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

class Sokoban_DPLPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(Sokoban_DPLPPO, self).__init__(*args, **kwargs)

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
    ) -> "PPO":
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

            # Display training infos # TODO: put the following in a callback
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
        self.abs_safeties_shielded = [] # TODO: can be put in call back
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
                for e in env.envs:
                    if e.env.render_or_not:
                        e.env.render()
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
                    object_detect_probs["ground_truth_box"],
                    object_detect_probs["ground_truth_corner"],
                )
                abs_safe_next_base = self.policy.get_step_safety(
                    base_policy,
                    object_detect_probs["ground_truth_box"],
                    object_detect_probs["ground_truth_corner"],
                )
                self.abs_safeties_shielded.append(abs_safe_next_shielded)
                self.abs_safeties_base.append(abs_safe_next_base)

                rel_safe_next_shielded = None
                rel_safe_next_base = None
                if object_detect_probs.get("prob_box_prior") is not None:
                    rel_safe_next_shielded = self.policy.get_step_safety(
                        mass.probs,
                        object_detect_probs["prob_box_prior"],
                        object_detect_probs["prob_corner_prior"],
                    )
                    rel_safe_next_base = self.policy.get_step_safety(
                        base_policy,
                        object_detect_probs["prob_box_prior"],
                        object_detect_probs["prob_corner_prior"],
                    )
                    self.rel_safeties_shielded.append(rel_safe_next_shielded)
                    self.rel_safeties_base.append(rel_safe_next_base)

                    error_box_posterior = (
                        object_detect_probs["ground_truth_box"]
                        - object_detect_probs["prob_box_posterior"]
                    ).abs()
                    avg_error_box_posterior = float(sum(error_box_posterior[0]))/len(error_box_posterior[0])
                    self.logger.record(f"error/avg_error_box_posterior", avg_error_box_posterior)

                    error_corner_posterior = (
                            object_detect_probs["ground_truth_corner"]
                            - object_detect_probs["prob_corner_posterior"]
                    ).abs()
                    avg_error_corner_posterior = float(sum(error_corner_posterior[0]))/len(error_corner_posterior[0])
                    self.logger.record(f"error/avg_error_corner_posterior", avg_error_corner_posterior)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            (new_obs, rewards, dones, infos) = env.step(clipped_actions)

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
                log_probs
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

    # def _update_info_buffer(
    #     self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None
    # ) -> None:
    #     """
    #     Retrieve reward, episode length, episode success and update the buffer
    #     if using Monitor wrapper or a GoalEnv.
    #
    #     :param infos: List of additional information about the transition.
    #     :param dones: Termination signals
    #     """
    #     if dones is None:
    #         dones = np.array([False] * len(infos))
    #     for idx, info in enumerate(infos):
    #         maybe_ep_info = info.get("episode")
    #         maybe_is_success = info.get("is_success")
    #         if maybe_ep_info is not None:
    #             self.ep_info_buffer.extend([maybe_ep_info])
    #         if maybe_is_success is not None and dones[idx]:
    #             self.ep_success_buffer.append(maybe_is_success)
    #
    # def train(self) -> None:
    #     """
    #     Update policy using the currently gathered rollout buffer.
    #     """
    #     # Update optimizer learning rate
    #     self._update_learning_rate(self.policy.optimizer)
    #     # Compute current clip range
    #     clip_range = self.clip_range(self._current_progress_remaining)
    #     # Optional: clip range for the value function
    #     if self.clip_range_vf is not None:
    #         clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
    #
    #     entropy_losses = []
    #     pg_losses, value_losses = [], []
    #     clip_fractions = []
    #
    #     continue_training = True
    #
    #     # train for n_epochs epochs
    #     for epoch in range(self.n_epochs):
    #         approx_kl_divs = []
    #         # Do a complete pass on the rollout buffer
    #         for rollout_data in self.rollout_buffer.get(self.batch_size):
    #             actions = rollout_data.actions
    #             if isinstance(self.action_space, spaces.Discrete):
    #                 # Convert discrete action from float to long
    #                 actions = rollout_data.actions.long().flatten()
    #
    #             # Re-sample the noise matrix because the log_std has changed
    #             # TODO: investigate why there is no issue with the gradient
    #             # if that line is commented (as in SAC)
    #             if self.use_sde:
    #                 self.policy.reset_noise(self.batch_size)
    #
    #             values, log_prob, entropy, _ = self.policy.evaluate_actions(
    #                 rollout_data.observations, actions
    #             )
    #
    #             values = values.flatten()
    #             # Normalize advantage
    #             advantages = rollout_data.advantages
    #             advantages = (advantages - advantages.mean()) / (
    #                 advantages.std() + 1e-8
    #             )
    #
    #             # ratio between old and new policy, should be one at the first iteration
    #             ratio = th.exp(log_prob - rollout_data.old_log_prob)
    #
    #             # clipped surrogate loss
    #             policy_loss_1 = advantages * ratio
    #             policy_loss_2 = advantages * th.clamp(
    #                 ratio, 1 - clip_range, 1 + clip_range
    #             )
    #             policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
    #
    #             # Logging
    #             pg_losses.append(policy_loss.item())
    #             clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
    #             clip_fractions.append(clip_fraction)
    #
    #             if self.clip_range_vf is None:
    #                 # No clipping
    #                 values_pred = values
    #             else:
    #                 # Clip the different between old and new value
    #                 # NOTE: this depends on the reward scaling
    #                 values_pred = rollout_data.old_values + th.clamp(
    #                     values - rollout_data.old_values, -clip_range_vf, clip_range_vf
    #                 )
    #             # Value loss using the TD(gae_lambda) target
    #             value_loss = F.mse_loss(rollout_data.returns, values_pred)
    #             value_losses.append(value_loss.item())
    #
    #             # Entropy loss favor exploration
    #             if entropy is None:
    #                 # Approximate entropy when no analytical form
    #                 entropy_loss = -th.mean(-log_prob)
    #             else:
    #                 entropy_loss = -th.mean(entropy)
    #
    #             entropy_losses.append(entropy_loss.item())
    #
    #             loss = (
    #                 policy_loss
    #                 + self.ent_coef * entropy_loss
    #                 + self.vf_coef * value_loss
    #             )
    #
    #             # Calculate approximate form of reverse KL Divergence for early stopping
    #             # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
    #             # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
    #             # and Schulman blog: http://joschu.net/blog/kl-approx.html
    #             with th.no_grad():
    #                 log_ratio = log_prob - rollout_data.old_log_prob
    #                 approx_kl_div = (
    #                     th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
    #                 )
    #                 approx_kl_divs.append(approx_kl_div)
    #
    #             if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
    #                 continue_training = False
    #                 if self.verbose >= 1:
    #                     print(
    #                         f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
    #                     )
    #                 break
    #
    #             # Optimization step
    #             self.policy.optimizer.zero_grad()
    #             loss.backward()
    #             # Clip grad norm
    #             th.nn.utils.clip_grad_norm_(
    #                 self.policy.parameters(), self.max_grad_norm
    #             )
    #             self.policy.optimizer.step()
    #
    #         if not continue_training:
    #             break
    #
    #     self._n_updates += self.n_epochs
    #     explained_var = explained_variance(
    #         self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
    #     )
    #
    #     # Logs
    #     self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    #     self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    #     self.logger.record("train/value_loss", np.mean(value_losses))
    #     self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    #     self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    #     self.logger.record("train/loss", loss.item())
    #     self.logger.record("train/explained_variance", explained_var)
    #     if hasattr(self.policy, "log_std"):
    #         self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
    #
    #     self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    #     self.logger.record("train/clip_range", clip_range)
    #     if self.clip_range_vf is not None:
    #         self.logger.record("train/clip_range_vf", clip_range_vf)
