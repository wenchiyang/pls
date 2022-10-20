import torch as th
from torch.nn import functional as F
from typing import Optional, NamedTuple, Generator
import gym
from gym.spaces import Box
from gym.vector.utils import spaces

from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback
)
from stable_baselines3 import PPO
import time
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from pls.dpl_policies.sokoban.util import safe_max, safe_min
from stable_baselines3.common.preprocessing import get_obs_shape
from pls.dpl_policies.sokoban.dpl_policy import box_stuck_in_corner


class RolloutBufferSamples_TinyGrid(NamedTuple):
    observations: th.Tensor
    tinygrids: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    # safeties: th.Tensor

class RolloutBuffer_TinyGrid(RolloutBuffer):
    def __init__(self, *args, tinygrid_space, **kwargs):
        self.tinygrid_shape = get_obs_shape(tinygrid_space)
        # self.safeties = None
        super(RolloutBuffer_TinyGrid, self).__init__(*args, **kwargs)

    def reset(self) -> None:
        super(RolloutBuffer_TinyGrid, self).reset()
        self.tinygrids = np.zeros((self.buffer_size, self.n_envs) + self.tinygrid_shape, dtype=np.float32)
        # self.safeties = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, *args, tinygrid, **kwargs) -> None:
        self.tinygrids[self.pos] = np.array(tinygrid).copy()
        # self.safeties[self.pos] = np.array(safety).copy()
        super(RolloutBuffer_TinyGrid, self).add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples_TinyGrid, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "tinygrids",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                # "safeties"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples_TinyGrid:
        data = (
            self.observations[batch_inds],
            self.tinygrids[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            # self.safeties[batch_inds].flatten(),
        )
        return RolloutBufferSamples_TinyGrid(*tuple(map(self.to_torch, data)))


class Sokoban_DPLPPO(PPO):
    def __init__(self, *args, safety_coef, **kwargs):
        self.safety_coef = safety_coef
        super(Sokoban_DPLPPO, self).__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super(Sokoban_DPLPPO, self)._setup_model()
        self.tinygrid_space = Box(
            low=0,
            high=1,
            shape=(
                self.policy.tinygrid_dim, self.policy.tinygrid_dim
            )
        )
        self.rollout_buffer = RolloutBuffer_TinyGrid(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            tinygrid_space=self.tinygrid_space,)

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
                    self.logger.record(
                        "safety/ep_rel_safety_impr",
                        safe_mean(
                            [
                                ep_info["rel_safety_shielded"] - ep_info["rel_safety_base"]
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
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        render_mode = env.envs[0].render_mode

        callback.on_rollout_start()
        ####### on_episode_start #######
        alphas = []
        nums_rejected_samples = []
        abs_safeties_shielded = []
        abs_safeties_base = []
        rel_safeties_shielded = []
        rel_safeties_base = []
        n_risky_states = []
        ##############################
        action_lookup = env.envs[0].get_action_lookup()

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
                tinygrid = self.env.render("tiny_rgb_array")
                tinygrid = obs_as_tensor(tinygrid, self.device).unsqueeze(0)

                (
                    actions,
                    values,
                    log_probs,
                    mass,
                    (object_detect_probs, base_policy),
                ) = self.policy.forward(obs_tensor, tinygrid)

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

            # safety = object_detect_probs["policy_safety"]

            if dones:
                ep_len = infos[0]["episode"]["l"]
                ##### on_episide_end ##########
                infos[0]["episode"]["n_risky_states"] = float(sum(n_risky_states))
                infos[0]["episode"]["abs_safety_shielded"] = float(min(abs_safeties_shielded))
                infos[0]["episode"]["abs_safety_base"] = float(min(abs_safeties_base))
                infos[0]["episode"]["rel_safety_shielded"] = float(min(rel_safeties_shielded))
                infos[0]["episode"]["rel_safety_base"] = float(min(rel_safeties_base))
                if infos[0]["episode"]["violate_constraint"]:
                    self.n_deaths += 1
                if object_detect_probs.get("alpha") is not None:
                    infos[0]["episode"]["alpha_min"] = float(min(alphas))
                    infos[0]["episode"]["alpha_max"] = float(max(alphas))
                if object_detect_probs.get("num_rejected_samples") is not None:
                    num_rejected_samples_max = float(max(nums_rejected_samples))
                    infos[0]["episode"]["num_rejected_samples_max"] = num_rejected_samples_max
                ##############################
                ##### on_episide_start ##########
                alphas = []
                nums_rejected_samples = []
                abs_safeties_shielded = []
                abs_safeties_base = []
                rel_safeties_shielded = []
                rel_safeties_base = []
                n_risky_states = []
                ##############################
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
                log_probs,
                tinygrid=tinygrid,
                # safety=safety
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with th.no_grad():
            # Compute value for the last timestep
            new_obs_tensor = obs_as_tensor(new_obs, self.device)
            new_obs_ = self.policy.image_encoder(new_obs_tensor).numpy()
            values = self.policy.predict_values(obs_as_tensor(new_obs_, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        safety_losses = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, policy_safeties = self.policy.evaluate_actions(rollout_data.observations, rollout_data.tinygrids, actions)
                values = values.flatten()
                policy_safeties = policy_safeties.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # safety loss
                safety_loss = -th.log(policy_safeties)
                safety_loss = th.mean(safety_loss)
                safety_losses.append(safety_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss  + self.safety_coef * safety_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/safety_loss", np.mean(safety_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/used_epochs", epoch + 1)
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)