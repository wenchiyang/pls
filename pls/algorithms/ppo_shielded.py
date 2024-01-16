from torch.nn import functional as F
import gym
from typing import Tuple
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.utils import explained_variance

from pls.shields.shields import Shield
from gym import spaces
import torch as th
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy


class ActorCriticPolicy_shielded(ActorCriticPolicy):
    """
    Policy class for PPO.

    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        shield_params=None,
        config_folder=None,
        get_sensor_value_ground_truth=None,
        **kwargs,
    ):
        """
        :param observation_space: Metadata about the observation space to operate over
        :param action_space: Metadata about the action space
        :param lr_schedule:
        :param shield_params: dict containing parameters of the
        :param config_folder: location of the config file
        :param get_sensor_value_ground_truth: function used to compute ground truth observations from image input
        :param kwargs:
        """

        super(ActorCriticPolicy_shielded, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )
        self.get_sensor_value_ground_truth = get_sensor_value_ground_truth

        if shield_params is None:
            # no shielding: does not add any shield to the base policy
            self.shield = None
        else:
            self.shield = Shield(
                config_folder=config_folder,
                get_sensor_value_ground_truth=get_sensor_value_ground_truth,
                **shield_params,
            )

        # debug_info is written by self.forward() and used by the callback for logging
        self.debug_info = {}
        # info is written by self.evaluate_actions() and used by PPO to compute the loss
        self.info = {}

        self._build(lr_schedule)

    def forward(self, x, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param x: tensor representing the states to produce policy distribution
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        # Preprocess the image input if needed
        obs = x
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Evaluate the values for the given image input
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        base_actions = distribution.distribution.probs

        # get the ground truth observation
        if self.shield is None:
            sensor_values = self.get_sensor_value_ground_truth(input=x)
        else:
            sensor_values = self.shield.get_sensor_values(x)

        self.debug_info = {"sensor_value": sensor_values, "base_policy": base_actions}

        if self.shield is None:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            self.debug_info["shielded_policy"] = base_actions

            return (actions, values, log_prob)

        elif self.shield.differentiable:  # PLPG
            # compute the shielded policy
            actions = self.shield.get_shielded_policy(base_actions, sensor_values)
            shielded_policy = Categorical(probs=actions)

            # get the most probable action of the shielded policy if we want to use a deterministic policy,
            # otherwuse, sample an action
            if deterministic:
                actions = th.argmax(shielded_policy.probs, dim=1)
            else:
                actions = shielded_policy.sample()

            log_prob = shielded_policy.log_prob(actions)
            self.debug_info["shielded_policy"] = shielded_policy.probs

            return (actions, values, log_prob)

        else:  # VSRL
            with th.no_grad():
                actions = self.shield.get_shielded_policy_vsrl(
                    base_actions, sensor_values
                )
                shielded_policy = Categorical(probs=actions)

                # get the most probable action of the shielded policy if we want to use a deterministic policy,
                # otherwuse, sample an action
                if deterministic:
                    actions = th.argmax(shielded_policy.probs, dim=1)
                else:
                    actions = shielded_policy.sample()

                log_prob = distribution.log_prob(actions)
                self.debug_info["shielded_policy"] = shielded_policy.probs

                return (actions, values, log_prob)

    def evaluate_actions(
        self, x: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param x: visual input
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        # Preprocess the image input if needed
        obs = x
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Evaluate the values for the given image input
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)


        base_actions = distribution.distribution.probs

        if self.shield is None or not self.shield.differentiable:
            sensor_values = self.get_sensor_value_ground_truth(input=x)
            log_prob = distribution.log_prob(actions)

            self.info = {"sensor_value": sensor_values, "base_policy": base_actions}

            return (values, log_prob, distribution.entropy())

        elif self.shield.differentiable:  # PLPG
            sensor_values = self.shield.get_sensor_values(x)
            # compute the shielded policy
            shielded_actions = self.shield.get_shielded_policy(base_actions, sensor_values)
            shielded_policy = Categorical(probs=shielded_actions)
            log_prob = shielded_policy.log_prob(actions)

            self.info = {"sensor_value": sensor_values, "base_policy": base_actions}

            return (values, log_prob, shielded_policy.entropy())
        else:  # VSRL
            sensor_values = self.shield.get_sensor_values(x)
            log_prob = distribution.log_prob(actions)

            self.info = {"sensor_value": sensor_values, "base_policy": base_actions}

            return (values, log_prob, distribution.entropy())


class PPO_shielded(PPO):
    """
    PPO class with a shield. It behaves as standard PPO if alpha=0 and if shield_params={}

    """

    def __init__(
        self,
        *args,
        policy=ActorCriticPolicy_shielded,
        alpha=0,
        policy_safety_params={},
        **kwargs,
    ):
        """
        :param policy: the policy class
        :param alpha: the coefficient for the policy safety in the loss
        :param policy_safety_params: parameters used to compute policy safety loss
        """
        super(PPO_shielded, self).__init__(*args, policy, **kwargs)
        self.alpha = alpha
        self.policy_safety_calculater = Shield(**policy_safety_params)

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
        safety_losses = []  # safety loss

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

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
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

                ####### Safety loss ###########################################
                policy_safeties = self.policy_safety_calculater.get_policy_safety(
                    self.policy.info["sensor_value"], self.policy.info["base_policy"]
                )
                policy_safeties = policy_safeties.flatten()
                safety_loss = -th.log(policy_safeties)
                safety_loss = th.mean(safety_loss)
                safety_losses.append(safety_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.alpha * safety_loss
                )
                ###############################################################

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

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
