import time
from typing import Union

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.callbacks import ConvertCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import (
    GymObs,
    GymStepReturn,
)
from stable_baselines3.common.utils import safe_mean
from torch import nn

from env_specific_classes.carracing.util import get_ground_truth_of_grass
from pls.shields.shields import Shield


class Carracing_FeaturesExtractor(BaseFeaturesExtractor):
    """
    A CNN based architecture for PPO features extractor.
    The architecture is a standard multi layer CNN with ReLU activations.

    """

    def __init__(self, observation_space, features_dim=256):
        """
        :param observation_space: Metadata about the observation space to operate over.
        :param features_dim: The number of features to extract from the observations.
        """

        super(Carracing_FeaturesExtractor, self).__init__(
            observation_space, features_dim
        )

        self._features_dim = features_dim  # This is the size of the output
        n_stacked_images, _, _ = observation_space.shape
        assert n_stacked_images == 4
        self.conv1 = nn.Conv2d(
            in_channels=n_stacked_images,
            out_channels=8,
            kernel_size=5,
            stride=2,
            padding=1,
        )  # (8, 23, 23)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=1
        )  # (16, 11, 11)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1
        )  # (32, 5, 5)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1
        )  # (64, 2, 2)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass through the model.

        :param x: tensor representing the states to extract features from.
        :return: Tensor representing a compressed view of the input image. Intended to be used for the policy
        """

        # convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x.view(-1, self._features_dim)


class Carracing_Monitor(Monitor):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    """

    def __init__(self, *args, render_or_not=False, **kwargs):
        self.vio_len = 200
        self.render_or_not = render_or_not
        super(Carracing_Monitor, self).__init__(*args, **kwargs)

    def reset(self, **kwargs) -> GymObs:
        # DEBUG
        self.cont_in_grass_len = 0
        self.max_cont_in_grass_len = 0

        self.total_violate_len = 0
        self.cont_violate_len = 0
        self.max_cont_violate_len = 0

        #
        output = super(Carracing_Monitor, self).reset(**kwargs)
        return output

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """

        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

        grass_ground_truth = get_ground_truth_of_grass(
            th.from_numpy(observation.copy()).unsqueeze(0)
        )

        is_in_grass = th.all(grass_ground_truth)
        if is_in_grass == True:
            self.cont_in_grass_len += 1
        else:
            self.cont_in_grass_len = 0
        self.max_cont_in_grass_len = max(
            self.cont_in_grass_len, self.max_cont_in_grass_len
        )

        left_grass = grass_ground_truth[0, 1].bool()
        right_grass = grass_ground_truth[0, 2].bool()
        left_grass_only = th.logical_and(left_grass, ~right_grass)
        right_grass_only = th.logical_and(~left_grass, right_grass)

        violate_cont = th.logical_or(left_grass_only, right_grass_only)

        if violate_cont == True:
            self.cont_violate_len += 1
            self.total_violate_len += 1
        else:
            self.cont_violate_len = 0
        self.max_cont_violate_len = max(
            self.cont_violate_len, self.max_cont_violate_len
        )

        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)

            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
                "violate_constraint": self.max_cont_in_grass_len > self.vio_len
                or info["out_of_field"],
                "is_success": info["is_success"],
                "out_of_field": info["out_of_field"],
                # debug
                "max_cont_in_grass_len": self.max_cont_in_grass_len,
                "max_cont_violate_len": self.max_cont_violate_len,
                "total_violate_len": self.total_violate_len,
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

        if self.render_or_not:
            self.env.render()

        return observation, reward, done, info


class Carracing_Callback(ConvertCallback):
    """
    Callback class for logging to files and TensorBoard for PPO.

    """

    def __init__(self, callback=None, policy_safety_params=None):
        super(Carracing_Callback, self).__init__(callback)
        # initialize a shield
        self.shield = Shield(**policy_safety_params)

    def _on_training_start(self):
        self.n_violations = 0
        self.action_lookup = self.locals["self"].env.envs[0].get_action_lookup()

    def on_rollout_start(self):
        self.shielded_policy_safeties = []
        self.base_policy_safeties = []

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        :return: If the callback returns False, training is aborted early.
        """

        debug_info = self.locals["self"].policy.debug_info
        dones = self.locals["dones"]
        info = self.locals["infos"][0]

        for act in range(self.locals["self"].action_space.n):
            self.logger.record(
                f"policy/shielded {self.action_lookup[act]}",
                float(debug_info["shielded_policy"][0][act]),
            )
        with th.no_grad():
            safe_next_shielded = self.shield.get_policy_safety(
                debug_info["sensor_value"],
                debug_info["shielded_policy"],
            )
            safe_next_base = self.shield.get_policy_safety(
                debug_info["sensor_value"],
                debug_info["base_policy"],
            )

        self.shielded_policy_safeties.append(safe_next_shielded)
        self.base_policy_safeties.append(safe_next_base)

        if dones:
            # on_episide_end
            info["episode"]["shielded_policy_safeties"] = float(
                min(self.shielded_policy_safeties)
            )
            info["episode"]["base_policy_safeties"] = float(
                min(self.base_policy_safeties)
            )
            if info["episode"]["violate_constraint"]:
                self.n_violations += 1

            # on_episide_start
            self.shielded_policy_safeties = []
            self.base_policy_safeties = []

    def on_rollout_end(self):
        ep_info_buffer = self.locals["self"].ep_info_buffer

        if len(ep_info_buffer) > 0 and len(ep_info_buffer[0]) > 0:
            # log rollout
            self.logger.record(
                "rollout/success_rate",
                safe_mean([ep_info["is_success"] for ep_info in ep_info_buffer]),
            )

            self.logger.record("rollout/n_violations", self.n_violations)
            self.logger.record(
                "rollout/violation_rate",
                safe_mean(
                    [ep_info["violate_constraint"] for ep_info in ep_info_buffer]
                ),
            )

            # log safety
            self.logger.record(
                "safety/shielded_policy_safety",
                safe_mean(
                    [ep_info["shielded_policy_safeties"] for ep_info in ep_info_buffer]
                ),
            )
            self.logger.record(
                "safety/base_policy_safety",
                safe_mean(
                    [ep_info["base_policy_safeties"] for ep_info in ep_info_buffer]
                ),
            )
            self.logger.record(
                "safety/policy_safety_impr",
                safe_mean(
                    [
                        ep_info["shielded_policy_safeties"]
                        - ep_info["base_policy_safeties"]
                        for ep_info in ep_info_buffer
                    ]
                ),
            )
            # log debug
            self.logger.record(
                "debug/max_cont_in_grass_len",
                safe_mean(
                    [ep_info["max_cont_in_grass_len"] for ep_info in ep_info_buffer]
                ),
            )
            self.logger.record(
                "debug/max_cont_violate_len",
                safe_mean(
                    [ep_info["max_cont_violate_len"] for ep_info in ep_info_buffer]
                ),
            )
            self.logger.record(
                "debug/total_violate_len",
                safe_mean([ep_info["total_violate_len"] for ep_info in ep_info_buffer]),
            )
            self.logger.record(
                "rollout/out_of_field",
                safe_mean([ep_info["out_of_field"] for ep_info in ep_info_buffer]),
            )


class Carracing_Observation_Net(nn.Module):
    """
    A CNN based architecture for getting observations from environment states.
    The architecture is a standard multi layer CNN with ReLU activations, followed by a linear layer.

    """

    def __init__(self, input_size, output_size):
        super(Carracing_Observation_Net, self).__init__()
        # input (4, 48, 48)
        # convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=1
        )  # (8, 23, 23)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=1
        )  # (16, 11, 11)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1
        )  # (32, 5, 5)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1
        )  # (64, 2, 2)
        # linear layers
        self.fc1 = nn.Linear(256, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: th.Tensor):
        """
        Forward pass through the model.

        :param x: tensor representing the states to extract sensor values from
        :return: tensor of sensor values and the xy coordinate of the agent
        """

        # convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # flattening the image
        x = x.view(-1, 256)
        # linear layers
        x = self.fc1(x)
        return x

    def get_sensor_values(self, x):
        """
        Forward pass through the model. Intend to be called by Shield.

        :param x: tensor representing the states to extract sensor values from.
        :return: tensor of sensor values in [0, 1]
        """

        # x's dimensions: (input, number of stacked images in the input, width and height)
        x = x[:, 0:1, :, :]
        x = self.forward(x)
        x = self.sigmoid(x)
        return x
