from torch import nn
from deepproblog.light import DeepProbLogLayer
import torch as th
from util import get_ground_wall
from typing import Any, Dict, Optional, Type, Union, List, Tuple
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
import gym
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)
from stable_baselines3.common.distributions import CategoricalDistribution
from torch.distributions import Categorical
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import time
import numpy as np
from stable_baselines3.common.type_aliases import GymObs, MaybeCallback, GymStepReturn
from stable_baselines3.common.utils import obs_as_tensor, safe_mean


WALL_COLOR = 0.25
GHOST_COLOR = 0.5
PACMAN_COLOR = 0.75
FOOD_COLOR = 1


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        n_actions,
        shield,
        detect_ghosts,
        detect_walls,
        program_path,
        # logger,
    ):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.shield = shield
        self.detect_ghosts = detect_ghosts
        self.detect_walls = detect_walls
        self.n_actions = n_actions
        self.program_path = program_path
        # self.logger = logger

    def forward(self, x):
        xx = th.flatten(x, 1)
        return xx

class DPLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        image_encoder: Encoder = None,
        **kwargs
    ):
        super(DPLActorCriticPolicy, self).__init__(
            *args,
            **kwargs
        )

        self.image_encoder = image_encoder
        # self.logger = self.image_encoder.logger
        self.input_size = self.image_encoder.input_size
        self.shield = self.image_encoder.shield
        self.detect_ghosts = self.image_encoder.detect_ghosts
        self.detect_walls = self.image_encoder.detect_walls
        self.n_actions = self.image_encoder.n_actions
        self.program_path = self.image_encoder.program_path

        # self.base_policy_layer = self.policy

        if self.detect_ghosts:
            self.ghost_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                # nn.Softmax()
                nn.Sigmoid(),  # TODO : add a flag
            )
        if self.detect_walls:
            self.wall_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                # nn.Softmax()
                nn.Sigmoid(),
            )

        if self.shield:
            with open(self.program_path) as f:
                self.program = f.read()

            self.queries = [
                "safe_action(stay)",
                "safe_action(up)",
                "safe_action(down)",
                "safe_action(left)",
                "safe_action(right)",
                "safe_next",
            ]
            self.dpl_layer = DeepProbLogLayer(
                program=self.program, queries=self.queries
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

        if not self.shield:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            return actions, values, log_prob

        ghosts_ground_relative = get_ground_wall(x[0], PACMAN_COLOR, GHOST_COLOR)
        wall_ground_relative = get_ground_wall(x[0], PACMAN_COLOR, WALL_COLOR)
        ghosts = self.ghost_layer(obs) if self.detect_ghosts else ghosts_ground_relative
        walls = self.wall_layer(obs) if self.detect_walls else wall_ground_relative

        base_actions = distribution.distribution.probs
        results = self.dpl_layer(
            x={"ghost": ghosts, "wall": walls, "action": base_actions}
        )

        actions = results["safe_action"]
        safe_next = results["safe_next"]

        actions = actions / safe_next

        mass = Categorical(probs=actions)
        actions = mass.sample()
        log_prob = mass.log_prob(actions)
        return actions, values, log_prob

class DPLPPO(PPO):
    def __init__(self, *args, **kwargs):
        super(DPLPPO, self).__init__(*args, **kwargs)

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
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_last_r_mean",
                                       safe_mean([ep_info["last_r"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self



class DPLPolicyGradientPolicy(OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        image_encoder: Encoder = None
    ):
        super(DPLPolicyGradientPolicy, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            policy_base=policy_base,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=supported_action_spaces,
        )

        self.image_encoder = image_encoder
        # self.logger = self.image_encoder.logger
        self.input_size = self.image_encoder.input_size
        self.shield = self.image_encoder.shield
        self.detect_ghosts = self.image_encoder.detect_ghosts
        self.detect_walls = self.image_encoder.detect_walls
        self.n_actions = self.image_encoder.n_actions
        self.program_path = self.image_encoder.program_path

        # self.base_policy_layer = self.policy

        if self.detect_ghosts:
            self.ghost_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Sigmoid(),  # TODO : add a flag
            )
        if self.detect_walls:
            self.wall_layer = nn.Sequential(
                nn.Linear(self.input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 4),
                nn.Sigmoid(),
            )

        if self.shield:
            with open(self.program_path) as f:
                self.program = f.read()

            self.queries = [
                "safe_action(stay)",
                "safe_action(up)",
                "safe_action(down)",
                "safe_action(left)",
                "safe_action(right)",
                "safe_next",
            ]
            self.dpl_layer = DeepProbLogLayer(
                program=self.program, queries=self.queries
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

        if not self.shield:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            return actions, values, log_prob

        ghosts_ground_relative = get_ground_wall(x[0], PACMAN_COLOR, GHOST_COLOR)
        wall_ground_relative = get_ground_wall(x[0], PACMAN_COLOR, WALL_COLOR)
        ghosts = self.ghost_layer(obs) if self.detect_ghosts else ghosts_ground_relative
        walls = self.wall_layer(obs) if self.detect_walls else wall_ground_relative

        base_actions = distribution.distribution.probs
        results = self.dpl_layer(
            x={"ghost": ghosts, "wall": walls, "action": base_actions}
        )

        actions = results["safe_action"]
        safe_next = results["safe_next"]

        actions = actions / safe_next

        mass = Categorical(probs=actions)
        actions = mass.sample()
        log_prob = mass.log_prob(actions)
        return actions, values, log_prob