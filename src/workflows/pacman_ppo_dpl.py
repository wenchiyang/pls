import random
import gym
import numpy as np

import torch as th
from os.path import join, abspath

from dpl_policy_stable_baselines import DPLActorCriticPolicy, Encoder, DPLPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from torch import nn
import pacman_gym
import gym_sokoban

def setup_env(folder, config):
    #####   Initialize env   #############
    env_name = config["env_type"]
    env_args = config["env_features"]

    env = gym.make(env_name, **env_args)

    env = Monitor(
        env,
        allow_early_resets=False,
        # reset_keywords=(),
        info_keywords=(["last_r"]),
    )
    return env

def main(folder, config):
    """
    Runs policy gradient with deep problog
    """
    #####   Read from config   #############
    # render = config["env_features"]["render"]

    random.seed(config["model_features"]["params"]["seed"])
    np.random.seed(config["model_features"]["params"]["seed"])
    th.manual_seed(config["model_features"]["params"]["seed"])

    #####   Initialize loggers   #############
    new_logger = configure(folder, ["stdout", "tensorboard"])


    env = setup_env(folder, config)

    grid_size = env.grid_size
    height = env.grid_height
    width = env.grid_weight
    color_channels = env.color_channels
    n_pixels = (height * grid_size) * (width * grid_size) * color_channels
    n_actions = env.action_size

    # #####   Initialize network   #############
    program_path = abspath(
        join("src", "data", f'{config["model_features"]["params"]["program_type"]}.pl')
    )

    image_encoder = Encoder(
        n_pixels,
        n_actions,
        config["model_features"]["params"]["shield"],
        config["model_features"]["params"]["detect_ghosts"],
        config["model_features"]["params"]["detect_walls"],
        program_path,
    )

    #####   Configure model   #############
    net_arch = config["model_features"]["params"]["net_arch_shared"] + [
        dict(
            pi=config["model_features"]["params"]["net_arch_pi"],
            vf=config["model_features"]["params"]["net_arch_vf"],
        )
    ]
    model = DPLPPO(
        DPLActorCriticPolicy,
        env,
        verbose=0,
        learning_rate=config["model_features"]["params"]["learning_rate"],
        n_steps=config["model_features"]["params"]["n_steps"],
        # n_steps: The number of steps to run for each environment per update
        batch_size=config["model_features"]["params"]["batch_size"],
        n_epochs=config["model_features"]["params"]["n_epochs"],
        gamma=config["model_features"]["params"]["gamma"],
        tensorboard_log=folder,
        _init_setup_model=True,
        policy_kwargs={
            "image_encoder": image_encoder,
            "net_arch": net_arch,
            "activation_fn": nn.ReLU,
            "optimizer_class": th.optim.Adam,
        },
        seed=config["model_features"]["params"]["seed"],
    )

    model.set_logger(new_logger)

    model.learn(total_timesteps=config["model_features"]["params"]["step_limit"])
    # model.learn(total_timesteps=500)
    model.save(join(folder, "model"))
