import random
import gym
import numpy as np
from logging import getLogger

from itertools import count

import torch as th
import torch.optim as optim
from torch.distributions import Categorical
from os.path import join, abspath
from os import getcwd
import json

from dpl_policy_stable_baselines import DPLActorCriticPolicy, Encoder, DPLPolicyGradientPolicy
import relenvs
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


def main(folder, config):
    """
    Runs policy gradient with deep problog
    """
    #####   Read from config   #############
    step_limit = config["model_features"]["params"]["step_limit"]
    render = config["env_features"]["render"]
    gamma = config["model_features"]["params"]["gamma"]

    random.seed(config["model_features"]["params"]["seed"])
    np.random.seed(config["model_features"]["params"]["seed"])
    th.manual_seed(config["model_features"]["params"]["seed"])

    #####   Initialize loggers   #############
    new_logger = configure(folder, ["stdout", "tensorboard"])

    # logger_info_name = config["info_logger"]
    # logger_raw_name = config["raw_logger"]
    # create_loggers(folder, [logger_info_name, logger_raw_name])
    #
    # logger_info = getLogger(logger_info_name)
    # logger_raw = getLogger(logger_raw_name)

    #####   Initialize env   #############
    env_name = "Pacman-v0"
    env_args = {
        "layout": config["env_features"]["layout"],
        "seed": config["env_features"]["seed"],
        "reward_goal": config["env_features"]["reward_goal"],
        "reward_crash": config["env_features"]["reward_crash"],
        "reward_food": config["env_features"]["reward_food"],
        "reward_time": config["env_features"]["reward_time"],
    }

    env = gym.make(env_name, **env_args)
    eval_env = gym.make(env_name, **env_args)

    # logger_raw_filename = join(folder, f'{config["info_logger"]}.log')
    # tb_log_filename = join(folder, f'{config["info_logger"]}_tb.log')
    env = Monitor(
        env,
        # filename=folder,
        allow_early_resets=False,
        # reset_keywords=(),
        # info_keywords=(),
    )

    grid_size = env.grid_size
    height = env.layout.height
    width = env.layout.width
    n_pixels = (height * grid_size) * (width * grid_size)
    n_actions = len(env.A)


    # #####   Initialize network   #############
    program_path = abspath(join("src", "data", f'{config["model_features"]["params"]["program_type"]}.pl'))
    image_encoder = Encoder(
        n_pixels,
        n_actions,
        config["model_features"]["params"]["shield"],
        config["model_features"]["params"]["detect_ghosts"],
        config["model_features"]["params"]["detect_walls"],
        program_path
    )

    #####   Configure model   #############
    model = PPO(
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
            "net_arch": [dict(pi=config["model_features"]["params"]["net_arch_pi"],
                              vf=config["model_features"]["params"]["net_arch_vf"])]
        },
        seed = config["model_features"]["params"]["seed"]
    )


    model.set_logger(new_logger)

    model.learn(
        total_timesteps=config["model_features"]["params"]["step_limit"]
    )


#
# exps_folder = abspath(join(getcwd(), "experiments"))
#
#
# exp = "grid2x2_1_ghost"
# types = ["ppo"]
#
# for type in types:
#     folder = join(exps_folder, exp, type)
#
#     path = join(folder, "config.json")
#     with open(path) as json_data_file:
#         config = json.load(json_data_file)
#
#     # example(folder, config)
#     main(folder, config)
