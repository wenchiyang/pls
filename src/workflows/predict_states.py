import torch as th
from dpl_policy.pg.pacman_pg import Encoder, DPLSafePolicy, PolicyNet
import os
import random
import numpy as np
import gym
import cherry.envs as envs
from itertools import count
from torch.distributions import Categorical


def load(folder, config):
    """
    todo
    """
    #####   Read from config   #############
    render = config["env_features"]["render"]

    random.seed(config["model_features"]["params"]["seed"])
    np.random.seed(config["model_features"]["params"]["seed"])
    th.manual_seed(config["model_features"]["params"]["seed"])

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
    # env = Logger(env, interval=1000, logger=logger_info, logger_raw=logger_raw)
    env = envs.Torch(env)
    env.seed(config["env_features"]["seed"])

    grid_size = env.grid_size
    height = env.layout.height
    width = env.layout.width
    n_pixels = (height * grid_size) * (width * grid_size)
    n_actions = len(env.A)

    #####   Load network   #############
    path = os.path.join(folder, "model")
    if config["model_features"]["params"]["shield"]:
        image_encoder = Encoder(
            n_pixels,
            n_actions,
            config["model_features"]["params"]["shield"],
            config["model_features"]["params"]["detect_ghosts"],
            config["model_features"]["params"]["detect_walls"],
            # logger=logger_raw,
        )
        policy = DPLSafePolicy(image_encoder=image_encoder)
        policy.load_state_dict(th.load(path))
    else:
        policy = PolicyNet(n_pixels, n_actions)
        policy.load_state_dict(th.load(path))

    return env, policy


def main(folder, config):
    env, policy = load(folder, config)
    render = config["env_features"]["render"]

    state = env.reset()
    for total_steps in count(1):
        shielded_probs = policy(state)
        mass = Categorical(probs=shielded_probs)
        action = mass.sample()
        state, reward, done, _ = env.step(action)
        # draw(state[0])
        if render:
            env.render()
