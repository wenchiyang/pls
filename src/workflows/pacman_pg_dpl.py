import random
import gym
import numpy as np
from logging import getLogger

from itertools import count

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from datetime import datetime
from os import path, getcwd
from os.path import abspath, join
import json
import cherry as ch
import cherry.envs as envs

from util import create_loggers, myformat

from dpl_policy import DPLSafePolicy, Encoder, PolicyNet
import pacman_gym


class Logger(envs.Logger):
    def __init__(
        self,
        env,
        interval=1000,
        episode_interval=10,
        title=None,
        logger=None,
        logger_raw=None,
    ):
        super(Logger, self).__init__(env, interval, episode_interval, title, logger)
        self.logger_raw = logger_raw

    def _episodes_length_rewards(self, rewards, dones):
        """
        When dealing with array rewards and dones (as for VecEnv) the length
        and rewards are only computed on the first dimension.
        (i.e. the first sub-process.)
        """
        episode_rewards = []
        episode_lengths = []
        accum = 0.0
        length = 0
        for r, d in zip(rewards, dones):
            if not isinstance(d, bool):
                d = bool(d.flat[0])
                r = float(r.flat[0])
            if not d:
                accum += r
                length += 1
            else:
                ### wenchi
                accum += r
                length += 1
                ###
                episode_rewards.append(accum)
                episode_lengths.append(length)
                accum = 0.0
                length = 0
        if length > 0:
            episode_rewards.append(accum)
            episode_lengths.append(length)
        return episode_rewards, episode_lengths

    def step(self, *args, **kwargs):
        state, reward, done, info = self.env.step(*args, **kwargs)
        self.all_rewards.append(reward)
        self.all_dones.append(done)
        self.num_steps += 1

        if self.interval > 0 and self.num_steps % self.interval == 0:
            msg, ep_stats, steps_stats = self.stats()
            if self.is_vectorized:
                info[0]["logger_steps_stats"] = steps_stats
                info[0]["logger_ep_stats"] = ep_stats
            else:
                info["logger_steps_stats"] = steps_stats
                info["logger_ep_stats"] = ep_stats
            self.logger.info(msg)
            self.logger.handlers[0].flush()
            self.logger.handlers[1].flush()
        if isinstance(done, bool):
            if done:
                self.num_episodes += 1
        else:
            self.num_episodes += sum(done)
        return state, reward, done, info


def update(replay, optimizer, GAMMA):

    policy_loss = []

    # Discount and normalize rewards
    rewards = ch.discount(GAMMA, replay.reward(), replay.done())
    rewards = ch.normalize(rewards)

    # Compute loss
    for sars, reward in zip(replay, rewards):
        log_prob = sars.log_prob
        policy_loss.append(-log_prob * reward)

    policy_loss_sum = th.stack(policy_loss).sum()

    # Take optimization step
    optimizer.zero_grad()
    policy_loss_sum.backward()
    optimizer.step()


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
    logger_info_name = config["info_logger"]
    logger_raw_name = config["raw_logger"]
    # timestamp = datetime.now().strftime("%Y%m%d_%H:%M")
    create_loggers(folder, [logger_info_name, logger_raw_name])

    logger_info = getLogger(logger_info_name)
    logger_raw = getLogger(logger_raw_name)

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
    env = Logger(env, interval=1000, logger=logger_info, logger_raw=logger_raw)
    env = envs.Torch(env)
    env.seed(config["env_features"]["seed"])

    grid_size = env.grid_size
    height = env.layout.height
    width = env.layout.width
    n_pixels = (height * grid_size) * (width * grid_size)
    n_actions = len(env.A)

    #####   Initialize network   #############

    program_path = abspath(
        join(
            getcwd(),
            "src",
            "data",
            f'{config["model_features"]["params"]["program_type"]}.pl',
        )
    )
    if config["model_features"]["params"]["shield"]:
        image_encoder = Encoder(
            n_pixels,
            n_actions,
            config["model_features"]["params"]["shield"],
            config["model_features"]["params"]["detect_ghosts"],
            config["model_features"]["params"]["detect_walls"],
            program_path,
            logger=logger_raw,
        )
        policy = DPLSafePolicy(image_encoder=image_encoder)
    else:
        policy = PolicyNet(n_pixels, n_actions)

    optimizer = optim.Adam(
        policy.parameters(), lr=config["model_features"]["params"]["learning_rate"]
    )

    replay = ch.ExperienceReplay()
    total_steps = 0
    for i_episode in count(1):
        state = env.reset()
        # draw(state[0])
        for t in range(100):  # Don't infinite loop while learning
            total_steps += 1
            if total_steps > step_limit:
                # save the model
                model_path = path.join(folder, "model")
                th.save(policy.state_dict(), model_path)
                break
            logger_raw.debug(f"---------  Step {total_steps}  ---------------")
            logger_raw.debug(f"State:          {myformat(state.data)}")

            # with th.no_grad():  # TODO
            shielded_probs = policy(state)
            mass = Categorical(probs=shielded_probs)
            action = mass.sample()
            log_prob = mass.log_prob(action)

            old_state = state
            state, reward, done, _ = env.step(action)

            replay.append(
                old_state,
                action,
                reward,
                state,
                done,
                # Cache log_prob for later
                log_prob=log_prob,
            )

            logger_raw.debug(f"Reward:         {reward}")
            logger_raw.debug(f"Done:           {done}")
            logger_raw.handlers[0].flush()
            if render:
                env.render()
            if done:
                break

        if total_steps > step_limit:
            break

        # Update policy
        update(replay, optimizer, gamma)
        replay.empty()


#
# exps_folder = abspath(join(getcwd(), "experiments"))
#
# exp = "grid2x2_1_ghost"
# types = ["pg_shield_detect"]
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
