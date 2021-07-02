import relenvs
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

import cherry as ch
import cherry.envs as envs
import matplotlib.pyplot as plt

from RL_test.util import create_loggers
from RL_test.util import draw as draw
import os

from RL_test.dpl_policy import DPLSafePolicy




class Logger(envs.Logger):
    def __init__(self, env, interval=1000, episode_interval=10, title=None, logger=None, logger_raw=None):
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
        # self.logger_raw.debug((reward, done))

        if self.interval > 0 and self.num_steps % self.interval == 0:
            msg, ep_stats, steps_stats = self.stats()
            if self.is_vectorized:
                info[0]['logger_steps_stats'] = steps_stats
                info[0]['logger_ep_stats'] = ep_stats
            else:
                info['logger_steps_stats'] = steps_stats
                info['logger_ep_stats'] = ep_stats
            self.logger.info(msg)
        if isinstance(done, bool):
            if done:
                self.num_episodes += 1
        else:
            self.num_episodes += sum(done)
        return state, reward, done, info


def draw(image):
    # image_rgb
    # cv2.imshow("image", image)
    plt.axis("off")
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.show()


class Encoder(nn.Module):
    def __init__(self, input_size, n_actions, relevant_grid_width, relevant_grid_height, object_detection, logger):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.relevant_grid_width = relevant_grid_width
        self.relevant_grid_height = relevant_grid_height
        self.object_detection = object_detection
        self.n_actions = n_actions
        self.logger = logger
        # hidden_size = 20
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),
        # )
        # self.hidden_size = hidden_size

    def forward(self, x):
        xx = th.flatten(x, 1)
        # hidden = self.encoder(xx)
        return xx

class PolicyNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x, T=1):
        xx = th.flatten(x, 1)
        action_scores = self.network(xx)
        return F.softmax(action_scores, dim=1)

def update(replay, optimizer, GAMMA):
    policy_loss = []

    # Discount and normalize rewards
    rewards = ch.discount(GAMMA, replay.reward(), replay.done())
    rewards = ch.normalize(rewards)

    # Compute loss
    for sars, reward in zip(replay, rewards):
        log_prob = sars.log_prob
        policy_loss.append(-log_prob * reward)

    # Take optimization step
    optimizer.zero_grad()
    policy_loss = th.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()



def main(args):
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    th.manual_seed(args['seed'])

    #####   Initialize loggers   #############
    logger_info_name = f"{args['logger_name']}_{args['layout']}"
    logger_raw_name = f"{args['logger_name']}_{args['layout']}_raw"
    create_loggers([logger_info_name, logger_raw_name], args, timestamp=args['timestamp'])

    logger_info = getLogger(logger_info_name)
    logger_raw = getLogger(logger_raw_name)

    #####   Initialize env   #############
    env_name = 'Pacman-v0'
    env_args = {
        'layout': args['layout'],
        'seed' : args['seed'],
        'reward_goal': args['reward_goal'],
        'reward_crash': args['reward_crash'],
        'reward_food': args['reward_food'],
        'reward_time': args['reward_time'],
    }

    env = gym.make(env_name, **env_args)
    env = Logger(env, interval=1000, logger=logger_info, logger_raw=logger_raw)
    env = envs.Torch(env)
    env.seed(args['seed'])

    grid_size = env.grid_size
    height = env.layout.height
    width = env.layout.width
    n_pixels = (height * grid_size) * (width * grid_size)
    n_actions = len(env.A)

    symbolic_grid_width = args['symbolic_grid_width']
    symbolic_grid_height = args['symbolic_grid_height']
    assert symbolic_grid_width < height
    assert symbolic_grid_height < width

    #####   Initialize network   #############
    if args['shield']:
        image_encoder = Encoder(n_pixels, n_actions, symbolic_grid_width, symbolic_grid_height, args['object_detection'], logger=logger_raw)
        policy = DPLSafePolicy(image_encoder=image_encoder)
        optimizer = optim.Adam(policy.parameters(), lr=args['learning_rate'])
    else:
        policy = PolicyNet(n_pixels, n_actions)
        optimizer = optim.Adam(policy.parameters(), lr=args['learning_rate'])

    running_reward = 400
    replay = ch.ExperienceReplay()

    total_steps = 0

    for i_episode in count(1):
        state = env.reset()
        # draw(state[0])
        for t in range(100):  # Don't infinite loop while learning
            total_steps += 1
            if total_steps > args['step_limit']: break
            logger_raw.debug(f"---------  Step {total_steps}  ---------------")
            # with th.no_grad():  # TODO
            probs = policy(state)
            mass = Categorical(probs=probs)
            action = mass.sample()
            log_prob = mass.log_prob(action)

            old_state = state
            state, reward, done, _ = env.step(action)

            replay.append(old_state,
                          action,
                          reward,
                          state,
                          done,
                          # Cache log_prob for later
                          log_prob=log_prob)

            logger_raw.debug(f"Reward:         {reward}")
            logger_raw.debug(f"Done:           {done}")

            if args['render']: env.render()
            if done: break
        if total_steps > args['step_limit']:
            break

        #  Compute termination criterion
        # running_reward = running_reward * 0.99 + t * 0.01
        # # if running_reward > env.spec.reward_threshold:
        # if running_reward > 200:
        #     print('Solved! Running reward is now {} and '
        #           'the last episode runs to {} time steps!'.format(running_reward, t))
        #     break

        # running_reward = running_reward * 0.99 + reward * 0.01
        # if running_reward > 450:
        #     print('Solved! Running reward is now {} and '
        #           'the last episode runs to {} time steps!'.format(running_reward, t))
        #     break

        # Update policy
        update(replay, optimizer, args['gamma'])
        replay.empty()


# if __name__ == '__main__':
#     from datetime import datetime
#     setting1 = {
#         'layout': 'grid2x2',  # Pick an layout from relenvs_pip/relenvs/envs/pacman/layouts
#         'learning_rate': 1e-3,
#         'reward_goal': 10,
#         'reward_crash': 0,
#         'reward_food': 0,
#         'reward_time': -1,
#         'step_limit': 40000,
#         'seed': 567,
#         'gamma': 0.99,
#         'render': False,
#         'timestamp': datetime.now().strftime('%Y%m%d_%H:%M')
#     }
#
#     shared_args = setting1
#
#     pg_args = {
#         'layout': shared_args['layout'],
#         'learning_rate': shared_args['learning_rate'],
#         'shield': False,
#         'object_detection': None,
#         'reward_goal': shared_args['reward_goal'],
#         'reward_crash': shared_args['reward_crash'],
#         'reward_food': shared_args['reward_food'],
#         'reward_time': shared_args['reward_time'],
#         'step_limit': shared_args['step_limit'],
#         'logger_name': 'pg',
#         'seed': shared_args['seed'],
#         'gamma': shared_args['gamma'],
#         'render': shared_args['render'],
#         'timestamp': shared_args['timestamp']
#     }
#
#     main(pg_args)