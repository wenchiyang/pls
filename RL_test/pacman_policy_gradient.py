import relenvs
import random
import gym
import numpy as np

from itertools import count

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import cherry as ch
import cherry.envs as envs
import matplotlib.pyplot as plt

SEED = 567
GAMMA = 0.99
RENDER = False
learning_rate = 1e-3

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)

LAYOUT = 'testGrid'# Pick an layout from relenvs_pip/relenvs/envs/pacman/layouts
REWARD_GOAL = 10
REWARD_DIE = 0
REWARD_FOOD = 0
REWARD_TIME = -1


class Logger(envs.Logger):
    def __init__(self, env, interval=1000, episode_interval=10, title=None, logger=None):
        super(Logger, self).__init__(env, interval, episode_interval, title, logger)

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



def draw(image):
    # image_rgb
    # cv2.imshow("image", image)
    plt.axis("off")
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.show()

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


def update(replay):
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



if __name__ == '__main__':
    # Wrap environments
    env_name = 'Pacman-v0'

    args = {
        'layout': LAYOUT,
        'reward_goal': str(REWARD_GOAL),
        'reward_crash': str(REWARD_DIE),
        'reward_food': str(REWARD_FOOD),
        'reward_time': str(REWARD_TIME)
    }

    env = gym.make(env_name, **args)

    grid_size = env.grid_size
    height = env.layout.height
    width = env.layout.width

    input_size = (height * grid_size) * (width * grid_size)
    output_size = 5



    # for CartPole-v0
    # env_name = 'CartPole-v0'
    # args = {}
    # env = gym.make(env_name, **args)
    # input_size = 4
    # output_size = 2

    env = Logger(env, interval=1000)
    env = envs.Torch(env)
    env.seed(SEED)

    policy = PolicyNet(input_size, output_size)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    running_reward = 400
    replay = ch.ExperienceReplay()


    for i_episode in count(1):
        state = env.reset()
        # draw(state[0])
        for t in range(100):  # Don't infinite loop while learning
            probs = policy(state, T=1)
            mass = Categorical(probs=probs)
            # stochastic policy
            action = mass.sample()
            # deterministic policy + epsilon greedy
            # if random.random() > epsilon:
            #     action = th.argmax(mass.probs)
            # else:
            #     action = random.choice(th.tensor(range(5)))

            old_state = state
            state, reward, done, _ = env.step(action)
            # draw(state[0])
            replay.append(old_state,
                          action,
                          reward,
                          state,
                          done,
                          # Cache log_prob for later
                          log_prob=mass.log_prob(action))
            if RENDER:
                env.render()
            if done:
                break

        #  Compute termination criterion
        # running_reward = running_reward * 0.99 + t * 0.01
        # # if running_reward > env.spec.reward_threshold:
        # if running_reward > 200:
        #     print('Solved! Running reward is now {} and '
        #           'the last episode runs to {} time steps!'.format(running_reward, t))
        #     break

        running_reward = running_reward * 0.99 + reward * 0.01
        if running_reward > 450:
            print('Solved! Running reward is now {} and '
                  'the last episode runs to {} time steps!'.format(running_reward, t))
            break

        # Update policy
        update(replay)
        replay.empty()