
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

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


class Logger(envs.Logger):
    def __init__(self, env, interval=1000, episode_interval=10, title=None, logger=None):
        super(Logger, self).__init__(env, interval=1000, episode_interval=10, title=None, logger=None)

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
        self.affine1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=4, stride=(1,1), padding="same")
        self.affine2 = nn.Linear(input_size*input_size*12, output_size)
        # self.affine1 = nn.Linear(input_size, 128)
        # self.affine2 = nn.Linear(128, output_size)

    def forward(self, x):
        xx = x.reshape(1, 1, x.shape[-2], x.shape[-1])
        xx = F.relu(self.affine1(xx))
        xx = th.flatten(xx, 1)
        action_scores = F.relu(self.affine2(xx))
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
    # Pick an layout from relenvs_pip/relenvs/envs/pacman/layouts
    layout = 'testGrid'
    sampling_episodes = 1

    from relenvs.envs.pacmanInterface import readCommand

    SIMPLE_ENV_ARGS = readCommand([
        '--layout', layout,
        '--withoutShield', '1',
        '--pacman', 'ApproximateQAgent',
        '--numGhostTraining', '0',
        '--numTraining', str(sampling_episodes),  # Training episodes
        '--numGames', str(sampling_episodes)  # Total episodes
    ])
    args = SIMPLE_ENV_ARGS



    env = gym.make(env_name, **args)

    grid_size = env.grid_size
    height = env.layout.height
    width = env.layout.width

    # input_size = grid_size * height * grid_size * width
    input_size = height * grid_size
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
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    running_reward = 400
    replay = ch.ExperienceReplay()

    for i_episode in count(1):
        state = env.reset()
        # draw(state[0])
        for t in range(100):  # Don't infinite loop while learning
            mass = Categorical(policy(state))
            action = mass.sample()
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