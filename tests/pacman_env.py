import gym
from pacman_gym.envs.pacmanInterface import readCommand
import random
# import cv2
import matplotlib.pyplot as plt
# import numpy as np

def draw(image):
    plt.axis("off")
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.show()


ENV_NAME = 'Pacman-v0'

# Pick an layout from pacman_gym/pacman_gym/envs/pacman/layouts
layout='grid2x2'
sampling_episodes = 1

SIMPLE_ENV_ARGS = readCommand([
        '--layout', layout,
        '--withoutShield', '1',
        '--pacman', 'ApproximateQAgent',
        '--numGhostTraining', '0',
        '--numTraining', str(sampling_episodes),  # Training episodes
        '--numGames', str(sampling_episodes)  # Total episodes
    ])

env_args = {
        "layout": "grid2x2",
        "seed": 456,
        "reward_goal": 10,
        "reward_crash": -10,
        "reward_food": 0,
        "reward_time": -1
    }

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME, **env_args)

# all actions
all_actions = env.A

state = env.reset()
while not env.game.gameOver:
    action = random.choice(range(5))
    state, reward, is_gameOver, _ = env.step(action)

    env.render()

env.game.end_game()







