import gym
import numpy as np
import pathlib, os
from Envs.envs import pacmanInterface

NUOF_EPISODES = 1
STEPS_PER_EPISODE = 10

ENV_NAME = 'pacman-v0'




# Get the environment and extract the number of actions.
args = [
    '--layout', 'smallGrid2',
    '--withoutShield', '1',
    '--pacman', 'ApproximateQAgent',
    '--numGhostTraining', '0',
    '--numTraining', '100',  # Training episodes
    '--numGames', '101'  # Total episodes
]
args = pacmanInterface.readCommand(args)



env = gym.make(ENV_NAME, **args)

env.episode_length = STEPS_PER_EPISODE
np.random.seed(123)
# env.seed(123)
nb_actions = len(env.A)
env.env.wrapper_run()






