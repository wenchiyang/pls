import gym
import relenvs
from relenvs.envs.pacmanInterface import SIMPLE_ENV_ARGS
import random

ENV_NAME = 'Pacman-v0'

# Get the environment and extract the number of actions.

env = gym.make(ENV_NAME, **SIMPLE_ENV_ARGS)

# all actions
all_actions = env.A

# Initial State
initial_state = env.game.state


while not env.game.gameOver:
    action = random.choice(all_actions)
    state, reward, is_gameOver, _ = env.step(action)

env.game.end_game()







