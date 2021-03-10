import gym
import relenvs
from relenvs.envs import pacmanInterface
import random

ENV_NAME = 'Pacman-v0'

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

# all actions
all_actions = env.A

# Initial State
initial_state = env.game.state


while not env.game.gameOver:
    action = random.choice(all_actions)
    state, reward, is_gameOver, _ = env.step(action)

env.game.end_game()







