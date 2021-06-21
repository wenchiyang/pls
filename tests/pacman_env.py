import gym
from relenvs.envs.pacmanInterface import readCommand
import random

ENV_NAME = 'Pacman-v0'

# Pick an layout from relenvs_pip/relenvs/envs/pacman/layouts
layout='testGrid'
sampling_episodes = 1

SIMPLE_ENV_ARGS = readCommand([
        '--layout', layout,
        '--withoutShield', '1',
        '--pacman', 'ApproximateQAgent',
        '--numGhostTraining', '0',
        '--numTraining', str(sampling_episodes),  # Training episodes
        '--numGames', str(sampling_episodes)  # Total episodes
    ])


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME, **SIMPLE_ENV_ARGS)
env.render()

# all actions
all_actions = env.A

# Initial State
initial_state = env.game.state


while not env.game.gameOver:
    action = random.choice(all_actions)
    state_rel, reward, is_gameOver, _ = env.step(action)
    # One can choose to use an image as an input as well
    state_image = env.render(mode="rgb_array")


env.game.end_game()







