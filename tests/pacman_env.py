import gym
import relenvs
from relenvs.envs import pacmanInterface

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


# Initial State
initial_state = env.game.state

while not env.game.gameOver:
    legal_actions = env.get_legal_actions()
    state, reward, is_gameOver, _ = env.step(legal_actions[0])
    env.game.render()

env.game.end_game()







