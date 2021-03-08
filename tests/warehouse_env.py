import gym
import relenvs
from relenvs.envs import warehouseInterface

ENV_NAME = 'Warehouse-v0'

# Get the environment and extract the number of actions.
args = [
    '--layout', 'warehouse',
    '--withoutShield', '1',
    '--pacman', 'ApproximateQAgent',
    '--numGhostTraining', '0',
    '--numTraining', '100',  # Training episodes
    '--numGames', '101'  # Total episodes
]
args = warehouseInterface.readCommand(args)
env = gym.make(ENV_NAME, **args)


# Initial State
initial_state = env.game.state

# get all agents in the environment
agents = env.game.agents
num_agents = len(agents)
# agentIndex starts with 0
agents_index = 0


while not env.game.gameOver:
    legal_actions = env.get_legal_actions(agents_index)
    state, reward, is_gameOver, _ = env.step(agents_index, legal_actions[0])
    env.game.render()
    agents_index = (agents_index+1)%num_agents

env.game.end_game()