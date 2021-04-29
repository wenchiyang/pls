import gym
import relenvs
from relenvs.envs.pacmanInterface import readCommand
import random
import numpy as np
from problog.logic import Term, Constant
import pickle


def coord_to_node_id(width, i, j):
    return i * width + j

def node_id_to_coord(width, node_id):
    return int(node_id) // width, int(node_id) % width


def relationize(state):
    relstate = []
    width = state.data.layout.width
    height = state.data.layout.height

    # Add agents atoms: pacman(loc) and ghost(id,loc)
    for i, loc in enumerate([i.configuration.pos for i in (state.data.agentStates)]):
        if i == 0:
            relstate.append(Term("pacman", Constant(coord_to_node_id(width, loc[0], loc[1]))))
        else:
            relstate.append(Term("ghost", Constant(i), Constant(coord_to_node_id(width, loc[0], loc[1]))))

    # Add wall(loc) atoms
    walls = np.array(state.data.layout.walls.data).T
    for i, row in enumerate(walls):
        for j, c in enumerate(row):
            if c:
                relstate.append(Term("wall", Constant(coord_to_node_id(width, i, j))))

    # Add food(loc) atoms
    food = np.array(state.data.layout.food.data).T
    for i, row in enumerate(food):
        for j, c in enumerate(row):
            if c:
                relstate.append(Term("food", Constant(coord_to_node_id(width, i, j))))

    # Add link(loc, loc) atoms
    for i in range(height):
        for j in range(width):
            loc = Constant(coord_to_node_id(width, i, j))
            neighboring_locs = [Constant(coord_to_node_id(width, ii, jj))
                                for (ii, jj) in [(i, j-1), (i-1, j), (i+1, j), (i, j+1)]
                                if 0 <= ii < height and 0 <= jj < width]
            links = [Term("link", Constant(loc), Constant(neighboring_loc)) for neighboring_loc in neighboring_locs]
            for link in links:
                relstate.append(link)

    return relstate, width



def sample(layout='smallGrid2', sampling_episodes=200):
    ENV_NAME = 'Pacman-v0'

    SIMPLE_ENV_ARGS = readCommand([
        '--layout', layout,
        '--withoutShield', '1',
        '--pacman', 'ApproximateQAgent',
        '--numGhostTraining', '0',
        '--numTraining', str(sampling_episodes),  # Training episodes
        '--numGames', str(sampling_episodes)  # Total episodes
    ])

    # Get an simple environment
    env = gym.make(ENV_NAME, **SIMPLE_ENV_ARGS)
    # all actions
    all_actions = env.A

    traces = []
    for i in range(sampling_episodes):
        # initialize a trace
        trace = []
        current_state = env.game.state
        while not env.game.gameOver:
            action = random.choice(all_actions)
            next_state, reward, is_gameOver, _ = env.step(action)
            rel_current_state, rel_width = relationize(current_state)
            trace.append((rel_current_state, action, reward))
            current_state = next_state
        # Add the last state of the trace
        rel_current_state, rel_width = relationize(current_state)
        trace.append((rel_current_state, action, reward))
        traces.append(trace)
        env.game.end_game()
        env.reset()

    filename = layout+"_traces_"+str(sampling_episodes)+".p"
    pickle_to_file(filename, traces)
    return filename, rel_width # rel_width is for calculating the location connections later

def pickle_to_file(filename, pickled):
    outfile = open(filename, 'wb')
    pickle.dump(pickled, outfile)
    outfile.close()

def unpickle_from_file(filename):
    infile = open(filename, 'rb')
    pickled = pickle.load(infile)
    infile.close()
    return pickled

if __name__ == "__main__":
    filename, rel_width = sample(sampling_episodes=20)
    # sampled_traces = unpickle_from_file(filename)
    print("done")
