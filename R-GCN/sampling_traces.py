import gym
import relenvs
from relenvs.envs.pacmanInterface import readCommand
import random
import numpy as np
from problog.logic import Term, Constant
import pickle


def coord_to_node_id(width, i, j):
    return j * width + i

def node_id_to_coord(width, node_id):
    return int(node_id) % width, int(node_id) // width


def relationize(state, action):
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
    for j, row in enumerate(walls):
        for i, c in enumerate(row):
            if c:
                relstate.append(Term("wall", Constant(coord_to_node_id(width, i, j))))

    # Add food(loc) atoms
    food = np.array(state.data.food.data).T
    for j, row in enumerate(food):
        for i, c in enumerate(row):
            if c:
                relstate.append(Term("food", Constant(coord_to_node_id(width, i, j))))

    # Add link(loc, loc) atoms
    for j in range(height):
        for i in range(width):
            loc = Constant(coord_to_node_id(width, i, j))
            neighboring_locs = [Constant(coord_to_node_id(width, ii, jj))
                                for (ii, jj) in [(i, j-1), (i-1, j), (i+1, j), (i, j+1)]
                                if 0 <= ii < width and 0 <= jj < height]
            links = [Term("link", Constant(loc), Constant(neighboring_loc)) for neighboring_loc in neighboring_locs]
            relstate += links

    # # Add available actions
    i, j = [i.configuration.pos for i in (state.data.agentStates)][0]
    pacman_loc = Constant(coord_to_node_id(width, i, j))
    # neighboring_locs = [Constant(coord_to_node_id(width, ii, jj))
    #                     for (ii, jj) in [(i, j - 1), (i - 1, j), (i + 1, j), (i, j + 1), (i, j)] # South, West, East, North, Stop
    #                     if 0 <= ii < width and 0 <= jj < height]
    # moves = [Term("move", Constant(pacman_loc), Constant(neighboring_loc)) for neighboring_loc in neighboring_locs]
    # relstate += moves

    # Add selected actions
    action_links = {
        "South": (i, j - 1),
        "West": (i - 1, j),
        "East": (i + 1, j),
        "North": (i, j + 1),
        "Stop": (i, j)
    }

    action_link = Constant(coord_to_node_id(width, action_links[action][0], action_links[action][1]))
    relaction = Term("move", Constant(pacman_loc), Constant(action_link))

    relstate.append(relaction)

    return relstate, relaction, width



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
            rel_current_state, rel_action, rel_width = relationize(current_state, action)
            trace.append((rel_current_state, rel_action, reward))
            current_state = next_state
        # # Add the last state of the trace
        # rel_current_state, rel_action, rel_width = relationize(current_state, action)
        # trace.append((rel_current_state, rel_action, reward))
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

def sample_no_env(num_states):
    from random import randrange
    height = 2
    width = 2
    traces = []
    relstates = []
    for n in range(num_states):
        relstate = []
        pacman_i = randrange(0, height)
        pacman_j = randrange(0, width)
        ghost_i = randrange(0, height)
        ghost_j = randrange(0, width)
        relstate.append(Term("pacman", Constant(coord_to_node_id(width, pacman_i, pacman_j))))
        relstate.append(Term("ghost", Constant(0), Constant(coord_to_node_id(width, ghost_i, ghost_j))))

        # Add link(loc, loc) atoms
        for j in range(height):
            for i in range(width):
                loc = Constant(coord_to_node_id(width, i, j))
                neighboring_locs = [Constant(coord_to_node_id(width, ii, jj))
                                    for (ii, jj) in [(i, j - 1), (i - 1, j), (i + 1, j), (i, j + 1)]
                                    if 0 <= ii < width and 0 <= jj < height]
                links = [Term("link", Constant(loc), Constant(neighboring_loc)) for neighboring_loc in neighboring_locs]
                for link in links:
                    relstate.append(link)

        relstates.append(relstate)

    traces.append(relstates)
    filename = f"random_{num_states}.p"
    pickle_to_file(filename, traces)
    return filename, width

def draw():
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    # G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    fig, ax = plt.subplots(figsize=(4,3))
    ax.axis("off")
    labels = {i: f"loc({i})" for i in range(4)}
    nx.draw_networkx(G, ax=ax, labels=labels,
                     node_color="#cee4f2", #["#ebe09d"] * len(node_name_dict[dsttype]),
                     edge_color="#a9acb0", arrows=False, node_size=500, font_size=8)
    plt.show()

if __name__ == "__main__":
    # draw()
    filename, rel_width = sample(sampling_episodes=20)
    # sampled_traces = unpickle_from_file(filename)
    # filename, rel_width = sample_no_env(10)
    # print("done")
