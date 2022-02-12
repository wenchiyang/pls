import gym
import pacman_gym
from pacman_gym.envs.pacmanInterface import readCommand
import random
import numpy as np
from problog.logic import Term, Constant
import pickle


def coord_to_node_id(width, i, j):
    return j * width + i

def node_id_to_coord(width, node_id):
    return int(node_id) % width, int(node_id) // width


def relationize(width, height, agentStates, layoutwalls, layoutfood, action):
    relstate = []

    # Add agents atoms: pacman(loc) and ghost(id,loc)
    pacman_i, pacman_j = [i for i in (agentStates)][0]
    pacman_loc = Constant(coord_to_node_id(width, int(pacman_i), int(pacman_j)))
    ghost_i, ghost_j = [i for i in (agentStates)][1]
    ghost_loc = Constant(coord_to_node_id(width, int(ghost_i), int(ghost_j)))

    relstate.append(Term("pacman", Constant(pacman_loc)))
    relstate.append(Term("ghost", Constant(1), Constant(ghost_loc)))

    # for i, loc in enumerate([f for f in agentStates]): # i.configuration.pos
    #     if i == 0:
    #         relstate.append(Term("pacman", Constant(coord_to_node_id(width, loc[0], loc[1]))))
    #     else:
    #         relstate.append(Term("ghost", Constant(i), Constant(coord_to_node_id(width, loc[0], loc[1]))))

    # Add wall(loc) atoms
    walls = np.array(layoutwalls).T
    for j, row in enumerate(walls):
        for i, c in enumerate(row):
            if c:
                relstate.append(Term("wall", Constant(coord_to_node_id(width, i, j))))

    # Add food(loc) atoms
    food = np.array(layoutfood).T
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






    i = pacman_i
    j = pacman_j
    # Add available actions
    neighboring_locs = [Constant(coord_to_node_id(width, ii, jj))
                        for (ii, jj) in [(i, j - 1), (i - 1, j), (i + 1, j), (i, j + 1), (i, j)] # South, West, East, North, Stop
                        if 0 <= ii < width and 0 <= jj < height]
    moves = [Term("move", Constant(pacman_loc), Constant(neighboring_loc)) for neighboring_loc in neighboring_locs]
    relstate += moves

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

    # for testing
    if ghost_loc in neighboring_locs:
        bad_actions = [Term("move", Constant(pacman_loc), Constant(ghost_loc)),
                       Term("move", Constant(pacman_loc), Constant(pacman_loc))
                       ]
    else:
        # bad_actions = [Term("move", Constant(pacman_loc), Constant(pacman_loc))]
        bad_actions = []
    return relstate, relaction, width, bad_actions



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
            rel_current_state, rel_action, rel_width, bad_actions = \
                relationize(current_state.data.layout.width, current_state.data.layout.height,
                            current_state.data.agentStates, current_state.data.layout.walls.data,
                            current_state.data.food.data, action)
            trace.append((rel_current_state, rel_action, reward, bad_actions))
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

def sample_no_env(layout='smallGrid2', num_states=20):
    ENV_NAME = 'Pacman-v0'

    SIMPLE_ENV_ARGS = readCommand([
        '--layout', layout,
        '--withoutShield', '1',
        '--pacman', 'ApproximateQAgent',
        '--numGhostTraining', '0',
        '--numTraining', str(1),  # Training episodes
        '--numGames', str(1)  # Total episodes
    ])

    # Get an simple environment
    env = gym.make(ENV_NAME, **SIMPLE_ENV_ARGS)
    current_state = env.game.state


    from random import randrange
    height = current_state.data.layout.height
    width = current_state.data.layout.width
    traces = []
    trace = []

    walls = np.array(current_state.data.layout.walls.data).T
    wall_coords = []
    for j, row in enumerate(walls):
        for i, c in enumerate(row):
            if c:
                wall_coords.append((i, j))

    for n in range(num_states):
        pacman_i = int(randrange(0, width))
        pacman_j = int(randrange(0, height))
        while (pacman_i, pacman_j) in wall_coords:
            pacman_i = int(randrange(0, width))
            pacman_j = int(randrange(0, height))
        ghost_i = int(randrange(0, width))
        ghost_j = int(randrange(0, height))
        while (ghost_i, ghost_j) == (pacman_i, pacman_j) or (ghost_i, ghost_j) in wall_coords:
            ghost_i = int(randrange(0, width))
            ghost_j = int(randrange(0, height))

        layoutfood = current_state.data.food.data
        sampling_food = []
        for row in layoutfood:
            sampling_food_row = []
            for cell in row:
                if cell:
                    sampling_food_row.append(random.choice([True, False]))
                else:
                    sampling_food_row.append(False)
            sampling_food.append(sampling_food_row)


        rel_current_state, rel_action, rel_width, bad_actions = \
            relationize(width,
                        height,
                        [(pacman_i, pacman_j), (ghost_i, ghost_j)],
                        current_state.data.layout.walls.data,
                        layoutfood,
                        action='Stop')
        trace.append((rel_current_state, rel_action, 0, bad_actions))

    traces.append(trace)  # one trace
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
