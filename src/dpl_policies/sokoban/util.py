import logging
import sys
from logging import getLogger
import matplotlib.pyplot as plt
import os
import torch as th
from pathlib import Path


def myformat(tensor):
    s = str(tensor)
    s = "".join(s.split())
    return s


def create_loggers(folder, names):
    # folderpath = os.path.join(os.path.dirname(__file__), timestamp)
    Path(folder).mkdir(parents=True, exist_ok=True)
    for name in names:
        logger_file = os.path.join(folder, f"{name}.log")
        logf = open(logger_file, "w")
        init_logger(verbose=3, name=name, out=logf)
        # initial_log(name, args)
        if "raw" not in name:
            init_logger(verbose=3, name=name)


def init_logger(verbose=None, name="policy_gradient", out=None):
    """Initialize default logger.

    :param verbose: verbosity level (0: WARNING, 1: INFO, 2: DEBUG)
    :type verbose: int
    :param name: name of the logger (default: policy_gradient)
    :type name: str
    :return: result of ``logging.getLogger(name)``
    :rtype: logging.Logger
    """
    if out is None:
        out = sys.stdout
    logger = logging.getLogger(name)
    ch = logging.StreamHandler(out)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if not verbose:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
        logger.info("Output level: INFO")
    elif verbose == 2:
        logger.setLevel(logging.DEBUG)
        logger.debug("Output level: DEBUG")
    else:
        level = max(1, 12 - verbose)  # between 9 and 1
        logger.setLevel(level)
        logger.log(level, "Output level: %s" % level)


def draw(image):
    plt.axis("off")
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.show()


def initial_log(name, args):
    logger = getLogger(name)
    logger.info(f"Layout:           {args['layout']}")
    logger.info(f"Learning rate:    {args['learning_rate']}")
    logger.info(f"Shield:           {args['shield']}")
    logger.info(f"Object detection: {args['object_detection']}")
    logger.info(f"Reward goal:      {args['reward_goal']}")
    logger.info(f"Reward crash:     {args['reward_crash']}")
    logger.info(f"Reward food:      {args['reward_food']}")
    logger.info(f"Reward time:      {args['reward_time']}")
    logger.info(f"Step limit:       {args['step_limit']}")
    logger.info(f"Logger:           {args['logger_name']}")
    logger.info(f"Seed:             {args['seed']}")
    logger.info(f"Gamma:            {args['gamma']}")
    logger.info(f"Render:           {args['render']}")


def get_ground_truth_of_corners(
    input,
    agent_colors,
    obsacle_colors,
    floor_color,
    neighbors_relative_locs,
    out_of_boundary_value=False,
):
    # find agent's location: We assume there's only one agent
    for agent_color in agent_colors:
        agent_loc_r, agent_loc_c = (input == agent_color)[:, :, 0].nonzero(
            as_tuple=True
        )
        if agent_loc_r.numel() == 0:
            continue
        else:
            agent_loc_r = int(agent_loc_r)
            agent_loc_c = int(agent_loc_c)
            break

    r_limit, c_limit = input.size()[:2]

    res = []
    for rel_r, rel_c in neighbors_relative_locs:
        neighbors_abs_loc_r, neighbors_abs_loc_c = (agent_loc_r + rel_r, agent_loc_c + rel_c)
        if not in_bound((neighbors_abs_loc_r, neighbors_abs_loc_c), (r_limit, c_limit)):
            res.append(out_of_boundary_value)
        elif not (input[neighbors_abs_loc_r, neighbors_abs_loc_c] == floor_color)[0]:
            res.append(False) # # if it is not a floor, it cannot be a corner
        else:
            abs_forward_loc, abs_side_locs = get_corresponding_corner_locs(agent_loc=(agent_loc_r, agent_loc_c), dir=(rel_r, rel_c))
            res.append(is_corner(input, (r_limit, c_limit), abs_forward_loc, abs_side_locs, obsacle_colors))

    res = th.tensor(res).float().reshape(1, -1)
    return res

def in_bound(loc, bound):
    loc_r, loc_c = loc
    r_limit, c_limit = bound
    in_bound = 0 <= loc_r < r_limit and 0 <= loc_c < c_limit
    return in_bound

def get_corresponding_corner_locs(agent_loc, dir):
    d = {
        (0, 2): {"forward": (0, 3), "side": [(1, 2), (-1, 2)]},
        (-2, 0): {"forward": (-3, 0), "side": [(-2, 1), (-2, -1)]},
        (2, 0): {"forward": (3, 0), "side": [(2, 1), (2, -1)]},
        (0, -2): {"forward": (0, -3), "side": [(-1, -2), (1, -2)]},
    }
    rel_forward_loc = d[dir]["forward"]
    rel_side_locs = d[dir]["side"]
    abs_forward_loc = (agent_loc[0]+rel_forward_loc[0], agent_loc[1]+rel_forward_loc[1])
    abs_side_locs = [(agent_loc[0]+rel_side_loc[0], agent_loc[1]+rel_side_loc[1]) for rel_side_loc in rel_side_locs]
    return abs_forward_loc, abs_side_locs


def is_corner(input, bound, forward_loc, side_locs, obsacle_colors):
    forward_loc_r, forward_loc_c = forward_loc

    if in_bound(forward_loc, bound) and not any(
            (input[forward_loc_r, forward_loc_c] == obsacle_color)[0]
            for obsacle_color in obsacle_colors
        ):
        return False
    for side_loc_r, side_loc_c in side_locs:
        if any(
                (input[side_loc_r, side_loc_c] == obsacle_color)[0]
                for obsacle_color in obsacle_colors
        ):
            return True
    return False


def get_ground_truth_of_box(
    input,
    agent_colors,
    box_colors,
    neighbors_relative_locs,
    out_of_boundary_value=False,
):
    # find agent's location: We assume there's only one agent
    for agent_color in agent_colors:
        agent_loc_r, agent_loc_c = (input == agent_color)[:, :, 0].nonzero(
            as_tuple=True
        )
        if agent_loc_r.numel() == 0:
            continue
        else:
            agent_loc_r = int(agent_loc_r)
            agent_loc_c = int(agent_loc_c)
            break

    r_limit, c_limit = input.size()[:2]
    neighbors_abs_locs = [
        (agent_loc_r + rel_r, agent_loc_c + rel_c)
        for rel_r, rel_c in neighbors_relative_locs
    ]

    res = []
    for neighbors_abs_loc_r, neighbors_abs_loc_c in neighbors_abs_locs:
        if not in_bound((neighbors_abs_loc_r, neighbors_abs_loc_c), (r_limit, c_limit)):
            res.append(out_of_boundary_value)
        elif any(
            (input[neighbors_abs_loc_r, neighbors_abs_loc_c] == box_color)[0]
            for box_color in box_colors
        ):
            res.append(True)
        else:
            res.append(False)
    res = th.tensor(res).float().reshape(1, -1)
    return res
