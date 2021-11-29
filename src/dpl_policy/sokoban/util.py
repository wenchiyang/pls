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

    # levels = [logging.WARNING, logging.INFO, logging.DEBUG] + list(range(9, 0, -1))
    # verbose = max(0, min(len(levels) - 1, verbose))
    # logger = getLogger(name)
    # ch = logging.StreamHandler(sys.stdout)
    # formatter = logging.Formatter("[%(levelname)s] %(message)s")
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)
    # logger.setLevel(levels[verbose])


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


def get_ground_ghost(input, center_color, detect_color):
    # find center coord
    r, c = (input == center_color).nonzero(as_tuple=True)
    neighbors = [
        input[r - 1, c],  # up
        input[r + 1, c],  # down
        input[r, c - 1],  # left
        input[r, c + 1],  # right
    ]
    neigh = (th.stack(neighbors) == detect_color).float().view(-1)
    no_ghost = (1 - neigh.sum()).view(-1)
    res = th.cat((no_ghost, neigh)).view(1, -1)
    return res


def get_ground_relatives(input, center_colors, detect_colors, neighbors_relative_locs, out_of_boundary_value=False):
    centers = [th.tensor([center_color]*3, dtype=th.float32) for center_color in center_colors]
    detects = [th.tensor([detect_color]*3, dtype=th.float32) for detect_color in detect_colors]

    # find center coord: assuming there's only one
    for r, row in enumerate(input):
        for c, cell in enumerate(row):
            if any([(cell == center).all() for center in centers]):
                c_center = c
                r_center = r
    dim_r,dim_c = input.size()[:2]
    res = []
    for nr, nc in neighbors_relative_locs:
        # if coord is not valid
        if not (0 <= r_center+nr < dim_r and 0 <= c_center+nc < dim_c):
            res.append(out_of_boundary_value)
        elif any([(input[r_center+nr, c_center+nc] == detect).all() for detect in detects]):
            res.append(True)
        else:
            res.append(False)
    res = th.tensor(res).float().view(1, -1)
    return res
