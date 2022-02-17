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


def get_ground_wall(input, center_color, detect_color):
    # find center coord
    # r, c = (input == center_color).nonzero(as_tuple=True)
    #
    # neighbors = [
    #     input[r - 1, c],  # up
    #     input[r + 1, c],  # down
    #     input[r, c - 1],  # left
    #     input[r, c + 1],  # right
    # ]
    # res = (th.stack(neighbors) == detect_color).float().view(1, -1)
    # r, c = (input[0] == center_color).nonzero(as_tuple=True)
    # neighbors = [
    #     input[0][r - 1, c],  # up
    #     input[0][r + 1, c],  # down
    #     input[0][r, c - 1],  # left
    #     input[0][r, c + 1],  # right
    # ]
    # res = (th.stack(neighbors) == detect_color).float().view(1, -1)

    centers = (input == center_color).nonzero()[:, 1:]

    neighbors = th.stack(
        (
            input[th.arange(input.size(0)), centers[:, 0] - 1 , centers[:, 1]],
            input[th.arange(input.size(0)), centers[:, 0] + 1, centers[:, 1]],
            input[th.arange(input.size(0)), centers[:, 0], centers[:, 1] - 1],
            input[th.arange(input.size(0)), centers[:, 0], centers[:, 1] - 1]
        ), dim=1)

    res2 = (neighbors == detect_color).float()
    return res2
