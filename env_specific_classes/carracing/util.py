import logging
import os
import sys
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th


def safe_max(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr:
    :return:
    """

    return np.nan if len(arr) == 0 else np.max(arr)


def safe_min(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr:
    :return:
    """

    return np.nan if len(arr) == 0 else np.min(arr)


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
    logger.info(f"Step limit:       {args['total_timesteps']}")
    logger.info(f"Logger:           {args['logger_name']}")
    logger.info(f"Seed:             {args['seed']}")
    logger.info(f"Gamma:            {args['gamma']}")
    logger.info(f"Render_or_not:    {args['render_or_not']}")


MY_BAR_COLOR = 1
MY_GRASS_COLOR = 0.6
MY_ROAD_COLOR = -0.1
MY_CAR_COLOR = 1

is_road = lambda x: th.logical_and(-0.15 < x, x < -0.05)
is_grass = lambda x: ~is_road(x)


def get_ground_truth_of_grass(input):
    """
    A heuristic to compute the ground truth of the presence of grass around the agent.

    :param input: tensor representing the states
    :return: tensor of {0, 1} representing the presence of grass on top, left and right
    """

    # take only the first frame in the stack
    arr = th.squeeze(input[:, 0, :, :], dim=1)

    left = th.mean(arr[:, 33:34, 22:23], dim=(1, 2))
    right = th.mean(arr[:, 33:34, 25:26], dim=(1, 2))
    top = th.mean(arr[:, 27:28, 23:25], dim=(1, 2))

    try:
        assert th.all(th.logical_or(is_grass(left), is_road(left))), f"left: {left}"
        assert th.all(th.logical_or(is_grass(right), is_road(right))), f"right: {right}"
        assert th.all(th.logical_or(is_grass(top), is_road(top))), f"top: {top}"
    except AssertionError:
        pass
        print("AssertionError, get_ground_truth_of_grass failed.")
        print(left, right, top)

    sym_state = is_grass(th.stack((top, left, right), dim=1)).float()

    return sym_state
