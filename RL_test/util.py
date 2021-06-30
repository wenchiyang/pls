import logging
import sys
from logging import getLogger
import matplotlib.pyplot as plt

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
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
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
    return logger

def create_logger(name, verbose):
    levels = [logging.WARNING, logging.INFO, logging.DEBUG] + list(range(9, 0, -1))
    verbose = max(0, min(len(levels) - 1, verbose))
    logger = getLogger(name)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(levels[verbose])


def draw(image):
    plt.axis("off")
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.show()

def initial_log(name, layout, map_seed, reward_goal, reward_crach, reward_food, reward_time,
                shield, lr):
    logger = getLogger(name)
    logger.info(f"Map: {layout}")
    logger.info(f"Map seed: {map_seed}")
    logger.info(f"Reward structure: ")
    logger.info(f"    goal = {reward_goal}")
    logger.info(f"    crash = {reward_crach}")
    logger.info(f"    food = {reward_food}")
    logger.info(f"    time = {reward_time}")
    logger.info(f"Shielded type: {shield['type']}")
    if shield['type'] != "None":
        logger.info(f"    Object detection: {shield['object_detection']}")
    logger.info(f"    Seed: {shield['seed']}")
    logger.info(f"    Learning rate: {lr}")
