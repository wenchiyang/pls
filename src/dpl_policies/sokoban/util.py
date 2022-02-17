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
    floor_color
):

    input2 = input[:, :, :, 0]
    agent_colors2 = agent_colors[:, 0]
    floor_color2 = floor_color[0]
    obsacle_colors2 = obsacle_colors[:, 0]

    centers1 = (input2 == agent_colors2[0]).nonzero()[:, 1:]
    centers2 = (input2 == agent_colors2[1]).nonzero()[:, 1:]
    centers = th.cat((centers1, centers2))

    # r_limit2, c_limit2 = input2[0].size()[:2]
    padded_input2 = th.nn.functional.pad(input2, (2,2,2,2), "constant", 0) # pad the grid with 1 dimension with "0" (WALL_COLOR)
    padded_centers = centers + th.tensor([2,2]) # shift centers
    neighbor_centers =  th.stack(
        (
            th.stack(
                (
                    padded_centers + th.tensor((-2,  0)), # neighbor
                    padded_centers + th.tensor((-3,  0)), # forward
                    padded_centers + th.tensor((-2,  1)), # side
                    padded_centers + th.tensor((-2, -1))  # side
                 ),dim=1
            ),
            th.stack(
                (
                    padded_centers + th.tensor((0, -2)),  # neighbor
                    padded_centers + th.tensor((0, -3)),  # forward
                    padded_centers + th.tensor((1, -2)),  # side
                    padded_centers + th.tensor((-1, -2))  # side
                ), dim=1
            ),
            th.stack(
                (
                    padded_centers + th.tensor((0, 2)),  # neighbor
                    padded_centers + th.tensor((0, 3)),  # forward
                    padded_centers + th.tensor((1, 2)),  # side
                    padded_centers + th.tensor((-1, 2))  # side
                ),dim=1
            ),
            th.stack(
                (
                    padded_centers + th.tensor((2, 0)),  # neighbor
                    padded_centers + th.tensor((3, 0)),  # forward
                    padded_centers + th.tensor((2, 1)),  # side
                    padded_centers + th.tensor((2, -1))  # side
                ), dim=1
            ),
        ), dim=1)

    neighbor_values = padded_input2[
        th.arange(input2.size(0))[:,None,None],
        neighbor_centers[:, :, :, 0],
        neighbor_centers[:, :, :, 1]]

    neighbor_values[:, :, 0] = neighbor_values[:, :, 0] == floor_color2
    neighbor_values[:, :, 1:] = th.any(
        th.stack(
            (neighbor_values[:, :, 1:] == obsacle_colors2[0],
            neighbor_values[:, :, 1:] == obsacle_colors2[1],
            neighbor_values[:, :, 1:] == obsacle_colors2[2],
             neighbor_values[:, :, 1:] == obsacle_colors2[3]
            ), dim=0
        ), dim=0
    )

    neighbor_sides = th.any(
        neighbor_values[:,:,2:3], dim=2, keepdim=True
    )
    neighbor_center_forward_side = th.cat(
        (neighbor_values[:, :, :2], neighbor_sides)
        , dim=2
    )
    res = th.all(neighbor_center_forward_side,dim=2).float()
    return res

def get_ground_truth_of_box(
    input,
    agent_colors,
    box_colors
):
    input2 = input[:,:,:,0]
    agent_colors2 = agent_colors[:, 0]
    box_colors2 = box_colors[:, 0]
    centers1 = (input2 == agent_colors2[0]).nonzero()[:, 1:]
    centers2 = (input2 == agent_colors2[1]).nonzero()[:, 1:]
    centers = th.cat((centers1, centers2))

    neighbors = th.stack(
        (
            input2[th.arange(input2.size(0)), centers[:, 0] - 1, centers[:, 1]],
            input2[th.arange(input2.size(0)), centers[:, 0], centers[:, 1] - 1],
            input2[th.arange(input2.size(0)), centers[:, 0], centers[:, 1] + 1],
            input2[th.arange(input2.size(0)), centers[:, 0] + 1, centers[:, 1]]
        ), dim=1)
    res2 = th.any(
        th.stack((
            (neighbors == box_colors2[0]).float(),
            (neighbors == box_colors2[1]).float(),
            (neighbors == box_colors2[2]).float()
            ), dim=2)
        , dim=2).float()


    return res2

