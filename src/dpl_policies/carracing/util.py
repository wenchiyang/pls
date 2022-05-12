import logging
import sys
from logging import getLogger
import matplotlib.pyplot as plt
import os
import torch as th
from pathlib import Path
import numpy as np

def myformat(tensor):
    s = str(tensor)
    s = "".join(s.split())
    return s

def safe_max(arr) :
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr:
    :return:
    """
    return np.nan if len(arr) == 0 else np.max(arr)
def safe_min(arr) :
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr:
    :return:
    """
    return np.nan if len(arr) == 0 else np.min(arr)

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

    centers1 = th.isclose(input2, agent_colors2[0], atol=1e-03).nonzero()[:, 1:]
    centers2 = th.isclose(input2, agent_colors2[1], atol=1e-03).nonzero()[:, 1:]
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
                    padded_centers + th.tensor((2, 0)),  # neighbor
                    padded_centers + th.tensor((3, 0)),  # forward
                    padded_centers + th.tensor((2, 1)),  # side
                    padded_centers + th.tensor((2, -1))  # side
                ), dim=1
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

        ), dim=1)

    neighbor_values = padded_input2[
        th.arange(input2.size(0))[:,None,None],
        neighbor_centers[:, :, :, 0],
        neighbor_centers[:, :, :, 1]]

    neighbor_values[:, :, 0] = th.isclose(neighbor_values[:, :, 0], floor_color2, atol=1e-03)
    neighbor_values[:, :, 1:] = th.any(
        th.stack(
            (
                th.isclose(neighbor_values[:, :, 1:], obsacle_colors2[0], atol=1e-03),
                th.isclose(neighbor_values[:, :, 1:], obsacle_colors2[1], atol=1e-03),
                th.isclose(neighbor_values[:, :, 1:], obsacle_colors2[2], atol=1e-03)
            ), dim=0
        ), dim=0
    )

    neighbor_sides = th.any(
        neighbor_values[:,:,2:], dim=2, keepdim=True
    )
    neighbor_center_forward_side = th.cat(
        (neighbor_values[:, :, :2], neighbor_sides)
        , dim=2
    )
    res = th.all(neighbor_center_forward_side,dim=2).float()
    return res

def stuck(
        input,
        box_colors,
        obsacle_colors
):
    input2 = input[:, :, :, 0]
    box_color2 = box_colors[0]
    obsacle_colors2 = obsacle_colors[:, 0]
    centers1 = th.isclose(input2, box_color2, atol=1e-03).nonzero()[:, 1:]
    if centers1.nelement() == 0:
        return False
    centers = centers1

    neighbors = th.stack(
        (
            input2[th.arange(input2.size(0)), centers[:, 0] - 1, centers[:, 1]],
            input2[th.arange(input2.size(0)), centers[:, 0] + 1, centers[:, 1]],
            input2[th.arange(input2.size(0)), centers[:, 0], centers[:, 1] - 1],
            input2[th.arange(input2.size(0)), centers[:, 0], centers[:, 1] + 1],
        ), dim=1)
    res2 = th.any(
        th.stack(
            (
                th.isclose(neighbors, obsacle_colors2[0], atol=1e-03).float(),
                th.isclose(neighbors, obsacle_colors2[1], atol=1e-03).float(),
                th.isclose(neighbors, obsacle_colors2[2], atol=1e-03).float()
            ), dim=2)
        , dim=2).float()

    box_surrendings = res2

    # up, down, left, right,
    corners = th.tensor([
        [1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1], # up, left
        [1, 0, 0, 1], [1, 1, 0, 1], # up, right
        [0, 1, 1, 0], [0, 1, 1, 1], # down, left
        [0, 1, 0, 1] # down, right
    ])
    box_in_corner = False
    for i in range(box_surrendings.size(0)):
        if th.any(th.all((box_surrendings[i] == corners), dim=1)):
            box_in_corner = True

            # box_in_corner2 = th.any(th.all((box_surrendings[1] == corners), dim=1))
    return box_in_corner

def get_ground_truth_of_box(
    input,
    agent_colors,
    box_colors
):
    input2 = input[:,:,:,0]
    agent_colors2 = agent_colors[:, 0]
    box_colors2 = box_colors[:, 0]
    centers1 = th.isclose(input2, agent_colors2[0], atol=1e-03).nonzero()[:, 1:]
    centers2 = th.isclose(input2, agent_colors2[1], atol=1e-03).nonzero()[:, 1:]
    centers = th.cat((centers1, centers2))

    neighbors = th.stack(
        (
            input2[th.arange(input2.size(0)), centers[:, 0] - 1, centers[:, 1]],
            input2[th.arange(input2.size(0)), centers[:, 0] + 1, centers[:, 1]],
            input2[th.arange(input2.size(0)), centers[:, 0], centers[:, 1] - 1],
            input2[th.arange(input2.size(0)), centers[:, 0], centers[:, 1] + 1]
        ), dim=1)
    res2 = th.any(
        th.stack(
            (
            th.isclose(neighbors, box_colors2[0], atol=1e-03).float(),
            th.isclose(neighbors, box_colors2[1], atol=1e-03).float()
            ), dim=2)
        , dim=2).float()


    return res2

MY_BAR_COLOR =   1
MY_GRASS_COLOR = 0.6
MY_ROAD_COLOR =  -0.1
MY_CAR_COLOR =   1


is_grass = lambda x: x > 0
is_road = lambda x: th.logical_and(-0.15 < x, x < -0.05)


def is_all_grass(
        input
):
    arr = th.tensor(input)
    image = arr[:, 0:84, :]
    mean_color = th.mean(image)

    return mean_color > 0.55

def get_ground_truth_of_grass(
    input
):
    # arr = self.env.render(mode="state_pixels")

    # take only the first frame in the stack
    arr = th.squeeze(input[:,0,:,:], dim=1)

    # from matplotlib import pyplot as plt
    # temp = arr.clone()
    # temp[:, 70:71, 44:45] = 1  # left
    # temp[:, 70:71, 51:52] = 1  # right
    # temp[:, 64:65, 47:49] = 1  # top
    # plt.imshow(temp[0], cmap=plt.get_cmap('gray'), vmin=-1, vmax=1)
    # plt.show()

    left = th.mean(arr[:, 70:71, 42:43], dim=(1, 2))
    right = th.mean(arr[:, 70:71, 53:54], dim=(1, 2))
    top = th.mean(arr[:, 58:59, 47:49], dim=(1, 2))


    try:
        assert th.all(th.logical_or(is_grass(left), is_road(left))), f"left: {left}"
        assert th.all(th.logical_or(is_grass(right), is_road(right))), f"right: {right}"
        assert th.all(th.logical_or(is_grass(top), is_road(top))), f"top: {top}"
    except AssertionError:
        pass
        # print("AssertionError, get_ground_truth_of_grass failed.")
        # print(left, right, top)
    sym_state = is_grass(th.stack((top, left, right), dim=1)).float()

    return sym_state