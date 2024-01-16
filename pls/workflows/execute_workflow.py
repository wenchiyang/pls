from pls.algorithms.learn import main as learn_ppo
from pls.algorithms.pretrain import main as pretrain
from pls.algorithms.evaluate import main as evaluate_policy
import os
import math
import json

from env_specific_classes.pacman.env_classes import (
    Pacman_Monitor,
    Pacman_Callback,
    Pacman_FeaturesExtractor,
    Pacman_Observation_Net,
)
from env_specific_classes.carracing.env_classes import (
    Carracing_Monitor,
    Carracing_Callback,
    Carracing_FeaturesExtractor,
    Carracing_Observation_Net,
)
from env_specific_classes.carracing.util import get_ground_truth_of_grass
from env_specific_classes.pacman.util import get_ground_wall
from functools import partial
from pls.algorithms.ppo_shielded import PPO_shielded


def get_env_classes(env_name, ghost_distance=1):
    """
    Given env_name, return the appropriate classes for the environment.
    :return:
    """
    # Pacman color encoding
    WALL_COLOR = 0.25
    GHOST_COLOR = 0.5
    PACMAN_COLOR = 0.75
    FOOD_COLOR = 1

    classes = {
        "Pacman-v0": {
            "model_cls": PPO_shielded,
            "get_sensor_value_ground_truth": partial(
                get_ground_wall, ghost_distance, PACMAN_COLOR, GHOST_COLOR
            ),
            "custom_callback_class": Pacman_Callback,
            "monitor_cls": Pacman_Monitor,
            "features_extractor_cls": Pacman_FeaturesExtractor,
            "observation_net_cls": Pacman_Observation_Net,
        },
        "CarRacingPLS-v1": {
            "model_cls": PPO_shielded,
            "get_sensor_value_ground_truth": get_ground_truth_of_grass,
            "custom_callback_class": Carracing_Callback,
            "monitor_cls": Carracing_Monitor,
            "features_extractor_cls": Carracing_FeaturesExtractor,
            "observation_net_cls": Carracing_Observation_Net,
        },
    }
    env_classes = classes[env_name]
    return (
        env_classes["model_cls"],
        env_classes["get_sensor_value_ground_truth"],
        env_classes["custom_callback_class"],
        env_classes["monitor_cls"],
        env_classes["features_extractor_cls"],
        env_classes["observation_net_cls"],
    )


def evaluate(config_file, model_at_step, n_test_episodes):
    config_folder = os.path.dirname(config_file)
    with open(config_file) as json_data_file:
        config = json.load(json_data_file)

    if config["env"] == "Pacman-v0":
        ghost_distance = config["policy_safety_params"]["ghost_distance"]
    else:
        ghost_distance = None
    (_, _, _, monitor_cls, _, _) = get_env_classes(config["env"], ghost_distance)

    if "ppo" == config["base_policy"]:
        return evaluate_policy(
            config_folder, config, model_at_step, n_test_episodes, monitor_cls
        )


def test(config_file):
    config_folder = os.path.dirname(config_file)
    with open(config_file) as json_data_file:
        config = json.load(json_data_file)
    config["policy_params"]["total_timesteps"] = 1

    if config["env"] == "Pacman-v0":
        ghost_distance = config["policy_safety_params"]["ghost_distance"]
    else:
        ghost_distance = None

    (
        model_cls,
        get_sensor_value_ground_truth,
        custom_callback_cls,
        monitor_cls,
        features_extractor_cls,
        observation_net_cls,
    ) = get_env_classes(config["env"], ghost_distance)

    if "ppo" == config["base_policy"]:
        learn_ppo(
            config_folder,
            config,
            model_cls,
            get_sensor_value_ground_truth,
            custom_callback_cls,
            monitor_cls,
            features_extractor_cls,
            observation_net_cls,
        )


def train(config_file):
    config_folder = os.path.dirname(config_file)
    with open(config_file) as json_data_file:
        config = json.load(json_data_file)

    if config["env"] == "Pacman-v0":
        ghost_distance = config["policy_safety_params"]["ghost_distance"]
    else:
        ghost_distance = None

    (
        model_cls,
        get_sensor_value_ground_truth,
        custom_callback_cls,
        monitor_cls,
        features_extractor_cls,
        observation_net_cls,
    ) = get_env_classes(config["env"], ghost_distance)

    if "ppo" == config["base_policy"]:
        learn_ppo(
            config_folder,
            config,
            model_cls,
            get_sensor_value_ground_truth,
            custom_callback_cls,
            monitor_cls,
            features_extractor_cls,
            observation_net_cls,
        )


def pretrain_observation(
    csv_file,
    img_folder,
    observation_net_folder,
    image_dim,
    downsampling_size,
    net_class,
    num_training_examples,
    epochs,
    labels,
    pretrain_w_extra_labels,
    num_test_examples=200,
):
    # Adjust the input size of the observation network considering sampling
    if downsampling_size is not None:
        net_input_size = math.ceil(image_dim / downsampling_size) ** 2
    else:
        net_input_size = image_dim

    net_output_size = len(labels)

    pretrain(
        csv_file=csv_file,
        image_folder=img_folder,
        model_folder=observation_net_folder,
        num_training_examples=num_training_examples,
        net_class=net_class,
        net_input_size=net_input_size,
        net_output_size=net_output_size,
        image_dim=image_dim,
        downsampling_size=downsampling_size,
        epochs=epochs,
        keys=labels,
        pretrain_w_extra_labels=pretrain_w_extra_labels,
        num_test_examples=num_test_examples,
    )
