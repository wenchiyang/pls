import os

from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from env_specific_classes.pacman.util import get_ground_wall
from pls.algorithms.ppo_shielded import PPO_shielded
from pls.workflows.execute_workflow import pretrain_observation
from env_specific_classes.pacman.env_classes import (
    Pacman_Observation_Net,
    Pacman_Monitor,
)
import json
import gym
import csv
import torch as th

import matplotlib.pyplot as plt
from stable_baselines3.common.utils import obs_as_tensor
from env_specific_classes.pacman.util import get_agent_coord
from pacman_gym import sample_layout

def generate_random_images_pacman2(
    policy_folder,
    model_at_step,
    image_folder,
    csv_path,
    sample_frequency,
    labels,
    ghost_distance,
    num_imgs=10
):
    WALL_COLOR = 0.25
    GHOST_COLOR = 0.5
    PACMAN_COLOR = 0.75
    FOOD_COLOR = 1

    config = {
        "env": "Pacman-v0",
        "eval_env_features": {
            "seed": 567,
            "render_mode": "dict",
            "move_ghosts": True,
            "stochasticity": 0.0,
            "render_or_not": False
        },
        "policy_safety_params": {
            "num_sensors": 4,
            "num_actions": 5,
            "ghost_distance": 2,
            "differentiable": False,
            "shield_program": "../train_a_policy/data/pacman_ghosts.pl"
        },
        "shield_params": None,
        "observation_params": {
            "observation_type": "ground truth"
        }
    }

    env_args = config["eval_env_features"]
    env = gym.make(config["env"], **env_args)

    observations = env.reset()

    # open and write the labels of the training images to a file
    f_csv = open(csv_path, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name"] + labels)


    env.env.gameDisplay = env.env.display
    env.env.rules.quiet = False

    num_ghosts = 30
    num_food = 30
    for n in range(num_imgs):
        layout = sample_layout(
            env.layout.width,
            env.layout.height,
            num_ghosts,
            num_food,
            env.env.non_wall_positions,
            env.env.wall_positions,
            env.env.all_edges,
            check_valid=False
        )
        env.env.game = env.rules.newGame(
            layout,
            env.env.pacman,
            env.env.ghosts,
            env.env.gameDisplay,
            env.env.beQuiet,
            env.env.catchExceptions,
            env.env.symX,
            env.env.symY,
            env.env.background_filename
        )
        env.game.start_game()
        env.env.render()


        img = env.game.compose_img("human")
        path = os.path.join(image_folder, f"img{n:06}.jpeg")
        plt.imsave(path, img)

        obs = obs_as_tensor(env.render(mode="dict"), "cpu")
        obs = {key: item.unsqueeze(dim=0) for key, item in obs.items()}
        ground_truth_ghost = get_ground_wall(
            ghost_distance, PACMAN_COLOR, GHOST_COLOR, input=obs
        )
        agent_r, agent_c = get_agent_coord(obs["tinygrid"], PACMAN_COLOR)
        row = [f"img{n:06}.jpeg"] + ground_truth_ghost.flatten().tolist() + [agent_r, agent_c]
        writer.writerow(row)
        f_csv.flush()
        if (n+1) % 10 == 0:
            print(f'Produce: {n+1}/{num_imgs} [({float(n+1)/num_imgs*100:.0f}%)]')

def generate_random_images_pacman(
    policy_folder,
    model_at_step,
    image_folder,
    csv_path,
    sample_frequency,
    labels,
    ghost_distance,
    num_imgs=10,
):
    WALL_COLOR = 0.25
    GHOST_COLOR = 0.5
    PACMAN_COLOR = 0.75
    FOOD_COLOR = 1
    # load the agent and its environment
    policy_path = os.path.join(policy_folder, "config.json")
    with open(policy_path) as json_data_file:
        config = json.load(json_data_file)

    env_args = config["eval_env_features"]
    env = gym.make(config["env"], **env_args)

    env = Pacman_Monitor(
        env,
        allow_early_resets=False,
    )

    if model_at_step == "end":
        path = os.path.join(policy_folder, "model.zip")
    else:
        path = os.path.join(
            policy_folder, "model_checkpoints", f"rl_model_{model_at_step}_steps.zip"
        )

    model = PPO_shielded.load(path, env)

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])
    observations = env.reset()

    # use deterministic actions from the given stochastic policy
    deterministic_action = False

    # open and write the labels of the training images to a file
    f_csv = open(csv_path, "w")
    writer = csv.writer(f_csv)
    writer.writerow(["image_name"] + labels)

    # for visualization
    current_lengths = 0
    n = 0
    while n < num_imgs:
        # use the loaded policy to control the agent
        actions = model.predict(observations, deterministic=deterministic_action)
        observations, rewards, dones, infos = env.step(actions[0])

        # sample an image every sample_frequency frames
        if current_lengths % sample_frequency == 0:
            # save the colored image
            img = env.envs[0].render(mode="human")
            image_path = os.path.join(image_folder, f"img{n:06}.png")
            plt.imsave(image_path, img)

            # create the ground truth labels of the image
            obs = obs_as_tensor(env.envs[0].render(mode="dict"), "cpu")
            obs = {key: item.unsqueeze(dim=0) for key, item in obs.items()}
            ground_truth_ghost = get_ground_wall(
                ghost_distance, PACMAN_COLOR, GHOST_COLOR, input=obs
            )

            # extra label for the xy coordinate of the agent
            agent_r, agent_c = get_agent_coord(obs["tinygrid"], PACMAN_COLOR)

            # write the ground truth labels to the csv file
            row = [f"img{n:06}.png"] + ground_truth_ghost.flatten().tolist() + [agent_r, agent_c]
            writer.writerow(row)
            f_csv.flush()

            # move on to the next image
            n += 1

        current_lengths += 1


if __name__ == "__main__":
    # current working folder
    cwd = os.path.dirname(os.path.realpath(__file__))
    # location of a pretrained agent (trained for 600k learning steps) used to
    # generate the images for pretraining the observation network
    policy_folder = os.path.join(
        cwd, "..", "train_a_policy/pacman/no_shield/seed1"
    )
    model_at_step = 600000
    image_dim = 482  # The size of each image is 48 x 48 pixels
    num_imgs = 5500
    sample_frequency = 50
    ghost_distance = 2
    # we use 4 sensors for the car racing domain
    labels = ["ghost(up)", "ghost(down)", "ghost(left)", "ghost(right)", "x", "y"]
    # location to save the generated images
    image_folder = os.path.join(cwd, "data/")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    # location of the labels of the generated imgaes
    csv_file = os.path.join(image_folder, "labels.csv")

    # generate the images and the corresponding labels
    generate_random_images_pacman2(
        policy_folder,
        model_at_step,
        image_folder,
        csv_file,
        sample_frequency,
        labels,
        ghost_distance,
        num_imgs=num_imgs,
    )

    net_class = Pacman_Observation_Net
    # downsample the image using block_reduce of the skimage package. If the value is 1 then there is no downsampling.
    downsampling_size = 8
    # location of the trained observation network
    observation_net_folder = cwd
    num_training_examples = 5000
    num_test_examples = 500
    epochs = 200

    # We pretrain the pacman observation function with agent coordinates as extra labels.
    # The coordinates are not used in the RL phase later
    pretrain_w_extra_labels = True

    # pretrain the observation net
    pretrain_observation(
        csv_file=csv_file,
        img_folder=image_folder,
        observation_net_folder=observation_net_folder,
        image_dim=image_dim,
        downsampling_size=downsampling_size,
        num_training_examples=num_training_examples,
        epochs=epochs,
        net_class=net_class,
        labels=labels,
        pretrain_w_extra_labels=pretrain_w_extra_labels,
        num_test_examples=num_test_examples,
    )
