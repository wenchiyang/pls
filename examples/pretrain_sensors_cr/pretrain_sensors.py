import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from env_specific_classes.carracing.util import get_ground_truth_of_grass
from pls.algorithms.ppo_shielded import PPO_shielded
from pls.workflows.execute_workflow import pretrain_observation
from env_specific_classes.carracing.env_classes import (
    Carracing_Observation_Net,
    Carracing_Monitor,
)
import json
import gym
import csv

import matplotlib.pyplot as plt
import torch as th


def generate_random_images_cr(
    policy_folder, model_at_step, image_folder, csv_path, sample_frequency, num_imgs=10
):
    # load the agent and its environment
    policy_path = os.path.join(policy_folder, "config.json")
    with open(policy_path) as json_data_file:
        config = json.load(json_data_file)

    env_args = config["eval_env_features"]
    env = gym.make(config["env"], **env_args)

    env = Carracing_Monitor(
        env,
        allow_early_resets=False,
        **config["monitor_features"]
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
    writer.writerow(
        ["image_name", "grass(in_front)", "grass(on_the_left)", "grass(on_the_right)"]
    )


    current_lengths = 0
    n = 0
    while n < num_imgs:
        # use the loaded policy to control the agent
        actions = model.predict(observations, deterministic=deterministic_action)
        observations, rewards, dones, infos = env.step(actions[0])

        # sample an image every sample_frequency frames
        if current_lengths % sample_frequency == 0:
            # save the colored image
            colored_img = env.envs[0].render(mode="state_pixels")
            image_path = os.path.join(image_folder, f"img{n:06}.png")
            plt.imsave(image_path, colored_img)

            # create the ground truth labels of the image
            gray_img = (
                th.tensor(env.envs[0].render(mode="gray"))
                .unsqueeze(dim=0)
                .unsqueeze(dim=1)
            )
            ground_truth_grass = get_ground_truth_of_grass(input=gray_img)

            # write the ground truth labels to the csv file
            row = [f"img{n:06}.png"] + ground_truth_grass.flatten().tolist()
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
        cwd, "..", "train_a_policy/carracing/no_shield/seed1"
    )
    model_at_step = 600000
    image_dim = 48  # The size of each image is 48 x 48 pixels
    num_imgs = 3000
    sample_frequency = 50  # sample every 50 frames
    # location to save the generated images
    img_folder = os.path.join(cwd, "data/")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    # location of the labels of the generated imgaes
    csv_file = os.path.join(img_folder, "labels.csv")

    # generate the images and the corresponding labels
    generate_random_images_cr(
        policy_folder,
        model_at_step,
        img_folder,
        csv_file,
        sample_frequency,
        num_imgs=num_imgs,
    )

    net_class = Carracing_Observation_Net
    # downsample the image using block_reduce of the skimage package. If the value is 1 then there is no downsampling.
    downsampling_size = 1
    # location of the trained observation network
    observation_net_folder = cwd
    num_training_examples = 2500
    num_test_examples = 500
    epochs = 30

    # we use three sensors for the car racing domain
    labels = ["grass(in_front)", "grass(on_the_left)", "grass(on_the_right)"]

    # pretrain the observation net
    pretrain_observation(
        csv_file=csv_file,
        img_folder=img_folder,
        observation_net_folder=observation_net_folder,
        image_dim=image_dim,
        downsampling_size=downsampling_size,
        num_training_examples=num_training_examples,
        epochs=epochs,
        net_class=net_class,
        labels=labels,
        pretrain_w_extra_labels=False,
        num_test_examples=num_test_examples,
    )
