from pls.workflows.ppo_dpl import main as ppo_dpl
from pls.workflows.pre_train import main
from pls.workflows.evaluate import evaluate as evaluate_policy
import json
import os
import math


def pretrain_observation_cr(csv_file, img_folder, model_folder, image_dim, downsampling_size, net_class, n_train, epochs):
    net_input_size = math.ceil(image_dim / downsampling_size) ** 2
    keys = ["grass(in_front)", "grass(on_the_left)", "grass(on_the_right)"]

    main(csv_file=csv_file, root_dir=img_folder, model_folder=model_folder, n_train=n_train,
         net_class=net_class, net_input_size=net_input_size, net_output_size=3,
         image_dim=image_dim,
         downsampling_size=downsampling_size, epochs=epochs, keys=keys, use_agent_coord=False)

def pretrain_observation_stars(csv_file, img_folder, model_folder, image_dim, downsampling_size, net_class, n_train, epochs):
    if downsampling_size is not None:
        net_input_size = math.ceil(image_dim / downsampling_size) ** 2
    else:
        net_input_size = image_dim
    keys = ["ghost(up)", "ghost(down)", "ghost(left)", "ghost(right)"]

    main(csv_file=csv_file, root_dir=img_folder, model_folder=model_folder, n_train=n_train,
         net_class=net_class, net_input_size=net_input_size, net_output_size=4,
         image_dim=image_dim, downsampling_size=downsampling_size, epochs=epochs, keys=keys)

def test(path):
    folder = os.path.dirname(path)
    with open(path) as json_data_file:
        config = json.load(json_data_file)
    config["model_features"]["params"]["step_limit"] = 1
    learner = config["workflow_name"]
    if "ppo" in learner:
        ppo_dpl(folder, config)

def train(path):
    folder = os.path.dirname(path)
    with open(path) as json_data_file:
        config = json.load(json_data_file)
    learner = config["workflow_name"]
    if "ppo" in learner:
        ppo_dpl(folder, config)


def evaluate(folder, model_at_step, n_test_episodes):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    learner = config["workflow_name"]
    if "ppo" in learner:
        return evaluate_policy(folder, model_at_step, n_test_episodes)


# def pretrain_observation_sokoban(csv_file, img_folder, model_folder, image_dim, downsampling_size, net_class, n_train, epochs):
#     net_input_size = math.ceil(image_dim / downsampling_size) ** 2
#     keys = ["box(up)", "box(down)", "box(left)", "box(right)", "corner(up)", "corner(down)", "corner(left)", "corner(right)"]
#
#     main(csv_file=csv_file, root_dir=img_folder, model_folder=model_folder, n_train=n_train,
#          net_class=net_class, net_input_size=net_input_size, net_output_size=8,
#          image_dim=image_dim,
#          downsampling_size=downsampling_size, epochs=epochs, keys=keys)

