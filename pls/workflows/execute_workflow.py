from pls.workflows.ppo_dpl import main as ppo_dpl
from pls.workflows.pre_train import main as pre_train
from pls.workflows.pre_train import generate_random_images as generate
from pls.workflows.evaluate import evaluate as evaluate_policy
import json
import os
from pls.observation_nets.observation_nets import Observation_net
import math


def generate_random_images(csv_file, folder, n_images=1000):
    generate(csv_file, folder, n_images)

def pretrain_observation(csv_file, img_folder, model_folder, n_train, epochs):
    downsampling_size = 7
    net_input_size = math.ceil(240 / downsampling_size) ** 2

    pre_train(csv_file=csv_file, root_dir=img_folder, model_folder=model_folder, n_train=n_train,
              net_class=Observation_net, net_input_size=net_input_size, net_output_size=4,
              downsampling_size=downsampling_size, epochs=epochs)

def test(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)
        print(config["arg"])

def train(folder):
    path = os.path.join(folder, "config.json")
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
        evaluate_policy(folder, model_at_step, n_test_episodes)


# def predict_states(folder):
#     path = os.path.join(folder, "config.json")
#     with open(path) as json_data_file:
#         config = json.load(json_data_file)
#     predict(folder, config)

