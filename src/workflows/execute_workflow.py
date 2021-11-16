from workflows.pacman_pg_dpl import main as pg_dpl
from workflows.predict_states import main as predict
from workflows.pacman_ppo_dpl import main as ppo_dpl
from workflows.pacman_a2c_dpl import main as a2c_dpl
import json
import os


def train(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)
    learner = config["workflow_name"]
    if learner == "pg":
        pg_dpl(folder, config)
    elif learner == "ppo":
        ppo_dpl(folder, config)
    elif learner == "a2c":
        a2c_dpl(folder, config)


# def predict_states(folder):
#     path = os.path.join(folder, "config.json")
#     with open(path) as json_data_file:
#         config = json.load(json_data_file)
#     predict(folder, config)



