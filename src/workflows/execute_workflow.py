from workflows.pg_dpl import main as pg_dpl
from workflows.predict_states import main as predict
from workflows.ppo_dpl import main as ppo_dpl
from workflows.a2c_dpl import main as a2c_dpl
import json
import os


def train(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    learner = config["workflow_name"]
    if "pg" in learner:
        pg_dpl(folder, config)
    elif "ppo" in learner:
        ppo_dpl(folder, config)
    elif "a2c" in learner:
        a2c_dpl(folder, config)


# def predict_states(folder):
#     path = os.path.join(folder, "config.json")
#     with open(path) as json_data_file:
#         config = json.load(json_data_file)
#     predict(folder, config)



