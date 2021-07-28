from workflows.pacman_pg_dpl import main as pg_dpl
from workflows.predict_states import main as predict
from workflows.pacman_ppo_dpl import main as ppo_dpl
import json
import os

def train_pg_models(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    pg_dpl(folder, config)

def predict_states(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    predict(folder, config)


def train_ppo_models(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    ppo_dpl(folder, config)

