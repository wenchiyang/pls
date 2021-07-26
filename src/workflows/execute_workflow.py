from workflows.pacman_pg_dpl import main as pg_dpl
from workflows.predict_states import main as predict
from workflows.predict_states import main as ac_dpl
import json
import os

def train_models(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    pg_dpl(folder, config)

def predict_states(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    predict(folder, config)


def train_ac_models(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    ac_dpl(folder, config)

