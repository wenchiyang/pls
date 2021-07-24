from src.workflows.pacman_pg_dpl import main as pg_dpl
from src.workflows.predict_states import main as predict
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

# train_models("/Users/wenchi/PycharmProjects/NeSysourse/experiments/grid3x3_1_ghost/pg")