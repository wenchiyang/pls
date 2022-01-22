from workflows.pg_dpl import main as pg_dpl
from workflows.predict_states import main as predict
from workflows.ppo_dpl import main as ppo_dpl
from workflows.ppo_dpl import load_model_and_env as ppo_load_model_and_env
# from workflows.a2c_dpl import main as a2c_dpl
from stable_baselines3.common.evaluation import evaluate_policy
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
    # elif "a2c" in learner:
    #     a2c_dpl(folder, config)


def evaluate(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)
    learner = config["workflow_name"]
    if "ppo" in learner:
        model, env = ppo_load_model_and_env(folder, config)

    ep_rewards, ep_lengths = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=10,
        deterministic=True,
        return_episode_rewards=True
        # If True, a list of rewards and episode lengths per episode will be returned instead of the mean.
    )
    # TODO: Store ep_rewards, ep_lengths somewhere


# def predict_states(folder):
#     path = os.path.join(folder, "config.json")
#     with open(path) as json_data_file:
#         config = json.load(json_data_file)
#     predict(folder, config)



