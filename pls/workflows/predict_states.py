import os
import gym
from itertools import count
from torch.distributions import Categorical
from ppo_dpl import setup_env
# from pls.dpl_policies.carracing.carracing_ppo import Carracing_DPLPPO
from pls.dpl_policies.sokoban.sokoban_ppo import Sokoban_DPLPPO
import torch as th

def load(folder, config, step=None):
    """
    todo
    """
    env, image_encoder_cls, shielding_settings, custom_callback = setup_env(folder, config)

    grid_size = env.grid_size
    height = env.grid_height
    width = env.grid_weight
    color_channels = env.color_channels
    n_pixels = (height * grid_size) * (width * grid_size) * color_channels
    n_actions = env.action_size

    program_path = os.path.join(folder, "../../../data", f'{config["model_features"]["params"]["program_type"]}.pl')
    debug_program_path = os.path.join(folder, "../../../data",
                                      f'{config["model_features"]["params"]["debug_program_type"]}.pl')

    #####   Load network   #############
    model_path = os.path.join(folder, "model_checkpoints", f"rl_model_{step}_steps")
    # model_path = os.path.abspath(model_path)
    image_encoder= image_encoder_cls(
            n_pixels, n_actions, shielding_settings, program_path, debug_program_path, folder
        )
    model = Sokoban_DPLPPO.load(model_path, image_encoder=image_encoder)
    # model.load_state_dict(th.load(model_path))


    return env, model


def main(folder, config, step):
    env, model = load(folder, config, step)
    render = config["env_features"]["render"]

    state = env.reset()
    if render:
        env.render()
    for total_steps in count(1):
        tinygrid = env.render('tiny_rgb_array')
        action = model.policy._predict(th.tensor(state).unsqueeze(0), tinygrid=th.tensor(tinygrid).unsqueeze(0))

        state, _, done , _ = env.step(int(action))
        if done:
            state = env.reset()
        if render:
            env.render()

import json
dir_path = os.path.dirname(os.path.realpath(__file__))
# folder = os.path.join(dir_path, "../../experiments_trials3/carracing/onemap/no_shielding/seed1")
folder = os.path.join(dir_path, "../../experiments_trials3/sokoban/2box5map_gray/PPO/seed1")
path = os.path.join(folder, "config.json")
with open(path) as json_data_file:
    config = json.load(json_data_file)
main(folder, config, step=500000)