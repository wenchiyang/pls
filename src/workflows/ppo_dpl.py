import gym
import pacman_gym
import gym_sokoban

import torch as th
from torch import nn
from os.path import join, abspath
from src.dpl_policies.pacman.dpl_policy import (
    Pacman_Encoder,
    Pacman_Monitor,
    Pacman_DPLActorCriticPolicy,
    Pacman_Callback,
)
from src.dpl_policies.sokoban.dpl_policy import (
    Sokoban_Encoder,
    Sokoban_Monitor,
    Sokoban_DPLActorCriticPolicy,
    Sokoban_Callback
)
from src.dpl_policies.sokoban.sokoban_ppo import Sokoban_DPLPPO
from src.dpl_policies.pacman.pacman_ppo import Pacman_DPLPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback




def setup_env(folder, config, program_path):
    #####   Initialize env   #############
    env_name = config["env_type"]
    env_args = config["env_features"]
    env = gym.make(env_name, **env_args)

    if "Pacman" in env_name:
        image_encoder_cls = Pacman_Encoder
        shielding_settings = {
            "shield": config["model_features"]["params"]["shield"],
            "detect_ghosts": config["model_features"]["params"]["detect_ghosts"],
            "detect_walls": config["model_features"]["params"]["detect_walls"],
            "ghost_layer_num_output": config["model_features"]["params"]["ghost_layer_num_output"],
            "wall_layer_num_output": config["model_features"]["params"]["wall_layer_num_output"]
        }
        env = Pacman_Monitor(
            env,
            allow_early_resets=False,
            program_path=program_path
        )
        custom_callback = None
        custom_callback = Pacman_Callback(custom_callback)
    elif "Sokoban" in env_name:
        image_encoder_cls = Sokoban_Encoder
        shielding_settings = {
            "shield": config["model_features"]["params"]["shield"],
            "detect_boxes": config["model_features"]["params"]["detect_boxes"],
            "detect_corners": config["model_features"]["params"]["detect_corners"],
            "box_layer_num_output": config["model_features"]["params"]["box_layer_num_output"],
            "corner_layer_num_output": config["model_features"]["params"][
                "corner_layer_num_output"
            ],
        }

        env = Sokoban_Monitor(
            env,
            allow_early_resets=False,
            program_path=program_path
        )
        custom_callback = None
        custom_callback = Sokoban_Callback(custom_callback)



    return env, image_encoder_cls, shielding_settings, custom_callback


def main(folder, config):
    """
    Runs policy gradient with deep problog
    """
    #####   Read from config   #############


    #####   Initialize loggers   #############
    new_logger = configure(folder, ["stdout", "tensorboard"])

    #####   Configure network   #############
    net_arch = config["model_features"]["params"]["net_arch_shared"] + [
        dict(
            pi=config["model_features"]["params"]["net_arch_pi"],
            vf=config["model_features"]["params"]["net_arch_vf"],
        )
    ]

    #####   Initialize env   #############
    program_path = abspath(
        join("src", "data", f'{config["model_features"]["params"]["program_type"]}.pl')
    )

    env, image_encoder_cls, shielding_settings, custom_callback = setup_env(
        folder, config, program_path
    )

    grid_size = env.grid_size
    height = env.grid_height
    width = env.grid_weight
    color_channels = env.color_channels
    n_pixels = (height * grid_size) * (width * grid_size) * color_channels
    n_actions = env.action_size

    env_name = config["env_type"]
    if "Pacman" in env_name:
        model_cls = Pacman_DPLPPO
        policy_cls = Pacman_DPLActorCriticPolicy
    elif "Sokoban" in env_name:
        model_cls = Sokoban_DPLPPO
        policy_cls = Sokoban_DPLActorCriticPolicy



    image_encoder = image_encoder_cls(
        n_pixels, n_actions, shielding_settings, program_path
    )

    model = model_cls(
        policy_cls,
        env,
        learning_rate=config["model_features"]["params"]["learning_rate"],
        n_steps=config["model_features"]["params"]["n_steps"],
        # n_steps: The number of steps to run for each environment per update
        batch_size=config["model_features"]["params"]["batch_size"],
        n_epochs=config["model_features"]["params"]["n_epochs"],
        gamma=config["model_features"]["params"]["gamma"],
        clip_range=config["model_features"]["params"]["clip_range"],
        tensorboard_log=folder,
        policy_kwargs={
            "image_encoder": image_encoder,
            "net_arch": net_arch,
            "activation_fn": nn.ReLU,
            "optimizer_class": th.optim.Adam,
        },
        verbose=0,
        seed=config["model_features"]["params"]["seed"],
        _init_setup_model=True,
    )

    model.set_random_seed(config["model_features"]["params"]["seed"])
    model.set_logger(new_logger)


    intermediate_model_path = join(folder, "model_checkpoints")
    checkpoint_callback = CheckpointCallback(save_freq=5e3, save_path=intermediate_model_path)


    model.learn(
        total_timesteps=config["model_features"]["params"]["step_limit"],
        callback=[custom_callback, checkpoint_callback])
    model.save(join(folder, "model"))



# def load_model_and_env(folder, config):
#     program_path = abspath(
#         join("src", "data", f'{config["model_features"]["params"]["program_type"]}.pl')
#     )
#     env, image_encoder_cls, shielding_settings, custom_callback = setup_env(
#         folder, config, program_path
#     )
#     env_name = config["env_type"]
#     if "Pacman" in env_name:
#         model_cls = Pacman_DPLPPO
#     elif "Sokoban" in env_name:
#         model_cls = Sokoban_DPLPPO
#
#     path = os.path.join(folder, "model")
#     model = model_cls.load(path, env)
#
#     return model, env


