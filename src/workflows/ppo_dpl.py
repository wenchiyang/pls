import gym
import pacman_gym
import gym_sokoban

import torch as th
from torch import nn
from os.path import join, abspath
from dpl_policies.goal_finding.dpl_policy import (
    GoalFinding_Encoder,
    GoalFinding_Monitor,
    GoalFinding_DPLActorCriticPolicy,
    GoalFinding_Callback,
)
from dpl_policies.pacman.dpl_policy import (
    Pacman_Encoder,
    Pacman_Monitor,
    Pacman_DPLActorCriticPolicy,
    Pacman_Callback,
)
from dpl_policies.sokoban.dpl_policy import (
    Sokoban_Encoder,
    Sokoban_Monitor,
    Sokoban_DPLActorCriticPolicy,
    Sokoban_Callback
)
from dpl_policies.sokoban.sokoban_ppo import Sokoban_DPLPPO
from dpl_policies.goal_finding.goal_finding_ppo import GoalFinding_DPLPPO
from dpl_policies.pacman.pacman_ppo import Pacman_DPLPPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback




def setup_env(folder, config, eval=False):
    #####   Initialize env   #############
    env_name = config["env_type"]
    if eval:
        env_args = config["eval_env_features"]
    else:
        env_args = config["env_features"]

    env = gym.make(env_name, **env_args)
    # env = gym.make(env_name)

    if "GoalFinding" in env_name:
        image_encoder_cls = GoalFinding_Encoder
        shielding_settings = {
            "n_ghost_locs": config["model_features"]["params"]["n_ghost_locs"],
            "sensor_noise": config["model_features"]["params"]["sensor_noise"]
        }
        env = GoalFinding_Monitor(
            env,
            allow_early_resets=False
        )
        custom_callback = None
        custom_callback = GoalFinding_Callback(custom_callback)
    elif "Pacman" in env_name:
        image_encoder_cls = Pacman_Encoder
        shielding_settings = {
            "n_ghost_locs": config["model_features"]["params"]["n_ghost_locs"]
        }
        env = Pacman_Monitor(
            env,
            allow_early_resets=False
        )
        custom_callback = None
        custom_callback = Pacman_Callback(custom_callback)
    elif "Sokoban" in env_name or "Boxoban" in env_name:
        image_encoder_cls = Sokoban_Encoder
        shielding_settings = {
            "n_box_locs": config["model_features"]["params"]["n_box_locs"],
            "n_corner_locs": config["model_features"]["params"]["n_corner_locs"],
        }

        env = Sokoban_Monitor(
            env,
            allow_early_resets=False
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
    debug_program_path = abspath(
        join("src", "data", f'{config["model_features"]["params"]["debug_program_type"]}.pl')
    )

    env, image_encoder_cls, shielding_settings, custom_callback = setup_env(
        folder, config
    )

    grid_size = env.grid_size
    height = env.grid_height
    width = env.grid_weight
    color_channels = env.color_channels
    n_pixels = (height * grid_size) * (width * grid_size) * color_channels
    n_actions = env.action_size

    env_name = config["env_type"]
    if "GoalFinding" in env_name:
        model_cls = GoalFinding_DPLPPO
        policy_cls = GoalFinding_DPLActorCriticPolicy
    elif "Pacman" in env_name:
        model_cls = Pacman_DPLPPO
        policy_cls = Pacman_DPLActorCriticPolicy
    elif "Sokoban" in env_name or "Boxoban" in env_name:
        model_cls = Sokoban_DPLPPO
        policy_cls = Sokoban_DPLActorCriticPolicy



    image_encoder = image_encoder_cls(
        n_pixels, n_actions, shielding_settings, program_path, debug_program_path, folder
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
            "alpha": config["model_features"]["params"]["alpha"],
            "differentiable_shield": config["model_features"]["params"]["differentiable_shield"],
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
    checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path=intermediate_model_path)


    model.learn(
        total_timesteps=config["model_features"]["params"]["step_limit"],
        callback=[custom_callback, checkpoint_callback])
    model.save(join(folder, "model"))



def load_model_and_env(folder, config, model_at_step, eval=True):
    program_path = abspath(
        join("src", "data", f'{config["model_features"]["params"]["program_type"]}.pl')
    )
    env, image_encoder_cls, shielding_settings, custom_callback = setup_env(
        folder, config, program_path, eval=eval
    )
    env_name = config["env_type"]
    if "GoalFinding" in env_name:
        model_cls = GoalFinding_DPLPPO
    elif "Sokoban" in env_name:
        model_cls = Sokoban_DPLPPO

    path = join(folder, "model_checkpoints", f"rl_model_{model_at_step}_steps.zip")
    model = model_cls.load(path, env)
    if eval:
        model.set_random_seed(config["eval_env_features"]["seed"])

    return model, env


