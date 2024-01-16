import os
import gym

from pls.algorithms.ppo_shielded import PPO_shielded
from stable_baselines3.common.evaluation import evaluate_policy


def main(config_folder, config, model_at_step, n_test_episodes, monitor_cls):
    """
    Evaluate the given policy by executing it in the given environment.
    Calls stable_baselines3's default evaluate_policy function.

    :param config_folder: location of the config file
    :param config: a dict containing the configuration
    :param model_at_step: load a snapshot of the policy trained after model_at_step steps.
                          If given "end", load the last saved model
    :param n_test_episodes: number of episode to run
    :param monitor_cls:
    :return: mean and standard deviation of reward
    """

    # initialize the environment for evaluation
    env = gym.make(config["env"], **config["eval_env_features"])

    env = monitor_cls(
        env,
        allow_early_resets=False,
    )

    # load the trained policy
    if model_at_step == "end":
        path = os.path.join(config_folder, "model.zip")
    else:
        path = os.path.join(
            config_folder, "model_checkpoints", f"rl_model_{model_at_step}_steps.zip"
        )

    model = PPO_shielded.load(path, env)

    # calls stable_baselines3's default evaluate_policy function
    mean_reward, std_reward = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=n_test_episodes,
        deterministic=False,
        return_episode_rewards=False,
        render=True,
    )

    return mean_reward, std_reward
