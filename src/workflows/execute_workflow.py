from workflows.ppo_dpl import main as ppo_dpl
from workflows.ppo_dpl import load_model_and_env as ppo_load_model_and_env
import json
import os
from stable_baselines3.common.utils import obs_as_tensor

def test(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)
        print(config["arg"])

def train(folder):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)

    learner = config["workflow_name"]
    if "ppo" in learner:
        ppo_dpl(folder, config)


def evaluate(folder, model_at_step, n_test_episodes):
    path = os.path.join(folder, "config.json")
    with open(path) as json_data_file:
        config = json.load(json_data_file)
    learner = config["workflow_name"]
    if "ppo" in learner:
        model, env = ppo_load_model_and_env(folder, config, model_at_step)

    mean_reward, n_deaths  = my_evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=n_test_episodes,
        deterministic=True,
        return_episode_rewards=False,
        # If True, a list of rewards and episode lengths per episode will be returned instead of the mean.
        render=True
    )
    return mean_reward, n_deaths


# def predict_states(folder):
#     path = os.path.join(folder, "config.json")
#     with open(path) as json_data_file:
#         config = json.load(json_data_file)
#     predict(folder, config)
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

def my_evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_last_rs = []
    episode_lengths = []
    episode_safeties = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_last_rs = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_abs_safeties = np.ones(n_envs)
    observations = env.reset()
    n_deaths = 0
    states = None
    while (episode_counts < episode_count_targets).any():
        if render:
            for e in env.envs:
                e.env.render()
        actions, states = model.predict(observations, state=states, deterministic=deterministic)
        obs_tensor = obs_as_tensor(observations, model.device)
        # abs_safeties = model.policy.evaluate_safety_shielded(obs_tensor)
        observations, rewards, dones, infos = env.step(actions)
        if render:
            for e in env.envs:
                e.env.render()
        current_rewards += rewards
        # s = np.array(abs_safeties[0])
        # current_abs_safeties = s if s < current_abs_safeties else current_abs_safeties
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]


                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            episode_last_rs.append(info["episode"]["last_r"])
                            if info["episode"]["violate_constraint"]:
                                n_deaths += 1
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_last_rs.append(current_last_rs[i])
                        episode_lengths.append(current_lengths[i])

                        episode_counts[i] += 1

                    ## TODO
                    episode_safeties.append(current_abs_safeties[i])

                    current_rewards[i] = 0
                    current_last_rs[i] = 0
                    current_lengths[i] = 0
                    current_abs_safeties[i] = 1
                    if states is not None:
                        states[i] *= 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    # n_deaths = episode_last_rs.count(-11)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_safeties

    return mean_reward, n_deaths
