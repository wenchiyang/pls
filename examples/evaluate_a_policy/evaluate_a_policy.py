from pls.workflows.execute_workflow import evaluate
import os

if __name__ == "__main__":
    # current working directory
    cwd = os.path.join(os.path.dirname(__file__))
    config_file = os.path.join(cwd, "../train_a_policy/carracing/no_shield/seed1/config.json")


    mean_reward, std_reward = evaluate(config_file, model_at_step="end", n_test_episodes=10)
    print(f"{mean_reward=}, {std_reward=}")