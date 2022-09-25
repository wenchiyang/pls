from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
import os
from pls.workflows.execute_workflow import train, test, evaluate
import itertools


def run_train():
    for exp in exps:
        train(exp)

def run_test():
    for exp in exps:
        test(exp)

def write_to_file(folder):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join("text.txt")
    with open(path, "w") as f:
        f.write(str(folder))
        f.write(str(dir_path))
        f.write(str(exps))

def run_evaluate():
    folder = exps[0]
    # no = os.path.join(folder, "no_shielding")
    # soft1 = os.path.join(folder, "soft_shielding")
    # hard = os.path.join(folder, "hard_shielding")
    # soft2 = os.path.join(folder, "soft_shielding2")

    # model_at_step = 1500000
    # model_at_step = 1000000
    model_at_step = "end"
    mean_reward, n_deaths = evaluate(folder, model_at_step=model_at_step, n_test_episodes=100)
    print(mean_reward, n_deaths)

def main_cluster():
    client = Client("134.58.41.100:8786")

    ## some dask computation
    futures = client.map(train, exps)
    results = client.gather(futures)

def main_cluster_test():
    client = Client("134.58.41.100:8786")

    ## some dask computation
    futures = client.map(test, exps)
    results = client.gather(futures)


if __name__ == "__main__":
    hyper_parameters = {
        "exp_folders": [
            # "experiments4",
            # "experiments_trials3"
            "experiments_safety"
        ],
        "domains": [
            # "goal_finding/7grid5g"
            "goal_finding/small"
            # "sokoban/2box1map",
            # "carracing/onemap"
            # "carracing/sparse_rewards4"
            # "goal_finding/test3"
        ],
        "exps": [
            "sflosscoef",
            "noisy",
            "thres",
            # "test"
            ],
        "seeds":
            [
            "PPO", "coef=0.1", "coef=1", "coef=5",
            "PLSnpisy", "coef=0.01", "coef=0.1", "coef=1", "coef=5", "coef=10",
            "PLSthres",
            ]
    }

    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # dir_path = os.path.abspath(os.path.join(dir_path, ".."))

    lengths = list(map(len, list(hyper_parameters.values())))
    lists_of_indices = list(map(lambda l: list(range(l)), lengths))
    combinations = list(itertools.product(*lists_of_indices))

    exps = []
    for combination in combinations:
        hyper = dict.fromkeys(hyper_parameters.keys())
        hyper["exp_folders"] = hyper_parameters["exp_folders"][combination[0]]
        hyper["domains"] = hyper_parameters["domains"][combination[1]]
        hyper["exps"] = hyper_parameters["exps"][combination[2]]
        hyper["seeds"] = hyper_parameters["seeds"][combination[3]]
        folder = os.path.join(
                              hyper["exp_folders"],
                              hyper["domains"],
                              hyper["exps"],
                              hyper["seeds"],
                              )
        exps.append(folder)
    main_cluster()
    # main_cluster_test()
    # run_train()
    # run_test()
    # run_evaluate()