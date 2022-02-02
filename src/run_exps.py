from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
import os
from workflows.execute_workflow import train, evaluate
import itertools


hyper_parameters= {
    "workflow_names": ["ppo"],
    "shield_types": ["no_shielding", "hard_shielding", "soft_shielding"],
    "batch_sizes": [32, 64, 128, 256, 512],
    "n_epochs": [10, 20, 40, 60],
    "learning_rates": [1e-4, 1e-3, 1e-2],
    "clip_ranges": [0.1, 0.2, 0.3]
}
cwd = os.getcwd()

lengths = list(map(len, list(hyper_parameters.values())))
lists_of_indices = list(map(lambda l: list(range(l)), lengths))
combinations = list(itertools.product(*lists_of_indices))

exps = []
for combination in combinations:
    hyper = dict.fromkeys(hyper_parameters.keys())
    hyper["workflow_name"] = hyper_parameters["workflow_names"][combination[0]]
    hyper["shield_type"] =  hyper_parameters["shield_types"][combination[1]]
    hyper["batch_size"] = hyper_parameters["batch_sizes"][combination[2]]
    hyper["n_epoch"] = hyper_parameters["n_epochs"][combination[3]]
    hyper["learning_rate"] = hyper_parameters["learning_rates"][combination[4]]
    hyper["clip_range"] = hyper_parameters["clip_ranges"][combination[5]]
    folder = os.path.join(cwd, "experiments_trials/sokoban",
                          hyper["workflow_name"],
                          hyper["shield_type"],
                          f'batch_size_{hyper["batch_size"]}',
                          f'n_epoch_{hyper["n_epoch"]}',
                          f'learning_rate_{hyper["learning_rate"]}',
                          f'clip_range_{hyper["clip_range"]}'
                          )

    exps.append(folder)

def run_train():
    folder = os.path.join(
        cwd,
        "experiments_trials",
        "sokoban_6x6",
        # "dqn",
        # "ppo",
        "a2c",
        "no_shielding",
        # "batch_size_128",
        # "n_epoch_40",
        # "learning_rate_0.001",
        # "clip_range_0.2",
        # "max_steps_20"

    )
    train(folder)
    # for exp in exps:
    #     train(exp)

def run_evaluate():
    folder = os.path.join(
        cwd,
        "experiments_trials",
        "sokoban_6x6",
        "ppo",
        "hard_shielding",
        f'batch_size_128',
        f'n_epoch_40',
        f'learning_rate_0.001',
        f'clip_range_0.2',
        # 'max_steps_200'
    )
    evaluate(folder)
    # for exp in exps:
    #     evaluate(exp)

def main_cluster():
    cluster = LocalCluster(
        n_workers=24,
        processes=True,
        threads_per_worker=1,
        dashboard_address=":8787"
    )

    client = Client(cluster)

    # with performance_report(filename="dask-report.html"):
    ## some dask computation
    futures = client.map(train, exps)
    results = client.gather(futures)

    cluster.close()


if __name__ == "__main__":
    # main_cluster()
    run_train()
    # run_evaluate()