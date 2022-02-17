from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
import os
from src.workflows.execute_workflow import train
import itertools


hyper_parameters= {
    "domains": [
        "pacman_5x5_new", "pacman_6x6_new", "pacman_6x6_2_new",
        "pacman_smallGrid_new", "pacman_smallGrid2_new",
        "sokoban_5x5_new", "sokoban_6x6_new", "sokoban_7x7_new"
    ],
    "workflow_names": ["ppo"],
    "shield_types": ["no_shielding", "hard_shielding", "soft_shielding"],
}
cwd = os.getcwd()

lengths = list(map(len, list(hyper_parameters.values())))
lists_of_indices = list(map(lambda l: list(range(l)), lengths))
combinations = list(itertools.product(*lists_of_indices))

exps = []
for combination in combinations:
    hyper = dict.fromkeys(hyper_parameters.keys())
    hyper["domain"] = hyper_parameters["domains"][combination[0]]
    hyper["workflow_name"] = hyper_parameters["workflow_names"][combination[1]]
    hyper["shield_type"] =  hyper_parameters["shield_types"][combination[2]]
    folder = os.path.join(cwd, "experiments_trials",
                          hyper["domain"],
                          hyper["workflow_name"],
                          hyper["shield_type"],
                          )

    exps.append(folder)

def run_train():
    for exp in exps:
        if os.path.isfile(exp):
            train(exp)

# def run_evaluate():
#     for exp in exps:
#         evaluate(exp)

def main_cluster():
    cluster = LocalCluster(
        n_workers=32,
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
    main_cluster()
    # run_train()
    # run_evaluate()