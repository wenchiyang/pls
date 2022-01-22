from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
from os.path import join, abspath
from os import getcwd
from workflows.execute_workflow import train
from workflows.execute_workflow import evaluate as evaluate

def run_train():
    exps_folder = abspath(join(getcwd(), "experiments"))
    types = [
        "sokoban/ppo",
    ]
    # for exp in exps:
    for type in types:
        path = join(exps_folder, type)
        train(path)

def run_evaluate():
    exps_folder = abspath(join(getcwd(), "experiments"))
    types = [
        "sokoban/ppo"
    ]
    # for exp in exps:
    for type in types:
        path = join(exps_folder, type)
        evaluate(path)

def main_cluster():
    cluster = LocalCluster(
        n_workers=24,
        processes=True,
        threads_per_worker=1,
        dashboard_address=":8787"
    )

    client = Client(cluster)

    exps_folder = abspath(join(getcwd(), "experiments"))

    types = [
        "grid2x2_1_ghost/pg",
    ]

    tasks = []
    # for exp in exps:
    for type in types:
        path = join(exps_folder, type)
        tasks.append(path)

    # with performance_report(filename="dask-report.html"):
    ## some dask computation
    futures = client.map(train, tasks)
    results = client.gather(futures)

    cluster.close()


if __name__ == "__main__":
    # main_cluster()
    # run_train()
    run_evaluate()