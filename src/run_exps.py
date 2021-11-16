from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
from os.path import join, abspath
from os import getcwd
from workflows.execute_workflow import train

def test():
    exps_folder = abspath(join(getcwd(), "experiments"))
    types = [
        # "grid2x2_1_ghost/pg",
        # "sokoban/ppo",
        "sokoban/pg",
    ]
    # for exp in exps:
    for type in types:
        path = join(exps_folder, type)
        train(path)

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
    test()