from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
from os.path import join, abspath
from os import getcwd
from workflows.execute_workflow import train_ppo_models
import time

if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=4,
        processes=True,
        threads_per_worker=1,
        dashboard_address=":8787"
    )

    client = Client(cluster)

    exps_folder = abspath(join(getcwd(), "experiments"))
    exps = ["grid2x3_1_ghost", "grid3x3_1_ghost"]
    types = ["ppo", "ppo_shield", "ppo_shield_detect"]

    tasks = []
    for exp in exps:
        for type in types:
            path = join(exps_folder, exp, type)
            tasks.append(path)

    # with performance_report(filename="dask-report.html"):
    ## some dask computation
    futures = client.map(train_ppo_models, tasks)
    results = client.gather(futures)

    cluster.close()


