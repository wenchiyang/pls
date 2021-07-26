from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
from os.path import join, abspath
from os import getcwd
from workflows.execute_workflow import train_models as train_models
import time

if __name__ == "__main__":
    # cluster = SSHCluster(
    #     hosts=[
    #         "134.58.41.141",
    #         "134.58.41.142",
    #            ],
    #     remote_python="/home/wenchi/.pyenv/shims/python",
    #     # connect_options={
    #     #     "config": "/Users/wenchi/PycharmProjects/NeSysourse/src/config.txt"
    #     # }
    #     dashboard_address=":8787"
    # )

    cluster = LocalCluster(
        n_workers=8,
        processes=True,
        threads_per_worker=1,
        dashboard_address=":8787"
    )

    client = Client(cluster)

    exps_folder = abspath(join(getcwd(), "experiments"))
    exps = ["smallGrid", "smallGrid2"]
    types = ["pg", "pg_shield", "pg_shield_detect"]

    tasks = []
    for exp in exps:
        for type in types:
            path = join(exps_folder, exp, type)
            tasks.append(path)

    with performance_report(filename="dask-report.html"):
        ## some dask computation
        futures = client.map(train_models, tasks)
        results = client.gather(futures)

    cluster.close()