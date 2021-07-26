from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
from os.path import join, abspath
from os import getcwd
from workflows.execute_workflow import train_models as train_models
import time

if __name__ == "__main__":
    cluster = SSHCluster(
        hosts=[
            "134.58.41.141",
            "134.58.41.142",
               ],
        remote_python="/home/wenchi/.pyenv/shims/python",
        # connect_options={
        #     "config": "/Users/wenchi/PycharmProjects/NeSysourse/src/config.txt"
        # }
    )
    print("hu")
    time.wait(10)
    cluster.close()

    # client = Client(cluster)

    # cluster = LocalCluster(
    #     n_workers=4,
    #     processes=True,
    #     threads_per_worker=1,
    #     dashboard_address=":58626"
    # )
    # client = Client(cluster)
    #
    # exp_folder = join(getcwd(), "experiments")
    # exp_folder = abspath(exp_folder)
    # exps = [
    #     str(join(exp_folder, "grid2x2_1_ghost", "pg")),
    #     str(join(exp_folder, "grid2x2_1_ghost", "pg_shield")),
    #     str(join(exp_folder, "grid2x2_1_ghost", "pg_shield_detect")),
    #
    #     str(join(exp_folder, "grid2x3_1_ghost", "pg")),
    #     str(join(exp_folder, "grid2x3_1_ghost", "pg_shield")),
    #     str(join(exp_folder, "grid2x3_1_ghost", "pg_shield_detect")),
    #
    #     str(join(exp_folder, "grid3x3_1_ghost", "pg")),
    #     str(join(exp_folder, "grid3x3_1_ghost", "pg_shield")),
    #     str(join(exp_folder, "grid3x3_1_ghost", "pg_shield_detect")),
    #
    #     # str(join(exp_folder, "grid5x5_1_ghost", "pg")),
    #     # str(join(exp_folder, "grid5x5_1_ghost", "pg_shield")),
    #     # str(join(exp_folder, "grid5x5_1_ghost", "pg_shield_detect")),
    #     #
    #     # str(join(exp_folder, "grid5x5_3_ghosts", "pg")),
    #     # str(join(exp_folder, "grid5x5_3_ghosts", "pg_shield")),
    #     # str(join(exp_folder, "grid5x5_3_ghosts", "pg_shield_detect")),
    #     #
    #     # str(join(exp_folder, "grid5x5_5_ghosts", "pg")),
    #     # str(join(exp_folder, "grid5x5_5_ghosts", "pg_shield")),
    #     # str(join(exp_folder, "grid5x5_5_ghosts", "pg_shield_detect")),
    # ]
    #
    # with performance_report(filename="dask-report.html"):
    #     ## some dask computation
    #     futures = client.map(train_models, exps)
    #     results = client.gather(futures)
