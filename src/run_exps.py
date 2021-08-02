from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
from os.path import join, abspath
from os import getcwd
from workflows.execute_workflow import train_ppo_models


if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=24,
        processes=True,
        threads_per_worker=1,
        dashboard_address=":8787"
    )

    client = Client(cluster)

    exps_folder = abspath(join(getcwd(), "experiments"))
    exps = [
        # "grid2x2_1_ghost",
        # "grid2x3_1_ghost",
        # "grid3x3_1_ghost",
        # "grid5x5_1_ghost",
        # "grid5x5_3_ghosts",
        # "grid5x5_5_ghosts",
            ]
    types = [
        "grid2x2_1_ghost/ppo",
        "grid2x2_1_ghost/ppo_shield",
        "grid2x2_1_ghost/ppo_shield_detect",

        "grid2x3_1_ghost/ppo",
        "grid2x3_1_ghost/ppo_shield",
        "grid2x3_1_ghost/ppo_shield_detect",

        "grid3x3_1_ghost/ppo",
        "grid3x3_1_ghost/ppo_shield",
        "grid3x3_1_ghost/ppo_shield_detect",

        "grid5x5_1_ghost/ppo",
        "grid5x5_1_ghost/ppo_shield",
        "grid5x5_1_ghost/ppo_shield_detect",

        "grid5x5_3_ghosts/ppo/16384",
        "grid5x5_3_ghosts/ppo_shield/16384",
        "grid5x5_3_ghosts/ppo_shield_detect/16384",

        "grid5x5_3_ghosts/ppo/32768",
        "grid5x5_3_ghosts/ppo_shield/32768",
        "grid5x5_3_ghosts/ppo_shield_detect/32768",

        "grid5x5_5_ghosts/ppo/16384",
        "grid5x5_5_ghosts/ppo_shield/16384",
        "grid5x5_5_ghosts/ppo_shield_detect/16384",

        "grid5x5_5_ghosts/ppo/32768",
        "grid5x5_5_ghosts/ppo_shield/32768",
        "grid5x5_5_ghosts/ppo_shield_detect/32768",

    ]

    tasks = []
    # for exp in exps:
    for type in types:
        path = join(exps_folder, type)
        tasks.append(path)

    # with performance_report(filename="dask-report.html"):
    ## some dask computation
    futures = client.map(train_ppo_models, tasks)
    results = client.gather(futures)

    cluster.close()


