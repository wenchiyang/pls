import os
from pls.workflows.pre_train import generate_random_images_gf, generate_random_images_sokoban
from pls.workflows.execute_workflow import pretrain_observation_gf, pretrain_observation_sokoban
from dask.distributed import Client

def pre_train_gf():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_folder = os.path.join(dir_path, "../pls/data/gftmp")
    csv_file = os.path.join(img_folder, "labels.csv")

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # generate_random_images_gf(csv_file, img_folder, 10)

    model_folder = os.path.join(dir_path, "../experiments_trials3/goal_finding/7grid5g_gray/data/")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # # pretrain_observation_gf(csv_file, img_folder, model_folder, 100, 300)
    # pretrain_observation(csv_file, img_folder, model_folder, 1000, 300)
    # # pretrain_observation(csv_file, img_folder, model_folder, 10000)


def pre_train_sokoban(n_train):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_folder = os.path.join(dir_path, "../pls/data/sokoban")
    csv_file = os.path.join(img_folder, "labels.csv")

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    generate_random_images_sokoban(csv_file, img_folder, 10100)

    model_folder = os.path.join(dir_path, "../experiments_trials3/sokoban/2box5map_gray/data/")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # pretrain_observation_sokoban(csv_file, img_folder, model_folder, n_train, 300)
    # pretrain_observation_sokoban(csv_file, img_folder, model_folder, n_train, 300)


def main_cluster():
    client = Client("134.58.41.100:8786")

    # with performance_report(filename="dask-report.html"):
    ## some dask computation
    futures = client.map(pre_train_sokoban, [1000, 10000])
    results = client.gather(futures)

if __name__ == "__main__":
    main_cluster()

