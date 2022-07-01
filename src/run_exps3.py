import os
from workflows.execute_workflow import pretrain_observation, generate_random_images



if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_folder = os.path.join(dir_path, "data/gf")
    csv_file = os.path.join(img_folder, "labels.csv")

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # generate_random_images(csv_file, img_folder, 10100)

    model_folder = os.path.join(dir_path, "../experiments_trials3/goal_finding/7grid5g_gray/data/")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # pretrain_observation(csv_file, img_folder, model_folder, 10, 10)
    # pretrain_observation(csv_file, img_folder, model_folder, 100, 300)
    # pretrain_observation(csv_file, img_folder, model_folder, 1000, 300)
    pretrain_observation(csv_file, img_folder, model_folder, 10000, 300)
