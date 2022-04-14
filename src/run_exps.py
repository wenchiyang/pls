from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
import os
from workflows.execute_workflow import train, evaluate, test
import itertools

hyper_parameters= {
    "exp_folders": ["experiments_trials3"],
    "domains": [
        "sokoban/2box10map",
        # "goal_finding/smallGrid100map"
        # "test"
    ],
    "exps":
        # ["test1", "test2"],
        # ["no_shielding"],
        [
        # "no_shielding", "hard_shielding",
        #  "alpha_0.1", "alpha_0.3",
        #  "alpha_0.5",
        #  "alpha_0.7", "alpha_0.9",
         "vsrl"],
    "seeds":
        # ["seed1", "seed2", "seed3", "seed4", "seed5"]
        ["seed1"]
}

cwd = os.getcwd()

lengths = list(map(len, list(hyper_parameters.values())))
lists_of_indices = list(map(lambda l: list(range(l)), lengths))
combinations = list(itertools.product(*lists_of_indices))

exps = []
for combination in combinations:
    hyper = dict.fromkeys(hyper_parameters.keys())
    hyper["exp_folders"] = hyper_parameters["exp_folders"][combination[0]]
    hyper["domains"] = hyper_parameters["domains"][combination[1]]
    hyper["exps"] = hyper_parameters["exps"][combination[2]]
    hyper["seeds"] = hyper_parameters["seeds"][combination[3]]
    folder = os.path.join(cwd,
                          hyper["exp_folders"],
                          hyper["domains"],
                          hyper["exps"],
                          hyper["seeds"],
                          )
    exps.append(folder)


def run_train():
    for exp in exps:
        train(exp)

def run_test():
    for exp in exps:
        test(exp)


def run_evaluate():
    folder = os.path.join(cwd, "experiments_trials",
                          # "pacman_test",
                          # "pacman_scosGridTraps2",
                          # "pacman_smallGrid3",
                          # "pacman_smallGrid_new",
                          # "sokoban_6x6_new",
                          "pacman_5x5_full",
                          "ppo",

                          )
    no = os.path.join(folder, "no_shielding")
    soft1 = os.path.join(folder, "soft_shielding")
    hard = os.path.join(folder, "hard_shielding")
    soft2 = os.path.join(folder, "soft_shielding2")

    model_at_step = 100000
    mean_reward, n_deaths = evaluate(no, model_at_step=model_at_step, n_test_episodes=500)
    print("no:", mean_reward, n_deaths)
    # mean_reward, n_deaths = evaluate(soft1, model_at_step=model_at_step, n_test_episodes=500)
    # print("soft:", mean_reward, n_deaths)
    # mean_reward, n_deaths = evaluate(hard, model_at_step=model_at_step, n_test_episodes=500)
    # print("hard:", mean_reward, n_deaths)
    # mean_reward, n_deaths = evaluate(soft2, model_at_step=model_at_step, n_test_episodes=500)
    # print("soft2:", mean_reward, n_deaths)

    # for exp in exps:
    #     evaluate(exp)th.argmax(mass.probs,dim=1)

def main_cluster():
    client = Client("134.58.41.100:8786")

    # with performance_report(filename="dask-report.html"):
    ## some dask computation
    futures = client.map(train, exps)
    results = client.gather(futures)



if __name__ == "__main__":
    run_train()