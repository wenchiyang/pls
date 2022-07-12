from dask.distributed import Client, LocalCluster, performance_report, SSHCluster
import os
from workflows.execute_workflow import train, test, evaluate
import itertools

def run_train():
    for exp in exps:
        train(exp)

def run_test():
    for exp in exps:
        test(exp)

def write_to_file(folder):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join("text.txt")
    with open(path, "w") as f:
        f.write(str(folder))
        f.write(str(dir_path))
        f.write(str(exps))

def run_evaluate():
    folder = os.path.join(dir_path, "experiments_trials3",
                          "goal_finding/7grid5g_gray",
                          "hard_shielding_learned_obs_discrete_100_bal",
                          "seed1"
                          )
    # no = os.path.join(folder, "no_shielding")
    # soft1 = os.path.join(folder, "soft_shielding")
    # hard = os.path.join(folder, "hard_shielding")
    # soft2 = os.path.join(folder, "soft_shielding2")

    # model_at_step = 100000
    model_at_step = "end"
    mean_reward, n_deaths = evaluate(folder, model_at_step=model_at_step, n_test_episodes=500)
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
    hyper_parameters = {
        "exp_folders": ["experiments_trials3"],
        "domains": [
            # "sokoban/2box10map_long",
            # "goal_finding/smallGrid100map",
            # "goal_finding/7grid5g"
            "goal_finding/7grid5g_gray2"
            # "sokoban/2box1map",
            # "sokoban/2box5map",
            # "carracing/onemap"
            # "carracing/sparse_rewards4"
        ],
        "exps": [
            # "test"
            "PPO",
            "PLS_perfect",
            "PLSnoisy_0.1k",
            "PLSnoisy_1k",
            "PLSnoisy_10k",
            "PLSthres_0.1k",
            "PLSthres_1k",
            "PLSthres_10k",
            "VSRL_perfect",
            "VSRL_0.1k",
            "VSRL_1k",
            "VSRL_10k",
            "PLSnoisy_imp_0.1k",
            "PLSnoisy_imp_1k",
            "PLSnoisy_imp_10k",
            ],
        "seeds":
            # ["seed1", "seed2", "seed3", "seed4", "seed5"]
            ["seed1"]
    }

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(dir_path, ".."))

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
        folder = os.path.join(dir_path,
                              hyper["exp_folders"],
                              hyper["domains"],
                              hyper["exps"],
                              hyper["seeds"],
                              )
        exps.append(folder)
    # main_cluster()
    run_train()
    # run_evaluate()