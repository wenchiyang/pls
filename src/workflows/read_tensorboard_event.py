import pandas as pd
import os
from tensorboard.backend.event_processing import event_accumulator
import altair as alt
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
domain_goal_finidng = os.path.abspath(os.path.join(dir_path, "../..", "experiments_trials3", "goal_finding", "smallGrid100map"))
domain_sokoban = os.path.abspath(os.path.join(dir_path, "../..", "experiments_trials3", "sokoban", "2box10map",))
names = {
    "sokoban": domain_sokoban,
    "goal_finding": domain_goal_finidng
}
norms = {
    "sokoban": {"low": -12, "high": 12},
    "goal_finding": {"low": 0, "high": 10}
}
alpha_names = {
    "no_shielding": "no shielding",
    "hard_shielding": "hard shielding",
    "alpha_0.1": "alpha=0.1",
    "alpha_0.3": "alpha=0.3",
    "alpha_0.5": "alpha=0.5",
    "alpha_0.7": "alpha=0.7",
    "alpha_0.9": "alpha=0.9",
    "vsrl": "vsrl"
}
NEW_TAGS = [
    "norm reward",
    "constraint satisfiability"
]
TAGS = [
    "rollout/ep_rew_mean",
    "rollout/#violations",
    # "rollout/success_rate"
    # "safety/ep_abs_safety_impr",
    # "safety/n_deaths"
]
SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
# SEEDS = ["seed1", "seed2"]

def load_dataframe_from_file(path, tag):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    df = pd.DataFrame(ea.Scalars(tag))
    return df

def smooth_dataframe(df, tsboard_smoothing):
    smoothed_df = df.ewm(alpha=(1 - tsboard_smoothing)).mean()
    return smoothed_df

def get_step_value_in_dataframe(df, steps):
    idx = df["step"].sub(steps).abs().idxmin()
    value = df.loc[idx]["value"]
    return value

def normalize_rew(v, norm):
    return (v-norm["low"])/(norm["high"]-norm["low"])

def normalize_vio(v, norm=None):
    return 1-v

def load_dataframe(folder, tag):
    for event_file in os.listdir(folder):
        if "event" not in event_file:
            continue
        path = os.path.join(folder, event_file)
        df = load_dataframe_from_file(path, tag)
        df = smooth_dataframe(df, tsboard_smoothing=0.95)
        return df

def load_single_value(exp,  steps, norm):
    rews = []
    sats = []
    for seed in SEEDS:
        folder = os.path.join(exp, seed)
        df0 = load_dataframe(folder, TAGS[0])
        df1 = load_dataframe(folder, TAGS[1])
        v0 = get_step_value_in_dataframe(df0, steps=steps)
        v1 = get_step_value_in_dataframe(df1, steps=steps)
        ## TODO: this assumes locations and values
        rew = normalize_rew(v0, norm=norm)
        satisfiability = normalize_vio(v1)
        rews.append(rew)
        sats.append(satisfiability)
    avg_rew = np.mean(rews)
    avg_sat = np.mean(sats)
    return [avg_rew, avg_sat]

def load_single_values(domain, alphas, steps, norm):
    d = {}
    for alpha in alphas:
        folder = os.path.join(domain, alpha)
        vs = load_single_value(folder, steps, norm)
        d[alpha_names[alpha]] = dict(zip(NEW_TAGS, vs))
    return d



def make_df(dict, x_title, y_key):
    y_values = []
    for x_key in dict.keys():
        y_values.append(dict[x_key][y_key])
    data = pd.DataFrame({x_title: list(dict.keys()), y_key: y_values})
    return data

def draw(dd, fig_path):
    charts = []
    for i in range(len(NEW_TAGS)):
        data = make_df(dd, x_title="alpha", y_key=NEW_TAGS[i])
        c = alt.Chart(data, title=NEW_TAGS[i]).mark_bar().encode(
            x=alt.X("alpha", sort=list(dd.keys())),
            y=alt.Y(NEW_TAGS[i], title="", scale=alt.Scale(domain=(0, 1)))
        ).properties(
            width=120,
            height=240
        )
        charts.append(c)
    c = charts[0] | charts[1]
    # c.show()
    c.save(fig_path)

def learning_curves(name):
    domain = names[name]
    alphas = [
        "no_shielding",
        "alpha_0.1",
        "alpha_0.3",
        "alpha_0.5",
        "alpha_0.7",
        "alpha_0.9",
        "hard_shielding",
        "vsrl"
    ]

    df_list = []
    for alpha in alphas:
        folder = os.path.join(domain, alpha)
        df_diff_seeds = None
        for seed in SEEDS:
            path = os.path.join(folder, seed)
            df = load_dataframe(path, TAGS[0])
            if df_diff_seeds is None:
                df_diff_seeds = df[["value", "step"]]
                df_diff_seeds = df_diff_seeds.rename(columns={"value": seed})
            else:
                df_diff_seeds[seed] = df["value"]
        df_diff_seeds["value"] = df_diff_seeds[SEEDS].mean(axis=1)
        avg_df = df_diff_seeds[["step", "value"]].copy()
        avg_df["alpha"] = alpha_names[alpha]
        df_list.append(avg_df)
    df_main = pd.concat(df_list)
    fig_path = os.path.join(domain, f"{name}_learning_curves.svg")
    c = alt.Chart(df_main).mark_line().encode(
        x=alt.X("step"),
        y=alt.Y("value"),
        color="alpha"
    )
    # c.show()
    c.save(fig_path)

def many_alpha():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    domain = os.path.join(
        dir_path,
        "../..",
        "experiments_trials2",
        "goal_finding",
        "smallGrid"
        # "sokoban",
        # "2box10map",
    )
    alpha = []
    dd = load_single_values(
        domain = domain,
        alphas = alpha,
        steps=1_000_000,
        norm={"low": -25, "high": 0}
    )

    fig_path = os.path.join(domain, "exp.svg")
    draw(dd, fig_path)

def diff_non_diff(name):
    domain = names[name]
    norm = norms[name]
    alpha = ["hard_shielding", "vsrl"]
    # alpha = ["no_shielding", "no_shielding"]
    dd = load_single_values(
        domain=domain,
        alphas=alpha,
        steps=500_000,
        norm=norm
    )
    fig_path = os.path.join(domain, f"{name}_diff_non_diff.svg")
    draw(dd, fig_path)


# learning_curves("sokoban")
learning_curves("goal_finding")
# diff_non_diff("sokoban")
diff_non_diff("goal_finding")