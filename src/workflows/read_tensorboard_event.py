import pandas as pd
import os
from tensorboard.backend.event_processing import event_accumulator
import altair as alt
from altair import Column
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
domain_goal_finidng = os.path.abspath(os.path.join(dir_path, "../..", "experiments_trials3", "goal_finding", "7grid5g"))
domain_sokoban = os.path.abspath(os.path.join(dir_path, "../..", "experiments_trials3", "sokoban", "2box10map_long"))

# dir_path = "/cw/dtaijupiter/NoCsBack/dtai/wenchi/NeSyProject/experiments_trials3"
# domain_goal_finidng = os.path.join(dir_path, "goal_finding", "7grid5g")
# domain_sokoban = os.path.join(dir_path, "sokoban", "2box10map")

NAMES = {
    "sokoban": domain_sokoban,
    "goal_finding": domain_goal_finidng
}
DOMAIN_ABBR= {
    "sokoban": "Sokoban",
    "goal_finding": "GF"
}
NORMS = {
    "sokoban": {"low": -12, "high": 12},
    "goal_finding": {"low": -10, "high": 10}
}
DOMAIN_NAMES = {
    "sokoban": "sokoban",
    "goal_finding": "goal finding"
}
ALPHA_NAMES_DIFF = {
    "vsrl": "VSRL",
    "hard_shielding": "PLS",
    "no_shielding": "PPO"
}
ALPHA_NAMES = {
    "no_shielding": "0.0",
    "hard_shielding": "1.0",
    "alpha_0.1": "0.1",
    "alpha_0.3": "0.3",
    "alpha_0.5": "0.5",
    "alpha_0.7": "0.7",
    "alpha_0.9": "0.9",
    "vsrl": "vsrl"
}
ALPHA_NAMES_LEARNING_CURVES = {
    "no_shielding": "PPO",
    "hard_shielding": "PLS BASE",
    "alpha_0.1": "PLS",
    "alpha_0.3": "PLS",
    "alpha_0.5": "PLS",
    "alpha_0.7": "PLS",
    "alpha_0.9": "PLS",
    "vsrl": "VSRL"
}
NEW_TAGS = [
    "reward",
    "constraint satisfiability",
    "accepted samples"
]
TAGS = [
    "rollout/ep_rew_mean",
    "rollout/#violations",
    "safety/num_rejected_samples_max"
    # "safety/ep_abs_safety_impr",
    # "safety/n_deaths"
]
SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
# SEEDS = ["seed1"]

def load_dataframe_from_file(path, tag):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    df = pd.DataFrame(ea.Scalars(tag))
    return df

def smooth_dataframe(df, tsboard_smoothing):
    df["value"] = df["value"].ewm(alpha=(1 - tsboard_smoothing)).mean()
    return df

def get_step_value_in_dataframe(df, steps):
    idx = df["step"].sub(steps).abs().idxmin()
    value = df.loc[idx]["value"]
    return value

def normalize_rew(v, norm):
    return (v-norm["low"])/(norm["high"]-norm["low"])

def normalize_vio(v, norm=None):
    return 1-v

def normalize_rej(v, norm=None):
    return 1- (v/100000)

def load_dataframe(folder, tag, smooth=True):
    for event_file in os.listdir(folder):
        if "event" not in event_file:
            continue
        path = os.path.join(folder, event_file)
        df = load_dataframe_from_file(path, tag)
        if smooth:
            df = smooth_dataframe(df, tsboard_smoothing=0.95)
        return df

def load_single_value(exp, steps, norm):
    rews = []
    sats = []
    rejs = []
    for seed in SEEDS:
        folder = os.path.join(exp, seed)
        df0 = load_dataframe(folder, TAGS[0])
        df1 = load_dataframe(folder, TAGS[1])
        v0 = get_step_value_in_dataframe(df0, steps=steps)
        v1 = get_step_value_in_dataframe(df1, steps=steps)
        ## TODO: this assumes locations and values
        rew = normalize_rew(v0, norm=norm)
        rews.append(rew)

        satisfiability = normalize_vio(v1)
        sats.append(satisfiability)

        if "vsrl" in folder:
            df2 = load_dataframe(folder, TAGS[2])
            v2 = get_step_value_in_dataframe(df2, steps=steps)
            rej = normalize_rej(v2)
            rejs.append(rej)

    avg_rew = np.mean(rews)
    avg_sat = np.mean(sats)
    if "vsrl" in folder:
        avg_rej = np.mean(rejs)
        return [avg_rew, avg_sat, avg_rej]
    else:
        return [avg_rew, avg_sat, 1.0]

def load_single_values(domain, alphas, steps, norm, names):
    d = {}
    for alpha in alphas:
        folder = os.path.join(domain, alpha)
        vs = load_single_value(folder, steps, norm)
        d[names[alpha]] = dict(zip(NEW_TAGS, vs))
    return d

def make_df(d, x_title, y_key):
    y_values = []
    for x_key in d.keys():
        y_values.append(d[x_key][y_key])
    data = pd.DataFrame({x_title: list(d.keys()), y_key: y_values})
    return data
    # data = pd.DataFrame.from_dict(d, orient='index').rename_axis(x_title).reset_index()
    # return data

def draw(dd, fig_path):
    charts = []
    for i in range(len(NEW_TAGS)):
        data = make_df(dd, x_title="alpha", y_key=NEW_TAGS[i])
        c = alt.Chart(data, title=NEW_TAGS[i]).mark_bar().encode(
            x=alt.X("alpha", sort=list(dd.keys())),
            y=alt.Y(NEW_TAGS[i], title="", scale=alt.Scale(domain=(0, 1))),
            color="alpha"
        ).properties(
            width=120,
            height=240
        )
        charts.append(c)
    c = charts[0] | charts[1] | charts[2]
    c.show()

def curves(domain_name, curve_type, alphas, names, step_limit, fig_title, fig_title_abbr):
    domain = NAMES[domain_name]

    df_list = []
    for alpha in alphas:
        folder = os.path.join(domain, alpha)
        for seed in SEEDS:
            path = os.path.join(folder, seed)
            df = load_dataframe(path, curve_type, smooth=True)
            df["seed"] = seed
            df["alpha"] = names[alpha]
            if alpha == "vsrl":
                # take only 1M steps
                df = df.drop(df[df.step > 1000000].index)
            df_list.append(df[["value", "step", "seed", "alpha"]])

    df_main = pd.concat(df_list)
    df_main["step"] = df_main["step"] / 1000000
    fig_path = os.path.join(domain, f"{domain_name}_{fig_title}.svg")

    line = alt.Chart(df_main).mark_line().encode(
        x=alt.X("step",
                scale=alt.Scale(domain=(0, step_limit)),
                axis=alt.Axis(
                    format='~s',
                    title="M steps",
                    grid=False)),
        y=alt.Y("mean(value)",
                axis=alt.Axis(
                    # format='~s',
                    title=f"Avg {fig_title_abbr} / Epis",
                    grid=False)),
        color=alt.Color("alpha",
                        legend=alt.Legend(
                            title=f"Avg {fig_title_abbr} on {DOMAIN_ABBR[domain_name]}",
                            orient='none',
                            direction='horizontal',
                            legendX=-10, legendY=-35,
                            titleAnchor='middle'
                        ))
    ).properties(
            width=200,
            height=100
        )
    band = alt.Chart(df_main).mark_errorband(extent='ci').encode(
        x=alt.X("step"),
        y=alt.Y("value",title=""),
        color="alpha"
    )
    c = line + band
    # c.show()
    c.save(fig_path)


def diff_non_diff_new(nnn):
    dds=[]
    for name in nnn:
        domain = NAMES[name]
        norm = NORMS[name]
        alpha = ["no_shielding", "vsrl", "hard_shielding"]
        # alpha = ["no_shielding", "no_shielding"]
        dd = load_single_values(
            domain=domain,
            alphas=alpha,
            steps=500_000,
            norm=norm,
            names=ALPHA_NAMES_DIFF
        )
        dds.append(dd)
    fig_path = os.path.abspath(os.path.join(dir_path, "../..", "experiments_trials3", "results.svg"))
    # fig_path = os.path.join(domain, f"{name}_diff_non_diff.svg")
    draw_dds(dds, nnn, fig_path, NEW_TAGS)

def many_alpha_new(nnn):
    dds=[]
    for name in nnn:
        domain = NAMES[name]
        norm = NORMS[name]
        alpha = [
            "no_shielding",
            "alpha_0.1",
            "alpha_0.3",
            "alpha_0.5",
            "alpha_0.7",
            "alpha_0.9",
            "hard_shielding"]
        # alpha = ["no_shielding", "no_shielding"]
        dd = load_single_values(
            domain=domain,
            alphas=alpha,
            steps=500_000,
            norm=norm,
            names=ALPHA_NAMES
        )
        dds.append(dd)
    fig_path = os.path.abspath(os.path.join(dir_path, "../..", "experiments_trials3", "alpha.svg"))
    draw_dds(dds, nnn, fig_path, NEW_TAGS[:2])

def draw_dds(dds, nnn, fig_path, tags):
    charts = []
    for i in range(len(tags)):
        datas = []
        for j,dd in enumerate(dds):
            data = make_df(dd, x_title="alpha", y_key=NEW_TAGS[i])
            data["domain"] = DOMAIN_NAMES[nnn[j]]
            datas.append(data)
        dataframmm = pd.concat(datas)
        c = alt.Chart(dataframmm, title="").mark_bar().encode(
            column=Column('domain', title=NEW_TAGS[i]),
            x=alt.X("alpha", sort=list(dd.keys()), title=None),
            y=alt.Y(NEW_TAGS[i], title=None, scale=alt.Scale(domain=(0, 1))),
            color=alt.Color("alpha", legend=None, scale=alt.Scale(scheme='accent')),
        ).properties(
            width=60,
            height=240
        )
        charts.append(c)
    if len(tags) == 3:
        c = charts[0] | charts[1] | charts[2]
    elif len(tags) == 2:
        c = charts[0] | charts[1]
    c.configure_view(
            strokeWidth=0
        )
    # c.show()
    c.save(fig_path)
    # data = make_df(dd, x_title="alpha")
    # c = alt.Chart(data, title=NEW_TAGS).mark_bar().encode(
    #     x=alt.X("alpha", sort=list(dd.keys())),
    #     y=alt.Y(NEW_TAGS, title="", scale=alt.Scale(domain=(0, 1))),
    #     color="alpha"
    # ).properties(
    #     width=120,
    #     height=240
    # )

curves("sokoban",
        alphas=[
            "no_shielding",
            "hard_shielding",
            "alpha_0.3",
            "vsrl"
        ],
        curve_type=TAGS[1], # violation_curves
        names=ALPHA_NAMES_LEARNING_CURVES,
        step_limit=5,
        fig_title="violation_curves",
        fig_title_abbr="Violation"
                )

curves("goal_finding",
        alphas=[
            "no_shielding",
            "hard_shielding",
            "alpha_0.3",
            "vsrl"
        ],
        curve_type=TAGS[1], # violation_curves
        names=ALPHA_NAMES_LEARNING_CURVES,
        step_limit=1,
        fig_title="violation_curves",
        fig_title_abbr="Violation"
        )

# curves("sokoban",
#         alphas=[
#             "no_shielding",
#             "hard_shielding",
#             "alpha_0.3",
#             "vsrl"
#         ],
#         curve_type=TAGS[0], # learning_curves
#         names=ALPHA_NAMES_LEARNING_CURVES,
#         step_limit=5,
#         fig_title="learning_curves",
#         fig_title_abbr="Return"
#         )
#
# curves("goal_finding",
#         alphas=[
#             "no_shielding",
#             "hard_shielding",
#             "alpha_0.3",
#             "vsrl"
#         ],
#         curve_type=TAGS[0], # learning_curves
#         names=ALPHA_NAMES_LEARNING_CURVES,
#         step_limit=1,
#         fig_title="learning_curves",
#         fig_title_abbr="Return"
#         )

# diff_non_diff_new(["goal_finding", "sokoban"])
# many_alpha_new(["goal_finding", "sokoban"])


