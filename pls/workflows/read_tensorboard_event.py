import pandas as pd
import os
from tensorboard.backend.event_processing import event_accumulator
import altair as alt
from altair import Column
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
domain_goal_finidng = os.path.join(dir_path, "../..", "experiments4", "goal_finding", "small")
domain_sokoban = os.path.join(dir_path, "../..", "experiments_trials3", "sokoban", "2box5map")
domain_carracing = os.path.join(dir_path, "../..", "experiments_trials3", "carracing", "sparse_rewards4")

# dir_path = "/cw/dtaijupiter/NoCsBack/dtai/wenchi/pls/experiments_trials3"
# domain_goal_finidng = os.path.join(dir_path, "goal_finding", "7grid5g")
# domain_sokoban = os.path.join(dir_path, "sokoban", "2box10map")

NAMES = {
    "sokoban": domain_sokoban,
    "goal_finding": domain_goal_finidng,
    "carracing": domain_carracing
}
DOMAIN_ABBR= {
    "sokoban": "Sokoban",
    "goal_finding": "Stars",
    "carracing": "CR"
}
NORMS_REW = {
    "sokoban": {"low": -12, "high": 4},
    "goal_finding": {"low": 0, "high": 60},
    "carracing": {"low": -50, "high": 800}
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
    "hard_shielding": "PLS",
    "PPO": "PPO",
    "PLSperf": "PLSperf",
    "PLSnoisy": "PLSnoisy",
    "PLSthres": "PLSthres",
    "VSRLperf": "VSRLperf",
    "VSRLthres": "VSRLthres"
}
NEW_TAGS = [
    "Return",
    "Safety",
    "Rejected Samples"
]
TAGS = [
    "rollout/ep_rew_mean",
    "rollout/#violations",
    "safety/num_rejected_samples_max",
    "safety/ep_abs_safety_shielded",
    "safety/ep_rel_safety_shielded",
]
SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
# SEEDS = ["PPO"]

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
    # return 1- (v/100000)
    return v

def load_dataframe(folder, tag, smooth=True):
    for event_file in os.listdir(folder):
        if "event" not in event_file:
            continue
        path = os.path.join(folder, event_file)
        df = load_dataframe_from_file(path, tag)
        if smooth:
            df = smooth_dataframe(df, tsboard_smoothing=0.95)
        return df

def load_step_value(folder, tag, n_step):
    df = load_dataframe(folder, tag)
    value = get_step_value_in_dataframe(df, steps=n_step)
    return value

def extract_values():
    folder = os.path.join(dir_path, "../..", "experiments_trials3", "goal_finding", "7grid5g_gray2")
    exp_names = [
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
    ]

    tags = ["rollout/ep_rew_mean", "safety/ep_abs_safety_shielded", "safety/ep_rel_safety_shielded"]

    print("REWARD")
    for exp in exp_names:
        exp_folder = os.path.join(folder, exp, "PPO")
        r = load_step_value(exp_folder, tags[0], 500_000)
        s1 = load_step_value(exp_folder, tags[1], 500_000)
        s2 = load_step_value(exp_folder, tags[2], 500_000)
        print(f"{exp}:")
        print(f"\tREWARD: \t\t{r}")
        print(f"\tABS SAFETY: \t\t{s1}")
        print(f"\tREL SAFETY: \t\t{s2}")

# extract_values()

def load_single_value_rej_vsrl(exp, steps):
    rejs = []
    for seed in SEEDS:
        folder = os.path.join(exp, "vsrl", seed)
        df2 = load_dataframe(folder, TAGS[2])
        v2 = get_step_value_in_dataframe(df2, steps=steps)
        rej = normalize_rej(v2)
        rejs.append(rej)

    avg_rej = np.mean(rejs)
    return avg_rej

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

def curves(domain_name, curve_type, exp_names, names, step_limit, fig_title_abbr, figure_height=100, fig_title=""):
    """
    Plot a safety or reward curve of the experiment of "domain_name/alpha" until "step_limit" steps
    curve_type: "rollout/ep_rew_mean" or "rollout/#violations"
    """
    domain = NAMES[domain_name]
    norm = NORMS_REW[domain_name]
    df_list = []
    for exp_name in exp_names:
        folder = os.path.join(domain, exp_name)
        for seed in SEEDS:
            path = os.path.join(folder, seed)
            df = load_dataframe(path, curve_type, smooth=True)
            if curve_type==TAGS[0]:
                df["value"] = df["value"].apply(lambda x: normalize_rew(x, norm))
            df["seed"] = seed
            df["alpha"] = names[exp_name]
            # take only step_limit steps
            df = df.drop(df[df.step > step_limit].index)
            df_list.append(df[["value", "step", "seed", "alpha"]])

    df_main = pd.concat(df_list)
    df_main["step"] = df_main["step"] / 100_000

    line = alt.Chart(df_main).mark_line().encode(
        x=alt.X("step",
                scale=alt.Scale(domain=(0, step_limit/100_000)),
                axis=alt.Axis(
                    format='~s',
                    title="100k steps",
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
                        ),
                        # scale=alt.Scale(domain=["PPO", "VSRL", "PLS", "\u03B1PLS"], range=["red", "blue", "gray", "green"]),
                        scale=alt.Scale(domain=exp_names, range=["red", "blue", "gray", "green"][:len(exp_names)])
                        )
    ).properties(
            width=200, #200
            height=figure_height #100
        )
    band = alt.Chart(df_main).mark_errorband(extent='ci', opacity=0.1).encode(
        x=alt.X("step"),
        y=alt.Y("value",title=""),
        color=alt.Color("alpha", 
                         # sort=["PPO", "VSRL", "PLS", "\u03B1PLS"],
                         sort=["PPO", "PLSperf", "PLSnoisy", "PLSthres"],
                         legend=None
        )
    )
    c = alt.layer(band, line).resolve_legend(color='independent')
    # c.show()
    fig_path = os.path.join(domain, f"{DOMAIN_ABBR[domain_name]}_{fig_title}.svg")
    c.save(fig_path)

def safety_optimality_df(domain_name, exp_names, n_step):
    norm = NORMS_REW[domain_name]
    domain = NAMES[domain_name]
    data = []
    for exp_name in exp_names:
        for seed in SEEDS:
            folder = os.path.join(domain, exp_name, seed)
            safety = load_step_value(folder, TAGS[1], n_step)
            optimality = load_step_value(folder, TAGS[0], n_step)
            safety = normalize_vio(safety)
            optimality = normalize_rew(optimality, norm)
            data.append([ALPHA_NAMES[exp_name], seed, safety, optimality])
    df = pd.DataFrame(data, columns=["alpha", "seed", "safety", "optimality"])
    return df

def safety_optimality_draw(domain_name, n_step, x_axis_range, y_axis_range):
    """
    Draw a 2D safety-optimality figure of the experiment of "domain_name" at "n_step" steps
    """
    exp_names = ["no_shielding", "alpha_0.1", "alpha_0.3", "alpha_0.5", "alpha_0.7", "alpha_0.9", "hard_shielding"]
    df = safety_optimality_df(domain_name, exp_names, n_step)
    x_tick_range = [int(v*10) for v in x_axis_range]
    x_tick_values = [v/10 for v in range(x_tick_range[0], x_tick_range[1]+1)]
    y_tick_range = [int(v*10) for v in y_axis_range]
    y_tick_values = [v/10 for v in range(y_tick_range[0], y_tick_range[1]+1)]
    c = alt.Chart(df, title=f"Safety-Return on {DOMAIN_ABBR[domain_name]}").mark_point().encode(
        x=alt.X("safety",
                scale=alt.Scale(domain=x_axis_range),
                axis=alt.Axis(
                    format='.1',
                    values=x_tick_values,
                    title="Safety",
                    grid=False)),
        y=alt.Y("optimality", title=None,
                scale=alt.Scale(domain=y_axis_range),
                axis=alt.Axis(
                    format='.1',
                    values=y_tick_values,
                    title="Return",
                    grid=False)),
        color=alt.Color("alpha",
                        legend=alt.Legend(
                            title=None,
                            orient='none',
                            direction='horizontal',
                            #legendX=10, legendY=70,
                            legendX=20, legendY=-30,
                            columns=4,
                            titleAnchor='middle'
                        ),
                        scale=alt.Scale(scheme='redblue')),
    ).properties(
        width=200,
        height=200
    )
    # c.show()
    fig_path = os.path.join(NAMES[domain_name], f"{domain_name}_safety_return.svg")
    c.save(fig_path)

def rejected_samples(domain_names):
    dds=[]
    for name in domain_names:
        domain = NAMES[name]
        # norm = NORMS_REW[name]
        # alpha = ["no_shielding", "vsrl", "hard_shielding"]
        dd = load_single_value_rej_vsrl(
            exp=domain,
            steps=1_000_000
        )
        dds.append(dd)

    # plot_bar_chart(dds, domain_names, fig_path, NEW_TAGS[2])
    dataframmm=pd.DataFrame({"alpha": [DOMAIN_ABBR[d] for d in domain_names], "Rejected Samples": dds})

    c = alt.Chart(dataframmm, title="Rejected Samples").mark_bar().encode(
        x=alt.X("alpha",
                sort=[DOMAIN_ABBR[d] for d in domain_names],
                title=None),
        y=alt.Y("Rejected Samples",
                title=None,
                axis=alt.Axis(format='~s'),
                scale=alt.Scale(domain=(0, 100_000))
                ),
        color=alt.Color("alpha", legend=None, scale=alt.Scale(scheme='accent')),
    ).properties(
        width=100,
        height=120
    )
    # c.show()
    fig_path = os.path.join(dir_path, "../..", "experiments_trials3", f"rejected_samples.svg")
    c.save(fig_path)

def get_number_of_rejected_samples(name):
    """
    Returns the number of rejected samples of the experiment of "name" at 500k steps
    """
    domain = NAMES[name]
    rejs = []
    for seed in SEEDS:
        folder = os.path.join(domain, "vsrl", seed)
        df = load_dataframe(folder, TAGS[2])
        df = df.drop(df[df.step > 500000].index)
        v2 = df["value"].mean()
        rej = normalize_rej(v2)
        rejs.append(rej)
    avg_rej = np.mean(rejs)
    return avg_rej

def get_time_sample_one_action(name, alpha):
    """
    Returns the runtime of the experiment of "name/alpha" for 500k steos
    """
    domain = NAMES[name]
    times = []
    for seed in SEEDS:
        folder = os.path.join(domain, alpha, seed)
        df = load_dataframe(folder, TAGS[0])
        df = df.drop(df[df.step > 500000].index)
        time = df["wall_time"].max() - df["wall_time"].min()
        times.append(time)
    avg_time = np.mean(times)
    return avg_time/500000


# SEEDS=["PPO"]
# print(get_time_sample_one_action("goal_finding", "hard_shielding"))
# print(get_time_sample_one_action("sokoban", "hard_shielding"))
# print(get_time_sample_one_action("carracing", "hard_shielding"))
#
# print(get_time_sample_one_action("goal_finding", "vsrl"))
# print(get_time_sample_one_action("sokoban", "vsrl"))
# print(get_time_sample_one_action("carracing", "vsrl"))

# print(get_number_of_rejected_samples("goal_finding"))
# print(get_number_of_rejected_samples("sokoban"))
# print(get_number_of_rejected_samples("carracing"))

curves("sokoban",
       exp_names=[
           "PPO", "PLSperf", "VSRLperf"
       ],
       curve_type=TAGS[1], # violation_curves
       names=ALPHA_NAMES_LEARNING_CURVES,
       step_limit=1_000_000,
       fig_title_abbr="Violation",
       fig_title="Violation"
       )

curves("sokoban",
       exp_names=[
           "PPO", "PLSperf", "VSRLperf"
       ],
       curve_type=TAGS[4], # safety
       names=ALPHA_NAMES_LEARNING_CURVES,
       step_limit=1_000_000,
       fig_title_abbr="Safety",
       fig_title="Safety"
       )

curves("sokoban",
       exp_names=[
           "PPO", "PLSperf", "VSRLperf"
       ],
       curve_type=TAGS[0], # learning_curves
       names=ALPHA_NAMES_LEARNING_CURVES,
       step_limit=1_000_000,
       fig_title_abbr="Return",
       fig_title="Return"
       )
#
# curves("goal_finding",
#        exp_names=[
#            "PPO", "PLSthres", "VSRLthres", "PLSnoisy"
#        ],
#        curve_type=TAGS[1], # violation_curves
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=1_000_000,
#        fig_title_abbr="Violation",
#        fig_title="Violation_Noisy"
#        )
#
# curves("goal_finding",
#        exp_names=[
#            "PPO", "PLSthres", "VSRLthres", "PLSnoisy"
#        ],
#        curve_type=TAGS[4], # safety
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=1_000_000,
#        fig_title_abbr="Safety",
#        fig_title="Safety_Noisy"
#        )
#
# curves("goal_finding",
#        exp_names=[
#            "PPO", "PLSthres", "VSRLthres", "PLSnoisy"
#        ],
#        curve_type=TAGS[0], # learning_curves
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=1_000_000,
#        fig_title_abbr="Return",
#        fig_title="Return_Noisy")



#
# curves("carracing",
#        alphas=[
#            "no_shielding",
#            "hard_shielding",
#            "alpha_0.5",
#            "vsrl"
#        ],
#        curve_type=TAGS[1], # violation_curves
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=500_000,
#        fig_title="violation_curves",
#        fig_title_abbr="Violation")
#
# curves("carracing",
#       alphas=[
#            "no_shielding",
#            "hard_shielding",
#            "alpha_0.5",
#            "vsrl"
#        ],
#        curve_type=TAGS[0], # learning_curves
#        names=ALPHA_NAMES_LEARNING_CURVES, # ALPHA_NAMES_LEARNING_CURVES
#        step_limit=500_000,
#        fig_title="learning_curves",
#        fig_title_abbr="Return")

# safety_optimality_draw("sokoban", n_step=500_000, x_axis_range=[0.0, 1.0], y_axis_range=[0.0, 0.6])
# safety_optimality_draw("goal_finding", n_step=500_000, x_axis_range=[0.6, 1.0], y_axis_range=[0.4, 1.0])
# safety_optimality_draw("carracing", n_step=500_000, x_axis_range=[0.0, 0.3], y_axis_range=[0.0, 0.8])

# WE USE THE FOLLOWING
# curves("goal_finding",
#        exp_names=[
#            "PPO", "PLSperf", "VSRLperf"
#        ],
#        curve_type=TAGS[1], # violation_curves
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=1_000_000,
#        fig_title_abbr="Violation",
#        fig_title="Violation"
#        )
#
# curves("goal_finding",
#        exp_names=[
#            "PPO", "PLSperf", "VSRLperf"
#        ],
#        curve_type=TAGS[4], # safety
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=1_000_000,
#        fig_title_abbr="Safety",
#        fig_title="Safety"
#        )
#
# curves("goal_finding",
#        exp_names=[
#            "PPO", "PLSperf", "VSRLperf"
#        ],
#        curve_type=TAGS[0], # learning_curves
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=1_000_000,
#        fig_title_abbr="Return",
#        fig_title="Return"
#        )
#
# curves("goal_finding",
#        exp_names=[
#            "PPO", "PLSthres", "VSRLthres", "PLSnoisy"
#        ],
#        curve_type=TAGS[1], # violation_curves
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=1_000_000,
#        fig_title_abbr="Violation",
#        fig_title="Violation_Noisy"
#        )
#
# curves("goal_finding",
#        exp_names=[
#            "PPO", "PLSthres", "VSRLthres", "PLSnoisy"
#        ],
#        curve_type=TAGS[4], # safety
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=1_000_000,
#        fig_title_abbr="Safety",
#        fig_title="Safety_Noisy"
#        )
#
# curves("goal_finding",
#        exp_names=[
#            "PPO", "PLSthres", "VSRLthres", "PLSnoisy"
#        ],
#        curve_type=TAGS[0], # learning_curves
#        names=ALPHA_NAMES_LEARNING_CURVES,
#        step_limit=1_000_000,
#        fig_title_abbr="Return",
#        fig_title="Return_Noisy")