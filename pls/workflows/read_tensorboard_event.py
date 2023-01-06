import pandas as pd
import os
from tensorboard.backend.event_processing import event_accumulator
import altair as alt
from altair import Column
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
domain_goal_finding1 = os.path.join(dir_path, "../..", "experiments5", "goal_finding", "small0")
domain_goal_finding2 = os.path.join(dir_path, "../..", "experiments5", "goal_finding", "small2")
domain_pacman1 = os.path.join(dir_path, "../..", "experiments5", "pacman", "small3")
domain_pacman2 = os.path.join(dir_path, "../..", "experiments5", "pacman", "small4")
domain_carracing1 = os.path.join(dir_path, "../..", "experiments5", "carracing", "map0")
domain_carracing2 = os.path.join(dir_path, "../..", "experiments5", "carracing", "map2")

NAMES = {
    "goal_finding1": domain_goal_finding1,
    "goal_finding2": domain_goal_finding2,
    "pacman1": domain_pacman1,
    "pacman2": domain_pacman2,
    "carracing1": domain_carracing1,
    "carracing2": domain_carracing2,
}
DOMAIN_ABBR = {
    "goal_finding1": "Stars1",
    "goal_finding2": "Stars2",
    "pacman1": "pacman1",
    "pacman2": "pacman2",
    "carracing1": "CR1",
    "carracing2": "CR2"
}
NORMS_REW = {
    "goal_finding1": {"low": 0, "high": 45},
    "goal_finding2": {"low": 0, "high": 45},
    "pacman1": {"low": 0, "high": 40},
    "pacman2": {"low": 0, "high": 40},
    "carracing1": {"low": 0, "high": 900},
    "carracing2": {"low": 0, "high": 900}
}

NORMS_VIO = {
    "goal_finding1": {"low": 0, "high": 15000},
    "goal_finding2": {"low": 0, "high": 15000},
    "pacman1": {"low": 3500, "high": 7000},
    "pacman2": {"low": 3500, "high": 15000},
    "carracing1": {"low": 0, "high": 1000},
    "carracing2": {"low": 0, "high": 1000}
}

ALPHA_NAMES_LEARNING_CURVES = {
    "PPO": "PPO",
    "PLPGperf": "PLPG",
    "PLPGperf3": "PLPG",
    "VSRLperf": "VSRL",
    "VSRLthres": "VSRL",
    "PLPGnoisy": "PLPG",
    "PLPGnoisy3": "PLPG",
    "PLPG_LTperf": "PLPG_LTperf",
    "PLPG_STperf": "PLPG_STperf",
    "PLPG_LTnoisy": "PLPG_LTnoisy",
    "PLPG_STnoisy": "PLPG_STnoisy",
    "epsVSRLthres0.005": "ε-VSRL",
    "epsVSRLthres0.01": "ε-VSRL",
    "epsVSRLthres0.1": "ε-VSRL",
}
NEW_TAGS = [
    "Return",
    "Violation",
    "Safety",
]
TAGS = [
    "rollout/ep_rew_mean",
    "safety/n_deaths",
    "safety/ep_rel_safety_shielded",
]
SEEDS = [
    "seed1", 
    "seed2",
    #"seed3",
    "seed4",
    "seed5"
]

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

def get_largest_step_value_in_dataframe(df, steps):
    idx = df["step"].sub(steps).abs().idxmin()
    largest_value_up_till_idx = df.loc[:idx]["value"].max()
    return largest_value_up_till_idx

def normalize_rew(v, norm):
    return (v-norm["low"])/(norm["high"]-norm["low"])

def normalize_vio(v, norm=None):
    return (v-norm["low"])/(norm["high"]-norm["low"])

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
    value = get_largest_step_value_in_dataframe(df, steps=n_step)
    return value

def extract_values(folder, name):
    exp_names = [
        "PPO",
        "VSRLperf",
        # "PLPGperf",
        "PLPGperf3",
        "VSRLthres",
        "epsVSRLthres0.005",
        #"epsVSRLthres0.1",
        #"PLPGnoisy",
        "PLPGnoisy3"
    ]
    exp_names = [
        "PLPG_LTperf",
        "PLPG_STperf",
        "PLPGperf3",
        "PLPG_LTnoisy",
        "PLPG_STnoisy",
        "PLPGnoisy3"
    ]
    exp_names = [
        #"epsVSRLthres0.005",
        #"epsVSRLthres0.01",
        "epsVSRLthres0.05",
        #"epsVSRLthres0.1"
        #"PLPGperf4",
        #"PLPGperf3",
        #"PLPGperf5",
        #"PLPGperf2"
    ]
    tags = [
        "rollout/ep_rew_mean", 
        "safety/ep_rel_safety_shielded", 
        "safety/n_deaths"
    ]

    results = {}
    for exp in exp_names:
        rs, ss, vs = [], [], []
        for seed in SEEDS:
            exp_folder = os.path.join(folder, exp, seed)
            r = load_step_value(exp_folder, tags[0], 600_000)
            v = load_step_value(exp_folder, tags[2], 600_000)
            rs.append(r)
            vs.append(v)
        avg_r = sum(rs)/len(rs)
        avg_v = sum(vs)/len(vs)

        results[exp] = {
            "r": avg_r, "v": avg_v
        }

    for exp in exp_names: 
        print(f"\t{exp}", end =" ")
    print()
    for exp in exp_names:
        print(f"\t{normalize_rew(results[exp]['r'], NORMS_REW[name]):.2f} / {normalize_vio(results[exp]['v'], NORMS_VIO[name]):.2f}", end =" ")
    print()


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

def curves(domain_name, curve_type, exp_names, names, step_limit, figure_height=100, fig_title="", setting=""):
    """
    Plot a safety or reward curve of the experiment of "domain_name/alpha" until "step_limit" steps
    curve_type: "rollout/ep_rew_mean" or "rollout/#violations"
    """
    domain = NAMES[domain_name]
    norm_rew = NORMS_REW[domain_name]
    df_list = []
    for exp_name in exp_names:
        folder = os.path.join(domain, exp_name)
        for seed in SEEDS:
            path = os.path.join(folder, seed)
            df = load_dataframe(path, curve_type, smooth=True)
            if curve_type==TAGS[0]:
                df["value"] = df["value"].apply(lambda x: normalize_rew(x, norm_rew))
                df["value"] = df["value"].cummax()
            df["seed"] = seed
            df["alpha"] = names[exp_name]
            # take only step_limit steps
            df = df.drop(df[df.step > step_limit].index)
            df_list.append(df[["value", "step", "seed", "alpha"]])

    df_main = pd.concat(df_list)
    df_main["step"] = df_main["step"] / 100_000

    if len(exp_names) == 3:
        legendX = 25
        legendY = -35
    elif len(exp_names) == 4:
        legendX = -10
        legendY = -35


    line = alt.Chart(df_main).mark_line().encode(
        x=alt.X("step",
                scale=alt.Scale(domain=(0, step_limit/100_000)),
                axis=alt.Axis(
                    format='~s',
                    title="100k steps",
                    grid=False)),
        y=alt.Y("mean(value)",
                axis=alt.Axis(
                    format='.1',
                    grid=False)),
        color=alt.Color("alpha",
                        legend=alt.Legend(
                            title=f"{fig_title} on {DOMAIN_ABBR[domain_name]}{setting}",
                            orient='none',
                            direction='horizontal',
                            legendX=legendX, legendY=legendY,
                            titleAnchor='middle'
                        ),
                        scale=alt.Scale(domain=[names[exp_name] for exp_name in exp_names], range=["red", "blue", "gray", "green"][:len(exp_names)])
                        )
    ).properties(
            width=200,
            height=figure_height
        )
    band = alt.Chart(df_main).mark_errorband(extent='ci', opacity=0.1).encode(
        x=alt.X("step"),
        y=alt.Y("value", title=""),
        color=alt.Color("alpha",
                         sort = [names[exp_name] for exp_name in exp_names],
                         legend=None
        )
    )
    c = alt.layer(band, line).resolve_legend(color='independent')
    # c.show()
    fig_path = os.path.join(domain, f"{DOMAIN_ABBR[domain_name]}_{fig_title}{setting}.svg")
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



# extract_values(domain_goal_finding1, "goal_finding1")
# extract_values(domain_goal_finding2, "goal_finding2")
#extract_values(domain_pacman1, "pacman1")
#extract_values(domain_pacman2, "pacman2")
extract_values(domain_carracing1, "carracing1")
# extract_values(domain_carracing2, "carracing2")

graph_settings = {
        "domain": [
            #"goal_finding1",
            #"goal_finding2",
            #"pacman1",
            #"pacman2"
            #"carracing1",
            #"carracing2"
        ],
        "exp_names": [
            ["PPO", "VSRLperf", "PLPGperf3"], # det. safety
            ["PPO", "VSRLthres", "epsVSRLthres0.005", "PLPGnoisy3"], # prob. safety
        ],
        "types": ["Acc Violation", "P(safety)", "Avg Return"],
        "curve_types": [TAGS[1], TAGS[2], TAGS[0]]
    }


POSTFIX = [" (perf)", " (noisy)"]

for domain in graph_settings["domain"]:
    for n, exp_name in enumerate(graph_settings["exp_names"]):
        for j, t in enumerate(graph_settings["types"]):
            print(f"{exp_name}\n" + \
                  f"{graph_settings['curve_types'][j]}\n" + \
                  f"{t}\n")
            curves(domain,
                   exp_names=exp_name,
                   curve_type=graph_settings["curve_types"][j],
                   names=ALPHA_NAMES_LEARNING_CURVES,
                   step_limit=700_000,
                   fig_title=t,
                   setting=POSTFIX[n]
                   )

