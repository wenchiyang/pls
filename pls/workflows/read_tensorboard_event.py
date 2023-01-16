import altair
import pandas as pd
import os
from tensorboard.backend.event_processing import event_accumulator
import altair as alt
from altair import Column
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


FOLDERS = {
    "goal_finding1": os.path.join(dir_path, "../..", "experiments5", "goal_finding", "small0"),
    "goal_finding2": os.path.join(dir_path, "../..", "experiments5", "goal_finding", "small2"),
    "pacman1": os.path.join(dir_path, "../..", "experiments5", "pacman", "small3"),
    "pacman2": os.path.join(dir_path, "../..", "experiments5", "pacman", "small4"),
    "carracing1": os.path.join(dir_path, "../..", "experiments5", "carracing", "map0"),
    "carracing2": os.path.join(dir_path, "../..", "experiments5", "carracing", "map2"),
}
DOMAIN_ABBR = {
    "goal_finding1": "Stars1",
    "goal_finding2": "Stars2",
    "pacman1": "Pac1",
    "pacman2": "Pac2",
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
    "PLPGperf2": "PLPG",
    "PLPGperf3": "PLPG",
    "PLPGperf4": "PLPG",
    "VSRLperf": "VSRL",
    "VSRLthres": "VSRL",
    "PLPGnoisy": "PLPG",
    "PLPGnoisy2": "PLPG",
    "PLPGnoisy3": "PLPG",
    "PLPGnoisy4": "PLPG",
    "PLPG_LTperf": "PLPG_LTperf",
    "PLPG_STperf": "PLPG_STperf",
    "PLPG_LTnoisy": "PLPG_LTnoisy",
    "PLPG_STnoisy": "PLPG_STnoisy",
    "epsVSRLthres0.005": "ε-VSRL",
    "epsVSRLthres0.01": "ε-VSRL",
    "epsVSRLthres0.05": "ε-VSRL",
    "epsVSRLthres0.1": "ε-VSRL",
    "epsVSRLthres0.2": "ε-VSRL",
    "epsVSRLthres0.5": "ε-VSRL",
}

TAGS = [
    "rollout/ep_rew_mean",
    "safety/n_deaths",
    "safety/ep_rel_safety_shielded",
]
SEEDS = [
    "seed1", 
    "seed2",
    "seed3",
    "seed4",
    "seed5"
]

def load_dataframe(folder, tags):
    """This function loads an event file from folder and returns len(tags) dataframes."""
    n_events = 0
    for file_name in os.listdir(folder):
        if "event" in file_name:
            event_file = file_name
            n_events += 1
    assert n_events == 1, f"Too many event files in {folder}."

    # Load dataframe from file
    path = os.path.join(folder, event_file)
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    tsboard_smoothing=0.95
    dfs = []
    for tag in tags:
        df = pd.DataFrame(ea.Scalars(tag))
        df = df.drop(columns=['wall_time'])
        # Smooth the return curve and safety curve
        if "rew" in tag or "safety_shielded" in tag:
            df["value"] = df["value"].ewm(alpha=(1 - tsboard_smoothing)).mean()
        dfs.append(df)
    return dfs

def get_step_value_in_dataframe(df, steps):
    idx = df["step"].sub(steps).abs().idxmin()
    value = df.loc[idx]["value"]
    return value

def get_value_on_step_in_dataframe(df, step):
    idx = df["step"].sub(step).abs().idxmin()
    largest_value_up_till_idx = df.loc[:idx]["value"].max()
    return largest_value_up_till_idx

def normalize_rew(v, norm):
    return (v-norm["low"])/(norm["high"]-norm["low"])

def normalize_vio(v, norm=None):
    return (v-norm["low"])/(norm["high"]-norm["low"])

def load_step_values(folder, tags, n_step):
    dfs = load_dataframe(folder, tags)
    values = []
    for df in dfs:
        value = get_value_on_step_in_dataframe(df, step=n_step)
        values.append(value)
    return values

def extract_values(name, exp_names):
    folder = FOLDERS[name]
    results = {}
    for exp in exp_names:
        rs, vs = [], []
        for seed in SEEDS:
            exp_folder = os.path.join(folder, exp, seed)
            r, v = load_step_values(exp_folder, TAGS[0:2], 600_000)
            rs.append(r)
            vs.append(v)
        avg_r = sum(rs)/len(rs)
        avg_v = sum(vs)/len(vs)

        results[exp] = {
            "r": avg_r, "v": avg_v
        }


    for exp in exp_names:
        print(f"\t{normalize_rew(results[exp]['r'], NORMS_REW[name]):.2f} / {normalize_vio(results[exp]['v'], NORMS_VIO[name]):.2f}", end =" ")
    print()
    return results


def curves_old(domain_name, curve_type, exp_names, names, step_limit, figure_height=100, fig_title="", setting=""):
    """
    Plot a safety or reward curve of the experiment of "domain_name/alpha" until "step_limit" steps
    curve_type: "rollout/ep_rew_mean" or "rollout/#violations"
    """
    folder = FOLDERS[domain_name]
    norm_rew = NORMS_REW[domain_name]
    df_list = []
    for exp_name in exp_names:
        exp_folder = os.path.join(folder, exp_name)
        for seed in SEEDS:
            exp_folder_path = os.path.join(exp_folder, seed)
            df = load_dataframe(exp_folder_path, [curve_type])[0]
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
    c.show()
    fig_path = os.path.join(domain, f"{DOMAIN_ABBR[domain_name]}_{fig_title}{setting}.svg")
    c.save(fig_path)


def curves(domain_name, exp_names, names, step_limit, row):
    """
    Plot a safety or reward curve of the experiment of "domain_name/alpha" until "step_limit" steps
    curve_type: "rollout/ep_rew_mean" or "rollout/#violations"
    """
    folder = FOLDERS[domain_name]
    df_list_rew, df_list_violation, df_list_safety = [], [], []
    for exp_name in exp_names:
        exp_folder = os.path.join(folder, exp_name)
        for seed in SEEDS:
            exp_folder_path = os.path.join(exp_folder, seed)
            dfs = load_dataframe(exp_folder_path, TAGS)
            # # Renormalize reward
            # dfs[0]["value"] = dfs[0]["value"].apply(lambda x: normalize_rew(x, norm_rew))
            dfs[0]["value"] = dfs[0]["value"].cummax()
            new_dfs = []
            for df in dfs:
                df["seed"] = seed
                df["alpha"] = names[exp_name]
                # take only step_limit steps
                df = df.drop(df[df.step > step_limit].index)
                df["step"] = df["step"] / 100_000
                new_dfs.append(df)
            df_list_rew.append(new_dfs[0])
            df_list_violation.append(new_dfs[1])
            df_list_safety.append(new_dfs[2])

    charts = []
    for i, df in enumerate([df_list_rew, df_list_violation, df_list_safety]):
        df_main = pd.concat(df)
        if row == 0 and i == 0:
            title = "Avg. Return"
        elif row == 0 and i == 1:
            title = "Acc. Violation"
        elif row == 0 and i == 2:
            title = "Policy Safety"
        else:
            title = ""
        line = alt.Chart(df_main,title=title).mark_line().encode(
            x=alt.X("step",
                    scale=alt.Scale(domain=(0, step_limit/100_000)),
                    title=None,
                    axis=alt.Axis(
                        format='~s',
                        labels=False,
                        grid=False)),
            y=alt.Y("mean(value)",
                    axis=alt.Axis(
                        title=DOMAIN_ABBR[domain_name] if i == 0 else None,
                        format='~s',
                        grid=False)),
            color=alt.Color("alpha",
                            legend=altair.Legend(title=None),
                            sort = [names[exp_name] for exp_name in exp_names],
                            # scale=alt.Scale(domain=[names[exp_name] for exp_name in exp_names], range=["gray", "blue", "green"]),
                            scale=alt.Scale(scheme="category10")
                            )
        ).properties(
            width=200,
            height=100
        )
        band = alt.Chart(df_main).mark_errorband(extent='ci', opacity=0.2).encode(
            x=alt.X("step"),
            y=alt.Y("value", title=""),
            color=alt.Color("alpha", sort = [names[exp_name] for exp_name in exp_names],)
        )
        c = alt.layer(band, line)#.resolve_scale(color="independent")
        charts.append(c)

    chart_row = altair.hconcat(*charts)

    return chart_row
    # chart_row.show()
    # fig_path = os.path.join(domain, f"svg")
    # c.save(fig_path)

def violation_return(type="Q1perf", title="Perfect Sensors"):
    data = {}
    for domain in table_settings:
        data[DOMAIN_ABBR[domain]] = {}

    for domain in table_settings:
        folder = FOLDERS[domain]
        for i_alpha, exp in enumerate(table_settings[domain][type]):
            data[DOMAIN_ABBR[domain]][exp] = {}
            exp_folder = os.path.join(folder, exp)
            for seed in SEEDS:
                data[DOMAIN_ABBR[domain]][exp][seed] = {}
                exp_folder_path = os.path.join(exp_folder, seed)
                r, v = load_step_values(exp_folder_path, TAGS[0:2], 600_000)
                # Normalize reward and violation
                n_r = normalize_rew(r, NORMS_REW[domain])
                n_v = normalize_vio(v, NORMS_VIO[domain])
                data[DOMAIN_ABBR[domain]][exp][seed]["return"] = n_r
                data[DOMAIN_ABBR[domain]][exp][seed]["violation"] = n_v

    df = pd.DataFrame.from_records(
        [
            (domain[:-1], domain, ALPHA_NAMES_LEARNING_CURVES[exp], seed, keys["violation"], keys["return"])
            for domain, exps in data.items()
            for exp, seeds in exps.items()
            for seed, keys in seeds.items()
        ],
        columns=['symbol', 'domain', 'agent', 'seed', 'violation', 'return']
    )

    c = alt.Chart(df, title=title).mark_point().encode(
        x=alt.X("violation",
                axis=alt.Axis(
                    format='.1',
                    grid=False)),
        y=alt.Y("return",
                axis=alt.Axis(
                    format='.1',
                    grid=False)),
        color=alt.Color("agent",
                        scale=alt.Scale(scheme='category10')),
        shape=alt.Shape('symbol', scale=alt.Scale(range=['circle', 'square', 'triangle-right']))
    ).properties(
        width=200,
        height=200
    )

    return c

def violation_return_combined():
    cs = []
    for type, title in [("Q1perf", "Perfect Sensors"), ("Q1noisy", "Noisy Sensors")]:
        c = violation_return(type, title)
        cs.append(c)
    cc = altair.hconcat(*cs).configure_legend(
        orient="top",
        direction='horizontal',
    ).configure_title(
        anchor="middle"
    )
    # cc.show()
    svg_path = os.path.join(dir_path, "../..", "experiments5")
    fig_path = os.path.join(svg_path, f"violation_return.svg")
    cc.save(fig_path)
    return


def curves_combined(type="perf"):
    graph_settings = {
        "goal_finding1": {
            "perf": ["PPO", "VSRLperf", "PLPGperf4"],
            "noisy": ["PPO", "VSRLthres", "epsVSRLthres0.005", "PLPGnoisy"]
        },
        "goal_finding2": {
            "perf": ["PPO", "VSRLperf", "PLPGperf"],
            "noisy": ["PPO", "VSRLthres", "epsVSRLthres0.01", "PLPGnoisy"]
        },
        "pacman1":{
            "perf": ["PPO", "VSRLperf", "PLPGperf3"],
            "noisy": ["PPO", "VSRLthres", "epsVSRLthres0.05", "PLPGnoisy4"]
        },
        "pacman2": {
            "perf": ["PPO", "VSRLperf", "PLPGperf4"],
            "noisy": ["PPO", "VSRLthres", "epsVSRLthres0.005", "PLPGnoisy3"]
        },
        "carracing1": {
            "perf": ["PPO", "VSRLperf", "PLPGperf2"],
            "noisy": ["PPO", "VSRLthres", "epsVSRLthres0.5", "PLPGnoisy3"]
        },
        "carracing2": {
            "perf": ["PPO", "VSRLperf", "PLPGperf"],
            "noisy": ["PPO", "VSRLthres", "epsVSRLthres0.5", "PLPGnoisy2"]
        },
    }

    chart_rows = []
    for row, domain in enumerate(graph_settings):
        print(domain)
        chart_row = curves(domain,
                           exp_names=graph_settings[domain][type],
                           names=ALPHA_NAMES_LEARNING_CURVES,
                           step_limit=600_000,
                           row=row
                           )
        chart_rows.append(chart_row)

    c = altair.vconcat(*chart_rows, title=""
                       ).configure_axisY(
        titleAngle=0,
        titleAlign="left",
        titleY=50,
        titleX=-60,
    ).configure_view(stroke=None
    ).configure_legend(
        orient="top",
        direction='horizontal',
    ).configure_title(
        anchor="middle"
    )
    # c.show()
    svg_path = os.path.join(dir_path, "../..", "experiments5")
    fig_path = os.path.join(svg_path, f"{type}.svg")
    c.save(fig_path)

table_settings = {
        "goal_finding1": {
            "eps": ["VSRLthres", "epsVSRLthres0.005", "epsVSRLthres0.01", "epsVSRLthres0.05", "epsVSRLthres0.1", "epsVSRLthres0.2", "epsVSRLthres0.5", "PPO"],
            "perf": ["PLPG_STperf", "PLPGperf2", "PLPGperf4", "PLPGperf", "PLPGperf3"],
            "noisy": ["PLPG_STnoisy", "PLPGnoisy3", "PLPGnoisy4", "PLPGnoisy", "PLPGnoisy2"],
            "Q1": ["PPO", "VSRLperf", "PLPGperf4", "VSRLthres", "epsVSRLthres0.005", "PLPGnoisy"],
            "Q1perf": ["PPO", "VSRLperf", "PLPGperf4"],
            "Q1noisy": ["PPO", "VSRLthres", "epsVSRLthres0.005", "PLPGnoisy"],
            "LTST": ["PLPG_LTperf", "PLPG_STperf", "PLPGperf4", "PLPG_LTnoisy", "PLPG_STnoisy", "PLPGnoisy"]
        },
        "goal_finding2": {
            "eps": ["VSRLthres", "epsVSRLthres0.005", "epsVSRLthres0.01", "epsVSRLthres0.05", "epsVSRLthres0.1", "epsVSRLthres0.2", "epsVSRLthres0.5", "PPO"],
            "perf": ["PLPG_STperf", "PLPGperf2", "PLPGperf4", "PLPGperf", "PLPGperf3"],
            "noisy": ["PLPG_STnoisy", "PLPGnoisy3", "PLPGnoisy4", "PLPGnoisy", "PLPGnoisy2"],
            "Q1": ["PPO", "VSRLperf", "PLPGperf", "VSRLthres", "epsVSRLthres0.01", "PLPGnoisy"],
            "Q1perf": ["PPO", "VSRLperf", "PLPGperf"],
            "Q1noisy": ["PPO", "VSRLthres", "epsVSRLthres0.01", "PLPGnoisy"],
            "LTST": ["PLPG_LTperf", "PLPG_STperf", "PLPGperf", "PLPG_LTnoisy", "PLPG_STnoisy", "PLPGnoisy"]
        },
        "pacman1": {
            "eps": ["VSRLthres", "epsVSRLthres0.005", "epsVSRLthres0.01", "epsVSRLthres0.05", "epsVSRLthres0.1", "epsVSRLthres0.2", "epsVSRLthres0.5", "PPO"],
            "perf": ["PLPG_STperf", "PLPGperf3", "PLPGperf5", "PLPGperf", "PLPGperf2"],
            "noisy": ["PLPG_STnoisy", "PLPGnoisy3", "PLPGnoisy4", "PLPGnoisy", "PLPGnoisy2"],
            "Q1": ["PPO", "VSRLperf", "PLPGperf3", "VSRLthres", "epsVSRLthres0.05", "PLPGnoisy4"],
            "Q1perf":  ["PPO", "VSRLperf", "PLPGperf3"],
            "Q1noisy": ["PPO", "VSRLthres", "epsVSRLthres0.05", "PLPGnoisy4"],
            "LTST": ["PLPG_LTperf", "PLPG_STperf", "PLPGperf3", "PLPG_LTnoisy", "PLPG_STnoisy", "PLPGnoisy4"]
        },
        "pacman2": {
            "eps": ["VSRLthres", "epsVSRLthres0.005", "epsVSRLthres0.01", "epsVSRLthres0.05", "epsVSRLthres0.1", "epsVSRLthres0.2", "epsVSRLthres0.5", "PPO"],
            "perf": ["PLPG_STperf", "PLPGperf3", "PLPGperf4", "PLPGperf", "PLPGperf2"],
            "noisy": ["PLPG_STnoisy", "PLPGnoisy3", "PLPGnoisy4", "PLPGnoisy", "PLPGnoisy2"],
            "Q1": ["PPO", "VSRLperf", "PLPGperf4", "VSRLthres", "epsVSRLthres0.005", "PLPGnoisy3"],
            "Q1perf": ["PPO", "VSRLperf", "PLPGperf4"],
            "Q1noisy": ["PPO", "VSRLthres", "epsVSRLthres0.005", "PLPGnoisy3"],
            "LTST": ["PLPG_LTperf", "PLPG_STperf", "PLPGperf4", "PLPG_LTnoisy", "PLPG_STnoisy", "PLPGnoisy3"]
        },
        "carracing1": {
            "eps": ["VSRLthres", "epsVSRLthres0.005", "epsVSRLthres0.01", "epsVSRLthres0.05", "epsVSRLthres0.1", "epsVSRLthres0.2", "epsVSRLthres0.5", "PPO"],
            "perf": ["PLPG_STperf", "PLPGperf", "PLPGperf2", "PLPGperf3", "PLPGperf4"],
            "noisy": ["PLPG_STnoisy", "PLPGnoisy", "PLPGnoisy2", "PLPGnoisy3", "PLPGnoisy4"],
            "Q1": ["PPO", "VSRLperf", "PLPGperf2", "VSRLthres", "epsVSRLthres0.5", "PLPGnoisy3"],
            "Q1perf": ["PPO", "VSRLperf", "PLPGperf2"],
            "Q1noisy": ["PPO", "VSRLthres", "epsVSRLthres0.5", "PLPGnoisy3"],
            "LTST": ["PLPG_LTperf", "PLPG_STperf", "PLPGperf2", "PLPG_LTnoisy", "PLPG_STnoisy", "PLPGnoisy3"]
        },
        "carracing2": {
            "eps": ["VSRLthres", "epsVSRLthres0.005", "epsVSRLthres0.01", "epsVSRLthres0.05", "epsVSRLthres0.1", "epsVSRLthres0.2", "epsVSRLthres0.5", "PPO"],
            "perf": ["PLPG_STperf", "PLPGperf", "PLPGperf2", "PLPGperf3", "PLPGperf4"],
            "noisy": ["PLPG_STnoisy", "PLPGnoisy", "PLPGnoisy2", "PLPGnoisy3", "PLPGnoisy4"],
            "Q1": ["PPO", "VSRLperf", "PLPGperf", "VSRLthres", "epsVSRLthres0.5", "PLPGnoisy2"],
            "Q1perf": ["PPO", "VSRLperf", "PLPGperf"],
            "Q1noisy": ["PPO", "VSRLthres", "epsVSRLthres0.5", "PLPGnoisy2"],
            "LTST": ["PLPG_LTperf", "PLPG_STperf", "PLPGperf", "PLPG_LTnoisy", "PLPG_STnoisy", "PLPGnoisy2"]
        },
    }

def create_tables(type):
    for row, domain in enumerate(table_settings):
        extract_values(domain, table_settings[domain][type])

def draw_Q5(type="perf",ALPHA_OR_EPS=None, symbol="ɑ"):
    data_rew, data_vio = {}, {}
    for domain in table_settings:
        data_rew[DOMAIN_ABBR[domain]]={
            "seed1": {}, "seed2": {}, "seed3": {}, "seed4": {}, "seed5": {}
        }
        data_vio[DOMAIN_ABBR[domain]]={
            "seed1": {}, "seed2": {}, "seed3": {}, "seed4": {}, "seed5": {}
        }
    for domain in table_settings:
        folder = FOLDERS[domain]
        for i_alpha, exp in enumerate(table_settings[domain][type]):
            exp_folder = os.path.join(folder, exp)
            for seed in SEEDS:
                exp_folder_path = os.path.join(exp_folder, seed)
                r, v = load_step_values(exp_folder_path, TAGS[0:2], 600_000)
                # Normalize reward and violation
                n_r = normalize_rew(r, NORMS_REW[domain])
                n_v = normalize_vio(v, NORMS_VIO[domain])
                data_rew[DOMAIN_ABBR[domain]][seed][i_alpha] = n_r
                data_vio[DOMAIN_ABBR[domain]][seed][i_alpha] = n_v

    df_rew = pd.DataFrame.from_records(
        [
            (domain, seed, alpha, value)
            for domain, seeds in data_rew.items()
            for seed, alphas in seeds.items()
            for alpha, value in alphas.items()
        ],
        columns=['domain', 'seed', 'alpha', 'value']
    )
    df_vio = pd.DataFrame.from_records(
        [
            (domain, seed, alpha, value)
            for domain, seeds in data_vio.items()
            for seed, alphas in seeds.items()
            for alpha, value in alphas.items()
        ],
        columns=['domain', 'seed', 'alpha', 'value']
    )
    if len(ALPHA_OR_EPS) == 8:
        axis_labels = (
            f"datum.label == 0 ? {ALPHA_OR_EPS[0]} : \
            datum.label == 1 ? {ALPHA_OR_EPS[1]}: \
            datum.label == 2 ? {ALPHA_OR_EPS[2]}: \
            datum.label == 3 ? {ALPHA_OR_EPS[3]}: \
            datum.label == 4 ? {ALPHA_OR_EPS[4]}: \
            datum.label == 5 ? {ALPHA_OR_EPS[5]}: \
            datum.label == 6 ? {ALPHA_OR_EPS[6]}: \
             {ALPHA_OR_EPS[7]}"
        )
    else:
        axis_labels = (
            f"datum.label == 0 ? {ALPHA_OR_EPS[0]} : \
            datum.label == 1 ? {ALPHA_OR_EPS[1]}: \
            datum.label == 2 ? {ALPHA_OR_EPS[2]}: \
            datum.label == 3 ? {ALPHA_OR_EPS[3]}: \
             {ALPHA_OR_EPS[4]}"
        )
    cs = []
    for df in [df_rew, df_vio]:
        line = alt.Chart(df, title="").mark_line().encode(
            x=alt.X("alpha:Q",
                    title=symbol,
                    axis=alt.Axis(
                        format='.1',
                        labelExpr=axis_labels,
                        # values=ALPHA_OR_EPS,
                        grid=False)),
            y=alt.Y("mean(value)",
                    axis=alt.Axis(title="", grid=False)),
            color=alt.Color("domain",
                            legend=altair.Legend(title=None),
                            scale=alt.Scale(scheme="category10"))
        ).properties(
            width=150 if symbol=="ɑ" else 250,
            height=150
        )
        band = alt.Chart(df).mark_errorband(extent='ci', opacity=0.2).encode(
            x=alt.X("alpha"),
            y=alt.Y("value", title=""),
            color=alt.Color("domain")
        )
        c = alt.layer(band, line)
        cs.append(c)
    return cs

def draw_Q5_together():
    ALPHA = [0, 0.1, 0.5, 1, 5]
    EPS = [0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    eps_rew, eps_vio= draw_Q5("eps", EPS, symbol="ε")
    alpha_rew, alpha_vio = draw_Q5("noisy", ALPHA, symbol="ɑ")

    violation_c = altair.hconcat(eps_vio, alpha_vio, title="Violation")
    return_c = altair.hconcat(eps_rew, alpha_rew, title="Return")
    c = altair.vconcat(violation_c, return_c
    ).configure_view(stroke=None
    ).configure_legend(
        orient="top",
        direction='horizontal',
    ).configure_title(
        anchor="middle"
    )

    svg_path = os.path.join(dir_path, "../..", "experiments5")
    fig_path = os.path.join(svg_path, f"Q5.svg")
    c.save(fig_path)
    # c.show()



# create_tables(type="Q1")
# create_tables(type="LTST")


ALPHA = [0, 0.1, 0.5, 1, 5]
EPS = [0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
# draw_Q5("perf", ALPHA, symbol="ɑ")
# draw_Q5("noisy", ALPHA, symbol="ɑ")
# draw_Q5("eps", EPS, symbol="ε")
# draw_Q5_together()
violation_return_combined()


# curves_combined("perf")
# curves_combined("noisy")

