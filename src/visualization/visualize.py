from collections import defaultdict

import altair as alt
import pandas as pd
import numpy as np
import re
import os
import json
from pathlib import Path


# TODO: refer to this function in the main file
def _episodes_length_rewards(rewards, dones):
    """
    When dealing with array rewards and dones (as for VecEnv) the length
    and rewards are only computed on the first dimension.
    (i.e. the first sub-process.)
    """
    episode_rewards = []
    episode_lengths = []
    episode_last_rewards = []
    accum = 0.0
    length = 0
    for r, d in zip(rewards, dones):
        if not isinstance(d, bool):
            d = bool(d.flat[0])
            r = float(r.flat[0])
        if not d:
            accum += r
            length += 1
        else:
            accum += r
            length += 1
            episode_rewards.append(accum)
            episode_lengths.append(length)
            episode_last_rewards.append(r)
            accum = 0.0
            length = 0
    if length > 0:
        episode_rewards.append(accum)
        episode_lengths.append(length)
    return episode_rewards, episode_lengths, episode_last_rewards


def parse_raw_dpl_new(logger_file):
    """
    Parses dpl logger files
    """
    float_pattern = re.compile(r"([-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+|\d+)")
    tensor_pattern = re.compile(r"\(\[(.*)\]")
    step_pattern = re.compile(r"----  Step (\d+)  ")
    timestamp_pos = 22  # Skip the timestamp at the start of the line
    f = open(logger_file, "r")
    line = f.readline()
    datapoints = []
    # count = 0
    while line:
        try:
            if "---  Step " in line:
                d = defaultdict(float)
                d["n_steps"] = int(step_pattern.findall(line, pos=timestamp_pos)[0])
                f.readline()
                tensor = tensor_pattern.findall(f.readline(), pos=timestamp_pos)[0]
                d["shield_probs"] = [
                    float(prob) for prob in float_pattern.findall(tensor)
                ]
                tensor = tensor_pattern.findall(f.readline(), pos=timestamp_pos)[0]
                d["base_probs"] = [
                    float(prob) for prob in float_pattern.findall(tensor)
                ]
                tensor = tensor_pattern.findall(f.readline(), pos=timestamp_pos)[0]
                d["ghost_probs"] = [
                    float(prob) for prob in float_pattern.findall(tensor)
                ]
                tensor = tensor_pattern.findall(f.readline(), pos=timestamp_pos)[0]
                d["ghost_truth"] = [
                    float(prob) for prob in float_pattern.findall(tensor)
                ]
                tensor = tensor_pattern.findall(f.readline(), pos=timestamp_pos)[0]
                d["wall_probs"] = [
                    float(prob) for prob in float_pattern.findall(tensor)
                ]
                tensor = tensor_pattern.findall(f.readline(), pos=timestamp_pos)[0]
                d["wall_truth"] = [
                    float(prob) for prob in float_pattern.findall(tensor)
                ]
                # tensor = tensor_pattern.findall(f.readline(), pos=timestamp_pos)[0]
                # d['safe_current'] = [float(prob) for prob in float_pattern.findall(tensor)]
                tensor = tensor_pattern.findall(f.readline(), pos=timestamp_pos)[0]
                d["safe_next"] = [float(prob) for prob in float_pattern.findall(tensor)]
                d["reward"] = float(
                    float_pattern.findall(f.readline(), pos=timestamp_pos)[0]
                )
                d["done"] = "True" in f.readline()
                datapoints.append(d)
        except:
            break
        line = f.readline()

    keys = list(datapoints[0].keys())
    values = list(zip(*[list(d.values()) for d in datapoints]))
    dataseries_raw = dict(zip(keys, values))

    return dataseries_raw


def parse_raw(logger_file):
    """
    Parses no dpl logger files
    """
    float_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
    timestamp_pos = 22  # Skip the timestamp at the start of the line
    f = open(logger_file, "r")
    line = f.readline()
    datapoints = []
    while line:
        if "---  Step " in line:
            d = defaultdict(float)
            d["n_steps"] = [int(s) for s in line.split() if s.isdigit()][0]
            f.readline()
            d["reward"] = float(
                float_pattern.findall(f.readline(), pos=timestamp_pos)[0]
            )
            d["done"] = "True" in f.readline()
            datapoints.append(d)
        line = f.readline()
    return datapoints


def create_data_series(logger_file, window_size, window_speed):
    """For pure PG"""
    datapoints = parse_raw(logger_file)
    datapoints_list = [[d["n_steps"], d["reward"], d["done"]] for d in datapoints]
    datapoints_list = list(zip(*datapoints_list))
    avg_rs, avg_ls, avg_last_rs = moving_average_rewards(
        datapoints_list[-2], datapoints_list[-1], window_size, window_speed
    )
    data_series = pd.DataFrame(
        {
            "n_steps": [i for i in range(1, len(avg_rs) + 1)],
            "reward": avg_rs,
            "length": avg_ls,
            "safety": avg_last_rs,  # average last reward in the past {window_size} steps
        }
    )
    data_series = data_series.melt("n_steps", var_name="feature")
    return data_series


def create_data_series_dpl(logger_file, window_size, window_speed):
    """For dpl"""
    dataseries = parse_raw_dpl_new(logger_file)
    ########  make data series  #########
    avg_rs, avg_ls, avg_last_rs = moving_average_rewards(
        dataseries["reward"], dataseries["done"], window_size, window_speed
    )
    df_features = pd.DataFrame(
        {
            "n_steps": [i for i in range(1, len(avg_rs) + 1)],
            "reward": avg_rs,
            "length": avg_ls,
            "safety": avg_last_rs,  # average last reward in the past {window_size} steps
        }
    ).melt("n_steps", var_name="feature")

    ########  make base policy accuracy data series  #########
    avg_base_policy_err = moving_average_probs(
        dataseries["shield_probs"], dataseries["base_probs"], window_size, window_speed
    )
    avg_ghost_detect_err = moving_average_probs(
        dataseries["ghost_probs"], dataseries["ghost_truth"], window_size, window_speed
    )
    avg_wall_detect_err = moving_average_probs(
        dataseries["wall_probs"], dataseries["wall_truth"], window_size, window_speed
    )

    df_avg_prob_err = pd.DataFrame(
        {
            "n_steps": [i for i in range(1, len(avg_ghost_detect_err) + 1)],
            "base policy safety diff": avg_base_policy_err,
            "ghost detection error": avg_ghost_detect_err,
            # 'pacman detection error': avg_pacman_detect_err,
            # 'food detection error': avg_food_detect_err.
            "wall detection error": avg_wall_detect_err,
        }
    ).melt("n_steps", var_name="feature")

    return df_features, df_avg_prob_err


def moving_average_rewards(rewards, dones, window_size, window_speed):
    start = window_speed
    end = len(rewards)

    avg_rs = []
    avg_ls = []
    avg_last_rs = []
    while start <= end:
        pos = 0 if start - window_size < 0 else start - window_size
        rs = rewards[pos:start]
        ds = dones[pos:start]
        eps_rs, eps_ls, eps_last_rs = _episodes_length_rewards(rs, ds)
        avg_r = sum(eps_rs) / len(eps_rs)
        avg_l = sum(eps_ls) / len(eps_ls)
        if len(eps_last_rs) != 0:
            avg_last_r = sum(eps_last_rs) / len(eps_last_rs)
        else:
            avg_last_r = 0
        avg_rs.append(avg_r)
        avg_ls.append(avg_l)
        avg_last_rs.append(avg_last_r)
        start += window_speed
    return avg_rs, avg_ls, avg_last_rs


def moving_average_probs(shielded_probs, base_probs, window_size, window_speed):
    diff_probs = [
        sum(abs(np.array(s_prob) - np.array(b_prob)))
        for s_prob, b_prob in zip(shielded_probs, base_probs)
    ]

    start = window_speed
    end = len(shielded_probs)
    avg_prob_diffs = []
    while start <= end:
        pos = 0 if start - window_size < 0 else start - window_size
        diff_probs_window = diff_probs[pos:start]
        avg_prob_diff = sum(diff_probs_window) / len(diff_probs_window)
        avg_prob_diffs.append(avg_prob_diff)
        start += window_speed
    return avg_prob_diffs


def make_chart2(data_series, title, keys, window_speed=1000):
    msg = "dataseries and names must have equal length."
    assert len(data_series) == len(keys), msg
    combined_data_series = pd.concat(data_series, keys=keys, names=["setting"])
    combined_data_series = combined_data_series.reset_index(0)
    combined_data_series["series"] = (
        combined_data_series["setting"] + " " + combined_data_series["feature"]
    )

    # Data is prepared, now make a chart
    selection_exp = alt.selection_multi(fields=["setting", "feature"], empty="none")
    color_exp = alt.condition(
        selection_exp,
        alt.Color("series:N", legend=None, scale=alt.Scale(scheme="dark2")),
        alt.value("lightgray"),
    )

    timeseries = (
        alt.Chart(combined_data_series)
        .properties(title=title, width=500, height=250)
        .mark_line(opacity=0.5)
        .encode(
            x=alt.X(
                "n_steps:Q",
                axis=alt.Axis(title=f"Steps (x{window_speed})", tickMinStep=1),
            ),
            y=alt.Y(
                "value:Q",
                sort="ascending",
                axis=alt.Axis(title="value", tickMinStep=0.1),
            ),
            color=color_exp,
        )
        .add_selection(selection_exp)
        .interactive()
    )

    legend = (
        alt.Chart(combined_data_series)
        .mark_rect()
        .encode(
            x=alt.X("setting:N", axis=alt.Axis(orient="bottom")),
            y="feature",
            color=color_exp,
        )
        .add_selection(selection_exp)
    )

    chart = timeseries | legend

    return chart


def process(folder, config):
    window_size = config["visualize_settings"]["window_size"]
    window_speed = config["visualize_settings"]["window_speed"]

    logger_name = config["raw_logger"]

    dpl = "dpl" in logger_name or "detect" in logger_name or "shield" in logger_name
    pkl_path = os.path.join(folder, f"{logger_name}.pkl")
    pkl_err_path = os.path.join(folder, f"{logger_name}_prob_err.pkl")

    # process log files if pkl files dont exist
    if not os.path.isfile(pkl_path) or (dpl and not os.path.isfile(pkl_err_path)):
        log_path = os.path.join(folder, f"{logger_name}.log")
        if dpl:
            df_features, df_avg_prob_err = create_data_series_dpl(
                log_path, window_size=window_size, window_speed=window_speed
            )
            df_avg_prob_err.to_pickle(pkl_err_path)
        else:
            df_features = create_data_series(
                log_path, window_size=window_size, window_speed=window_speed
            )
        # store pkl file for later
        df_features.to_pickle(pkl_path)


def read_dfs(folder, config):
    process(folder, config)

    logger_name = config["raw_logger"]
    dpl = "dpl" in logger_name or "detect" in logger_name or "shield" in logger_name
    pkl_path = os.path.join(folder, f"{logger_name}.pkl")
    pkl_err_path = os.path.join(folder, f"{logger_name}_prob_err.pkl")

    # load pkl files
    df_features = pd.read_pickle(pkl_path)
    df_policy_err = None

    if dpl:
        df_policy_err = pd.read_pickle(pkl_err_path)

    return df_features, df_policy_err


def create_chart_shield(exp_folder):

    dfs_features = []
    dfs_prob_err = []

    for exp in ["pg", "pg_shield", "pg_shield_detect"]:
        folder = os.path.join(exp_folder, exp)
        config_file = os.path.join(folder, "config.json")
        with open(config_file) as json_data_file:
            config = json.load(json_data_file)

        df_features, df_policy_err = read_dfs(folder, config)
        dfs_features.append(df_features)
        if df_policy_err is not None:
            dfs_prob_err.append(df_policy_err)

    window_speed = config["visualize_settings"]["window_speed"]
    ######  make a charts  ###################
    #  make a chart for reward, length and safety during training
    feature_chart = make_chart2(
        dfs_features,
        title="Features over training time",
        keys=[
            "PG",
            "PG shield",
            "PG shield detect",
        ],
        window_speed=window_speed,
    )

    #  make a chart for prob_diff (base policy, shielded policy)
    prob_err_chart = make_chart2(
        dfs_prob_err,
        title="DPL features",
        keys=[
            "PG shield",
            "PG shield detect",
        ],
        window_speed=window_speed,
    )

    ######  save charts  ###################
    chart = feature_chart & prob_err_chart

    chart_path = os.path.join(exp_folder, "figs", "pg.json")
    Path(os.path.dirname(chart_path)).mkdir(parents=True, exist_ok=True)
    chart.save(chart_path)
    chart_path = os.path.join(exp_folder, "figs", "pg.html")
    chart.save(chart_path)
    # chart.show(chart_path)


def main():
    exps_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "experiments",
    )

    exps = [
        "grid2x2_1_ghost",
        # "grid3x3_1_ghost",
        # "grid5x5_1_ghost",
        # "grid5x5_3_ghosts",
        # "grid5x5_5_ghosts",
    ]
    for exp in exps:
        exp_folder = os.path.join(exps_folder, exp)
        create_chart_shield(exp_folder)


main()
