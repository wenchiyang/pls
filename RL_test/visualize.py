from collections import defaultdict

import altair as alt
import pandas as pd
import numpy as np
import re
import os

#TODO: refer to this function in the main file
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

def parse(logger_file):
    f = open(logger_file, 'r')
    line = f.readline()
    datapoints = []
    while line:
        if '--- Pacman-v0 Log' in line:
            d = defaultdict(float)
            d['log_number'] = [int(s) for s in line.split() if s.isdigit()][0]
            f.readline()
            d['overall_steps'] = [int(s) for s in f.readline().split() if s.isdigit()][0]
            d['episodes'] = [int(s) for s in f.readline().split() if s.isdigit()][0]
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            d['mean_episode_length'], d['deviation_episode_length'] = re.findall(r"[-+]?\d*\.\d+|\d+", f.readline())
            d['mean_episode_reward'], d['deviation_episode_reward'] = re.findall(r"[-+]?\d*\.\d+|\d+", f.readline())
            datapoints.append(d)
        line = f.readline()

    return datapoints

def parse_raw_dpl(logger_file):
    """
    Parses dpl logger files
    """
    float_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
    tensor_pattern = re.compile(r"\(\[(.*)\]")
    step_pattern = re.compile(r"----  Step (\d+)  ")
    timestamp_pos = 22 # Skip the timestamp at the start of the line
    f = open(logger_file, 'r')
    line = f.readline()
    datapoints = []
    # count = 0
    while line:
        if '---  Step ' in line:
            d = defaultdict(float)
            d['n_steps'] = int(step_pattern.findall(line, pos=timestamp_pos)[0])
            l = f.readline()
            while l and "tensor" not in l: l = f.readline()
            d['shield_probs'] = tensor_pattern.findall(l, pos=timestamp_pos)
            l = f.readline()
            while l and "tensor" not in l: l = f.readline()
            d['base_probs'] = tensor_pattern.findall(l, pos=timestamp_pos)
            l = f.readline()
            while l and "tensor" not in l: l = f.readline()
            d['ghost_probs'] = tensor_pattern.findall(l, pos=timestamp_pos)
            l = f.readline()
            while l and "tensor" not in l: l = f.readline()
            d['pacman_probs'] = tensor_pattern.findall(l, pos=timestamp_pos)
            l = f.readline()
            while l and "Reward" not in l: l = f.readline()
            if not l: break
            d['reward'] = float(float_pattern.findall(l, pos=timestamp_pos)[0])
            d['done'] = "True" in f.readline()
            datapoints.append(d)
            # if count > 34755:
            #     print(count)
            # count += 1
        line = f.readline()
    return datapoints

def parse_raw(logger_file):
    """
    Parses no dpl logger files
    """
    float_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
    timestamp_pos = 22 # Skip the timestamp at the start of the line
    f = open(logger_file, 'r')
    line = f.readline()
    datapoints = []
    while line:
        if '---  Step ' in line:
            d = defaultdict(float)
            d['n_steps'] = [int(s) for s in line.split() if s.isdigit()][0]
            d['reward'] = float(float_pattern.findall(f.readline(), pos=timestamp_pos)[0])
            d['done'] = "True" in f.readline()
            datapoints.append(d)
        line = f.readline()
    return datapoints

def create_data_series(logger_file, dpl, window_size, window_speed):
    if dpl:
        datapoints = parse_raw_dpl(logger_file)
        datapoints_list = [
            [
                d['n_steps'],
                d['shield_probs'],
                d['base_probs'],
                d['ghost_probs'],
                d['pacman_probs'],
                d['reward'],
                d['done']
            ] for d in datapoints]
    else:
        datapoints = parse_raw(logger_file)
        datapoints_list = [
            [
                d['n_steps'],
                d['reward'],
                d['done']
            ] for d in datapoints]

    datapoints_list = list(zip(*datapoints_list))
    avg_rs, avg_ls, avg_last_rs = moving_average(datapoints_list[-2], datapoints_list[-1], window_size, window_speed)

    data_series = pd.DataFrame({
        'n_steps': [i for i in range(len(avg_rs))],
        'reward': avg_rs,
        'length': avg_ls,
        'success': avg_last_rs
    })
    data_series = data_series.melt('n_steps')
    return data_series

def moving_average(rewards, dones, window_size, window_speed):
    start = window_speed
    end = len(rewards)

    avg_rs = []
    avg_ls = []
    avg_last_rs = []
    while start < end:
        pos = 0 if start - window_size < 0 else start - window_size
        rs = rewards[pos:start]
        ds = dones[pos:start]
        eps_rs, eps_ls, eps_last_rs = _episodes_length_rewards(rs, ds)
        avg_r = sum(eps_rs) / len(eps_rs)
        avg_l = sum(eps_ls) / len(eps_ls)
        avg_last_r = sum(eps_last_rs) / len(eps_last_rs)
        avg_rs.append(avg_r)
        avg_ls.append(avg_l)
        avg_last_rs.append(avg_last_r)
        start += window_speed
    return avg_rs, avg_ls, avg_last_rs


def make_chart(data_series):
    combined_data_series = pd.concat(data_series, keys=['pg','pg_dpl','pg_dpl_detect'], names=['setting'])
    combined_data_series = combined_data_series.reset_index(0)
    combined_data_series['series'] = combined_data_series['setting'] + " " + combined_data_series['variable']

    selector = alt.selection_single(empty='all', fields=['series'])
    base = alt.Chart(combined_data_series).properties(
        width=500,
        height=250
    ).add_selection(selector)

    timeseries = base.mark_line(
        opacity=0.5
    ).encode(
        x=alt.X('n_steps:O',
                axis=alt.Axis(title='Steps (x100)', tickMinStep=30)),
        y=alt.Y('value:O',
                sort="descending"),
        color=alt.Color('series:O')
    ).transform_filter(
        selector
    )

    timeseries.show()

def make_chart2(data_series, keys=['pg','pg_dpl','pg_dpl_detect'], window_speed=1000):
    combined_data_series = pd.concat(data_series, keys=keys, names=['setting'])
    combined_data_series = combined_data_series.reset_index(0)
    combined_data_series['series'] = combined_data_series['setting'] + " " + combined_data_series['variable']

    # Data is prepared, now make a chart
    selection_exp = alt.selection_multi(fields=['setting', 'variable'], empty='none')
    color_exp = alt.condition(selection_exp,
                      alt.Color('series:N', legend=None),
                      alt.value('lightgray'))

    timeseries = alt.Chart(combined_data_series).properties(
        width=500,
        height=250
    ).mark_line(
        opacity=0.5
    ).encode(
        x=alt.X('n_steps:Q',
                axis=alt.Axis(title=f'Steps (x{window_speed})', tickMinStep=1)),
        y=alt.Y('value:Q',
                sort="ascending",
                axis=alt.Axis(title='value', tickMinStep=0.1),
                ),
        color=color_exp
    ).add_selection(
        selection_exp
    )

    legend = alt.Chart(combined_data_series).mark_rect().encode(
        x=alt.X('setting:N', axis=alt.Axis(orient='bottom')),
        y='variable',
        color=color_exp
    ).add_selection(
        selection_exp
    )

    chart = timeseries | legend

    return chart



if __name__ == "__main__":
    folderpath = os.path.join(os.path.dirname(__file__), "20210702_13:08")
    file_pg = "pg_grid2x3_raw"
    file_pg_dpl = "pg_dpl_nodetect_grid2x2_raw"
    file_pg_dpl_detect = "pg_dpl_detect_grid2x3_raw"

    logger_files = [
        file_pg,
        # file_pg_dpl,
        file_pg_dpl_detect
    ]

    # create dataframes
    for logger_file in logger_files:
        log_path = os.path.join(folderpath, f"{logger_file}.log")
        dpl = "dpl" in logger_file
        d = create_data_series(log_path, dpl=dpl, window_size=2000, window_speed=1000)
        pkl_path = os.path.join(folderpath, f"{logger_file}.pkl")
        d.to_pickle(pkl_path)


    # load dataframes
    data_frames = []
    for logger_file in logger_files:
        pkl_path = os.path.join(folderpath, f"{logger_file}.pkl")
        d = pd.read_pickle(pkl_path)
        data_frames.append(d)


    chart = make_chart2(data_frames, keys=['pg','pg_dpl_detect'], window_speed=1000)
    chart.show()


