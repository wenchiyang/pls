from collections import defaultdict

import altair as alt
import pandas as pd
import numpy as np
import re

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

def create_data_series(logger_file_dpl):
    datapoints = parse(logger_file_dpl)
    datapoints_list = [
        [d['log_number'], d['overall_steps'], d['episodes'], d['mean_episode_length'], d['deviation_episode_length'],
         d['mean_episode_reward'], d['deviation_episode_reward']] for d in datapoints]
    datapoints_list = list(zip(*datapoints_list))

    data_series = pd.DataFrame({
        # 'time': range(n_datapoints),
        'log_number': datapoints_list[0],
        # 'overall_steps': datapoints_list[1],
        # 'episodes': datapoints_list[2],
        # 'mean_episode_length': datapoints_list[3],
        # 'deviation_episode_length': datapoints_list[4],
        'mean_episode_reward': datapoints_list[5],
        # 'deviation_episode_reward': datapoints_list[6]
    })
    data_series = data_series.melt('log_number')
    return data_series

def make_chart1(data_series):
    charts = []
    for d in data_series:
        selector = alt.selection_single(empty='all', fields=['series'])
        base = alt.Chart(d).properties(
            width=500,
            height=250
        ).add_selection(selector)

        timeseries = base.mark_line(
            opacity=0.5
        ).encode(
            x=alt.X('log_number:O',
                    # title="steps (x100)",
                    axis=alt.Axis(title='Steps (x100)', tickMinStep=30)),
            y=alt.Y('value:O',
                    sort="descending"),
            color=alt.Color('series:O')
        ).transform_filter(
            selector
        )
        charts.append(timeseries)
    chart = alt.vconcat(charts[0], charts[1])
    chart.show()


def make_chart2(data_series):

    combined_data_series = pd.concat(data_series, keys=['no_dpl','dpl'], names=['exp'])
    combined_data_series = combined_data_series.reset_index(0)
    combined_data_series['series'] = combined_data_series['exp'] + " " + combined_data_series['variable']

    # Data is prepared, now make a chart
    # selection_first_name = alt.selection_multi(fields=['exp_name'], empty='none')
    # selection_last_name = alt.selection_multi(fields=['variable'], empty='none')

    selector = alt.selection_single(empty='all', fields=['series'])
    base = alt.Chart(combined_data_series).properties(
        width=500,
        height=250
    ).add_selection(selector)

    timeseries = base.mark_line(
        opacity=0.5
    ).encode(
        x=alt.X('log_number:O',
                # title="steps (x100)",
                axis=alt.Axis(title='Steps (x100)', tickMinStep=30)),
        y=alt.Y('value:O',
                sort="descending"),
        color=alt.Color('series:O')
    ).transform_filter(
        selector
    )

    # TODO: prettier charts by multiple legends
    # timeseries = alt.Chart(dd).properties(
    #     width=500,
    #     height=250
    # ).mark_point().encode(
    #     x=alt.X('log_number:O',
    #             title="steps (x100)"),
    #     y=alt.Y('value:O',
    #             sort="descending"),
    #     color=alt.condition(selection_first_name & selection_last_name,
    #                         alt.Color('exp_name:O', legend=None),
    #                         alt.value('lightgray')),
    #     shape = alt.Shape('variable:O', legend=None)
    # ).transform_filter(
    #     alt.selection_interval(bind='scales')
    # )
    timeseries.show()



if __name__ == "__main__":
    # logger_file_no_dpl = "policy_gradient_good.log"
    # logger_file_dpl = "policy_gradient_dpl_good.log"
    logger_file_no_dpl = "policy_gradient_grid2x2.log"
    logger_file_dpl = "policy_gradient_dpl_grid2x2_1e-3.log"
    data_series_no_dpl = create_data_series(logger_file_no_dpl)
    data_series_dpl = create_data_series(logger_file_dpl)
    make_chart2([data_series_no_dpl, data_series_dpl])
