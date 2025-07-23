import pandas as pd
import matplotlib.pyplot as plt
import sys
from itertools import accumulate

red = 'tab:red'
blue = 'navy'
teal = 'teal'
black = 'k'
orange = 'orange'
lblue = 'lightsteelblue'
olive = 'tab:olive'
cyan = 'tab:cyan'
rblue = 'royalblue'
brown = 'brown'
green = 'green'


def get_data(csv_file):
    df = pd.read_csv(csv_file)
    ts = list(df['Times(sec)'])
    thr = list(df['Throughput'])
    cost = list(df['Cost'])

    ts = list(accumulate(ts))
    num_available_gpus = list(df['Num_available_gpus'])
    num_used_gpus = list(df['Num_used_gpus'])
    search_time_list = list(df['Search Time(sec)'])

    iters_done_ts = [x*y for x, y in zip(thr, ts)]
    total_iters = sum(iters_done_ts)
    total_cost = sum(cost)

    search_time = round(sum(search_time_list), 1)
    iters_per_dollar = round(total_iters/total_cost, 1)
    iters_per_dollar_ts = [x/y if y > 0 else 0 for x, y in zip(iters_done_ts, cost)]

    print(iters_done_ts, total_iters, ts[-1])

    return {
        'timestamps': ts,
        'throughput': thr,
        'cost': cost,
        'num_available_gpus': num_available_gpus,
        'num_used_gpus': num_used_gpus,
        'search_time': search_time,
        'iters_per_dollar': iters_per_dollar,
        'iters_per_dollar_ts': iters_per_dollar_ts,
        'total_throughput': round(total_iters/ts[-1], 5)
    }


def get_data_baseline_model(baseline, model):
    results_dir = sys.argv[2]
    input_file = f"{results_dir}/{baseline}_{model}.csv"
    data = get_data(input_file)
    return data


def plot_step_throughput(baselines):

    fig, ax1 = plt.subplots(figsize=(10, 5))
    label_font_size = 20

    for key, data in baselines.items():
        print(key, data['total_throughput'])

    for baseline, data in baselines.items():
        ax1.step(data['timestamps'], data['throughput'],
                 label=f"{baseline}, ST: {data['search_time']} sec, Thr: {data['total_throughput']} iters/sec",
                 linewidth=3)
    ax2 = ax1.twinx()

    ax2.step(data['timestamps'], data['num_available_gpus'],
             label='Num Available GPUs', linestyle='dashed', linewidth=2, color='black')

    plt.yticks(fontsize=label_font_size)
    plt.xticks(fontsize=label_font_size)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    plt.legend(handles1+handles2, labels1+labels2, loc='lower right',
               fontsize=label_font_size, ncols=3, bbox_to_anchor=(1.1, 1.05))

    ax1.set_ylabel('Throughput (iters/sec)', fontsize=label_font_size)
    ax1.set_xlabel('Time (sec)', fontsize=label_font_size)
    ax2.set_ylabel('Number of GPUs', fontsize=label_font_size)

    ax1.tick_params(axis='y', which='major', labelsize=label_font_size)
    ax1.tick_params(axis='x', which='major', labelsize=16)
    ax2.tick_params(axis='y', which='major', labelsize=label_font_size)

    # ax1.set_yticks(fontsize=label_font_size)
    # ax2.set_yticks(fontsize=label_font_size)

    plt.savefig(sys.argv[3], bbox_inches="tight", dpi=500, pad_inches=0.1)


def plot_step_iters_cost(baselines):

    fig, ax1 = plt.subplots(figsize=(10, 5))
    label_font_size = 16

    for key, data in baselines.items():
        print(key, data['iters_per_dollar'])

    for baseline, data in baselines.items():
        ax1.step(data['timestamps'], data['iters_per_dollar_ts'],
                 label=f"{baseline}, {data['search_time']} sec", linewidth=3)
    ax2 = ax1.twinx()

    ax2.step(data['timestamps'], data['num_available_gpus'],
             label='Num Available GPUs', linestyle='dashed', linewidth=2, color='black')

    plt.yticks(fontsize=label_font_size)
    plt.xticks(fontsize=label_font_size)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # plt.legend(handles1+handles2, labels1+labels2, loc='lower right', fontsize=12, ncols=3, bbox_to_anchor=(1.1, 1.05))

    ax1.set_ylabel('Value (iters/dollar)', fontsize=label_font_size)
    ax1.set_xlabel('Time (sec)', fontsize=label_font_size)
    ax2.set_ylabel('Number of GPUs', fontsize=label_font_size)

    plt.savefig(sys.argv[3], bbox_inches="tight", dpi=500, pad_inches=0.1)


if __name__ == "__main__":
    baselines = ["Varuna", "AMP", "Oobleck", "SAILOR"]
    model = sys.argv[1]
    data = {}

    for baseline in baselines:
        data[baseline] = get_data_baseline_model(baseline, model)

    plot_step_throughput(data)
    # plot_step_iters_cost(data)
