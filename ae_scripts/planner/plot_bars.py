import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
from itertools import accumulate

colors = {
    "Piper": "#377eb8",  # Blue
    "AMP": "#ff7f00",  # Orange
    "Varuna": "#4daf4a",  # Green
    "Metis": "#a65628",  # Brown
    "FlashFlex": "#dede00",  # Purple
    "Galvatron": "#999999",  # Gray
    "Aceso": "#e41a1c",  # Red
    "DTFM": "#17becf",  # Cyan
    "SAILOR": "#984ea3",  # Lavender
}

hatches = {
    "Piper": "/",   # Diagonal lines
    "AMP": "\\",  # Opposite diagonal
    "Varuna": "|",   # Vertical lines
    "Metis": "-",   # Horizontal lines
    "FlashFlex": "+",   # Grid (cross)
    "Galvatron": "x",   # Crosshatch
    "Aceso": "o",   # Small circles
    "DTFM": ".",   # Dots
    "Atlas": "*",   # Stars
    "SAILOR": " ",   # No hatch (empty)

}

def get_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    ts = [d['duration'] for d in data]
    thr = [d['throughput'] for d in data]
    cost = [d["cost_per_iteration"] for d in data]
    oom_plans = [d['oom_plans'] for d in data]
    num_available_gpus = list(d['num_gpus'] for d in data)
    search_time_list = list(d['search_time'] for d in data)

    ts_acc = list(accumulate(ts))
    iters_done_ts = [x*y for x, y in zip(thr, ts)]
    total_iters = sum(iters_done_ts)
    cost_ts = [x*y for x,y in zip(iters_done_ts, cost)]
    total_cost = sum(cost_ts)

    search_time = round(sum(search_time_list), 3)
    iters_per_dollar = round(total_iters/total_cost, 3) if total_cost != 0.0 else 0.0
    dollars_per_iter = 1/iters_per_dollar if iters_per_dollar>0 else 0.0

    iters_per_dollar_ts = [1/y if y > 0 else 0 for y in cost_ts]
    all_oom_plans = int(sum(oom_plans))

    total_throughput = total_iters/ts_acc[-1]

    #print(iters_done_ts, total_iters, ts_acc[-1], value, total_value)
    num_available_gpus_list = []
    for case in num_available_gpus:
        all_gpus = 0
        if isinstance(case, dict):
            for _, gpu_count in case.items():
                all_gpus += gpu_count
            num_available_gpus_list.append(all_gpus)
        else:
            num_available_gpus_list.append(case)

    return {
        'timestamps': ts_acc,
        'throughput': thr,
        'cost': cost,
        'num_available_gpus': num_available_gpus_list,
        'search_time': search_time,
        'search_time_list': search_time_list,
        'iters_per_dollar': iters_per_dollar,
        'dollars_per_iter': dollars_per_iter,
        'iters_per_dollar_ts': iters_per_dollar_ts,
        'dollars_per_iter_ts': cost,
        'total_throughput': round(total_throughput, 5),
        'total_cost': round(total_cost, 1),
        'all_oom_plans': all_oom_plans,
        'oom_plans': oom_plans
    }


def get_data_baseline_model(model,baseline, results_dir):
    input_file = f"{results_dir}/{baseline}_{model}.json"
    data = get_data(input_file)
    return data


def plot_config(baselines, idx_list, setup, destination):
    #fig, ax1 = plt.subplots(figsize=(15, 9))
    fig, ax1 = plt.subplots(figsize=(25, 8))
    label_font_size = 40
    if setup=='homogeneous':
        width = 0.1
    elif setup=='heterogeneous' or setup=='heterogeneous-imbalanced':
        width = 0.14
    else:
        width = 0.25

    baselines_list = list(baselines.keys())
    x = np.arange(len(idx_list))

    num_machines = [32,80,128]
    if setup in ['homogeneous', 'geo']:
        num_machines_labels = num_machines
    else:
        if setup=='heterogeneous-imbalanced':
            num_machines_labels = [f"{x} A100 + {3*x} V100" for x in num_machines]
        else:
            num_machines_labels = [f"{x} A100 + {x} V100" for x in num_machines]

    all_bars = []
    max_height=0
    for i, baseline in enumerate(baselines_list):
        data = baselines[baseline]
        throughput = [data["throughput"][j] for j in idx_list]
        st_list = [round(data["search_time_list"][j],2) for j in idx_list]
        cost_list = [round(data["dollars_per_iter_ts"][j],2) for j in idx_list]
        oom_list = [data["oom_plans"][j] for j in idx_list]
        #print(x+i*width, throughput)
        print(baseline, f"Search Time: {st_list}, OOM plans: {oom_list}")
        bars = ax1.bar(x+i*width, throughput, width=width, color=colors[baseline], hatch=hatches[baseline])
        for bar in bars:
            max_height = max(bar.get_height(), max_height)
        all_bars.append(bars)

    ymin, ymax = ax1.get_ylim()
    offset = (ymax - ymin) * 0.03
    ax1.set_ylim(ymin, max_height+5*offset)

    for i, baseline in enumerate(baselines_list):
        data = baselines[baseline]
        throughput = [data["throughput"][j] for j in idx_list]
        cost_list = [round(data["dollars_per_iter_ts"][j],2) for j in idx_list]
        oom_list = [data["oom_plans"][j] for j in idx_list]
        bars = all_bars[i]
        for j,rect in enumerate(bars):
            if (setup!='homogeneous') and (throughput[j] != 0.0):
                height = rect.get_height()
                t = plt.text(rect.get_x() + rect.get_width() / 2.0, height+0.75 * offset, f"${cost_list[j]}", ha='center', va='bottom', fontsize=30)
                if 'heterogeneous' in setup:
                    plt.text(rect.get_x() + rect.get_width() / 2.0, height+2.5*offset, f"{oom_list[j]}", ha='center', va='bottom', fontsize=22, fontweight='bold')

            if throughput[j]==0.0:
                t = plt.text(rect.get_x() + rect.get_width() / 2.0, 0.0, f"X", ha='center', va='bottom', fontsize=24, color=colors[baseline])
                plt.text(rect.get_x() + rect.get_width() / 2.0, 0.005, f"{oom_list[j]}", ha='center', va='bottom', fontsize=24, fontweight='bold')

    x_offset = x+width*len(baselines_list)/2-width/2
    plt.xticks(x_offset, num_machines_labels)


    if setup=='geo':
        ax1.set_xlabel('Number of A100 per zone', fontsize=label_font_size)
    else:
        ax1.set_xlabel('Number of GPUs', fontsize=label_font_size)

    ax1.tick_params(axis='x', which='major', labelsize=label_font_size-3)

    ax1.set_ylabel('Throughput\n(iters/sec)', fontsize=label_font_size)
    ax1.tick_params(axis='y', which='major', labelsize=label_font_size)

    baselines_list = [x if 'SAILOR' not in x else x.replace('SAILOR', 'Sailor') for x in baselines_list]

    if baselines=='homoegenous':
        plt.legend(baselines_list, fontsize=35, ncols=3, loc='upper left')
    else:
        plt.legend(baselines_list, fontsize=38, ncols=2, loc='upper left')


    plt.yticks(fontsize=label_font_size)
    plt.xticks(fontsize=label_font_size)

    print(f"Save at: {destination}")

    plt.savefig(destination, bbox_inches="tight", dpi=500, pad_inches=0.1)


if __name__ == "__main__":
    results_dir = sys.argv[1]
    model = sys.argv[2]
    setup = sys.argv[3]
    destination = sys.argv[4]

    if setup=='homogeneous':
        baselines = ['Varuna', 'AMP', 'Piper', 'Galvatron', 'Aceso', 'FlashFlex', 'Metis', 'DTFM', 'SAILOR']
    elif setup=='heterogeneous' or setup=='heterogeneous-imbalanced':
        baselines = ['AMP', 'FlashFlex', 'Metis', 'SAILOR']
    else:
        baselines = ['DTFM', 'SAILOR']

    data = {}

    for baseline in baselines:
        print(f"------------ {baseline}")
        baseline_res = get_data_baseline_model(model, baseline, results_dir)
        data[baseline] = baseline_res
        print(f"Data is {baseline_res}")

    idx_list = list(range(3))
    plot_config(data, idx_list, setup, destination)

    sailor_thr =[data['SAILOR']["throughput"][j] for j in idx_list]
    sailor_cost =[data['SAILOR']["dollars_per_iter_ts"][j] for j in idx_list]

    for baseline in baselines:
        baseline_thr = [data[baseline]["throughput"][j] for j in idx_list]
        baseline_cost = [data[baseline]["dollars_per_iter_ts"][j] for j in idx_list]
        speedups = []
        for x,y in zip(sailor_thr, baseline_thr):
            if y>0:
                speedups.append(x/y)
        print(baseline, speedups, round(np.average(speedups),2), baseline_cost)
        costs = []
        for x,y in zip(sailor_cost, baseline_cost):
            costs.append(y/x)
        print(costs, round(np.average(costs), 2))
        print(f"--------------------------------------------------")