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
    "FlashFlex": "#984ea3",  # Purple
    "Galvatron": "#999999",  # Gray
    "Aceso": "#e41a1c",  # Red
    "NnScaler": "#dede00",  # Yellow
    "DTFM": "#17becf",  # Cyan
    "Atlas": "#fccde5",  # Light Pink
    "SAILOR": "#bc80bd",  # Lavender
    "Real": "#f781bf"
}

hatches = {
    "Piper": "/",   # Diagonal lines
    "AMP": "\\",  # Opposite diagonal
    "Varuna": "|",   # Vertical lines
    "Metis": "-",   # Horizontal lines
    "FlashFlex": "+",   # Grid (cross)
    "Galvatron": "x",   # Crosshatch
    "Aceso": "o",   # Small circles
    "NnScaler": "O",   # Large circles
    "DTFM": ".",   # Dots
    "Atlas": "*",   # Stars
    "SAILOR": " ",   # No hatch (empty)
    "Real": "H",   # Hexagons
}


def get_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    thr = [d['throughput'] for d in data]
    cost = [d["cost_per_iteration"] for d in data]

    return {
        'throughput': thr,
        'cost': cost,
    }


def get_data_baseline_model(baseline, results_dir):
    input_file = f"{results_dir}/{baseline}_{model}.json"
    data = get_data(input_file)
    return data



def plot_metric(baselines, target_total, destination):
    fig, ax1 = plt.subplots(figsize=(10, 3))
    label_font_size = 23
    width = 0.45

    baselines_list = list(baselines.keys())
    x = np.arange(len(baselines))

    throughput = []
    dollars_per_iter = []
    for i, baseline in enumerate(baselines_list):
        data = baselines[baseline]
        throughput.append(data["throughput"][0])
        dollars_per_iter.append(data["cost"][0])

    if target_total == 'total_throughput':
        bar_color="navy"
    else:
        bar_color="lightgray"

    ax2 = ax1.twinx()
    if target_total == 'total_throughput':
        plt.rcParams.update({'lines.markersize': 18})
        ax1.bar(x, throughput, width=width, color=bar_color, label='Throughput')
        ax2.scatter(x, dollars_per_iter, color="grey", marker='*', label='Cost')
        ax1.set_ylabel('Throughput \n (iters/sec)', fontsize=label_font_size)
        ax2.set_ylabel('USD/iter', fontsize=label_font_size)
        ax2.axhline(y=1.2, color='red', linewidth=2, linestyle='dashed')
        plt.text(x=3.5, y=1.22, s="max usd/iter", color='red', fontsize=14, fontweight='bold')
        plt.ylim(0,1.3)
    elif target_total == 'dollars_per_iter':
        plt.rcParams.update({'lines.markersize': 18})
        ax1.bar(x, throughput, width=width, color=bar_color, label='Throughput')
        ax2.scatter(x, dollars_per_iter, color="black", marker='*', label='Cost')
        ax1.set_ylabel('Throughput \n (iters/sec)', fontsize=label_font_size)
        ax2.set_ylabel('USD/iter', fontsize=label_font_size)
        ax1.axhline(y=0.2, color='red', linewidth=2, linestyle='dashed')
        plt.text(x=-0.25, y=1.7, s="min throughput", color='red', fontsize=14, fontweight='bold')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    handles = handles1 + handles2
    labels = labels1 + labels2
    plt.legend(handles, labels, loc='upper left', fontsize=15, ncols=1)

    baselines = [x if 'SAILOR' not in x else x.replace('SAILOR', 'Sailor') for x in baselines]
    baselines = [x if x != 'Atlas' else 'Atlas \n+ Exh Search \n+ Orbit Sim' for x in baselines]

    plt.xticks(x, baselines)
    ax1.tick_params(axis='y', which='major', labelsize=22)
    ax2.tick_params(axis='y', which='major', labelsize=22)

    ax1.set_xlabel('Baselines', fontsize=label_font_size)
    ax1.tick_params(axis='x', which='major', labelsize=14)

    print(f"Save at: {destination}")
    plt.savefig(destination, bbox_inches="tight", dpi=500, pad_inches=0.1)


if __name__ == "__main__":
    baselines = ['Galvatron', 'AMP', 'FlashFlex', 'Metis', 'DTFM',  'SAILOR']
    results_dir = sys.argv[1]
    model = sys.argv[2]
    keyword = sys.argv[3]
    destination = sys.argv[4]
    data = {}

    for baseline in baselines:
        print(f"------------ {baseline}")
        baseline_res = get_data_baseline_model(baseline, results_dir)
        data[baseline] = baseline_res
        print(f"Data is {baseline_res}")

    plot_metric(data, keyword, destination)