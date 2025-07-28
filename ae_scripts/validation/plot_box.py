import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from itertools import accumulate
import os

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

def get_data(input_file):
    df = pd.read_csv(input_file)
    real = list(df['real'])
    sim = list(df['estimated'])
    percentages = [abs(x-y)*100/x for x,y in zip(real, sim) if x>0]
    return percentages

def plot(baselines_sim, results_path):
    fig, ax = plt.subplots(figsize=(20, 5))
    label_font_size = 28
    baselines = list(baselines_sim.keys())
    data = [baselines_sim[x] for x in baselines]
    x = np.arange(1,len(baselines)+1)

    plt.boxplot(data, patch_artist=True, manage_ticks=True)
    if plot_time:
        img_name = f"{results_path}/box_time.png"
    else:
        img_name = f"{results_path}/box_mem.png"
    print(f"Save at: {img_name}")

    ax.set_xlabel('Baselines', fontsize=label_font_size)
    ax.set_xticks(x, baselines)
    ax.set_ylabel('Absolute diff (%)', fontsize=label_font_size)

    plt.yticks(fontsize=label_font_size)
    plt.xticks(fontsize=28)

    plt.savefig(img_name, bbox_inches="tight", dpi=500, pad_inches=0.1)



if __name__ == "__main__":
    results_path = sys.argv[1]
    baselines = ['Piper', 'Varuna',  'Metis', 'FlashFlex', 'SAILOR'] # 'Aceso',
    baselines_sim = {}
    plot_time=(sys.argv[2]=="time")
    sz = None
    for baseline in baselines:
        print(f"------------------- BASELINE: {baseline}")
        if plot_time:
            baseline_path = f"{results_path}/{baseline}-time.csv"
        else:
            baseline_path = f"{results_path}/{baseline}-mem.csv"
        if os.path.exists(baseline_path):
            baseline_percentages = get_data(baseline_path)
            sz = len(baseline_percentages)
        else:
            baseline_percentages = [0]*sz
        baselines_sim[baseline] = baseline_percentages
        print(baseline, np.average(baseline_percentages))
        #labels, real, baselines_sim[baseline] = get_data_indexes(labels, real, baselines_sim[baseline], [3,5,8,11,-1])
    plot(baselines_sim, results_path)