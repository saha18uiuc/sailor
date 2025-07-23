import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scenario(ax, scenario_name: str, bandwidth_df: pd.DataFrame):
    percentile_df = bandwidth_df.quantile(0.95).reset_index()
    ax.plot(percentile_df["index"], percentile_df[0.95], label=scenario_name)


def main(input_dir: str, output_dir: str):

    sns.set_theme(
        context="notebook",
        font_scale=1,
        style="darkgrid",
        rc={
            "lines.linewidth": 2,
            "axes.linewidth": 1,
            "axes.edgecolor":"black",
            "xtick.bottom": True,
            "ytick.left": True
        }
    )

    fig, ax = plt.subplots()
    fig.set_size_inches((8, 6))

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for file_name in json_files:
        with open(os.path.join(input_dir, file_name), "r") as f:
            json_data = json.load(f)

            results = {}

            for obj in json_data:
                num_streams = obj["start"]["test_start"]["num_streams"]
                num_streams_list = results.get(num_streams, [])
                bandwidth = obj["end"]["sum_sent"]["bits_per_second"] / 1e6
                num_streams_list.append(bandwidth)
                results[num_streams] = num_streams_list
            
            bandwidth_df = pd.DataFrame.from_dict(results)
            
            plot_scenario(ax, file_name[:-5], bandwidth_df)
    
    ax.set_ylabel("95-percentile Bandwidth [Mbps]")
    ax.set_xlabel("Number of streams")
    plt.legend()
    plt.title("95th percentile send bandwidth for V100 using Iperf3 across 3 runs")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/p95_bandwidth.png", format="png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default="results")
    parser.add_argument('--output_dir', type=str, default="results")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)