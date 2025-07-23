import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scenario(ax, scenario_name: str, df: pd.DataFrame):
    grouped = df.groupby(by=["socket_nthreads"]).quantile(0.95)
    x = np.arange(grouped.shape[0])
    ax.plot(x, grouped["MB/s"], label=scenario_name)

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

    # read in all csv files into a common pandas dataframe
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    
    dfs = {}
    for f in csv_files:
        df = pd.read_csv(os.path.join(input_dir, f))
        plot_scenario(ax, f[:-4], df)

    ax.set_ylabel("95-percentile Bandwidth [MB/s]")
    ax.set_xlabel("NCCL_SOCKET_NTHREADS")
    plt.legend()
    plt.title("95th percentile bandwidth for V100 using PyTroch distributed on top of NCCL backend")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/p95_bandwidth.png", format="png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, help="Input Directory containing csv files")
    parser.add_argument('--output_dir', type=str, help="Output Directory for plot")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)