import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scenario(ax, scenario_name: str, df: pd.DataFrame):
    grouped = df[["bw_Mbps", "socket_nthreads"]].groupby(by=["socket_nthreads"]).quantile(0.95)
    x = np.arange(grouped.shape[0])
    ax.plot(x, grouped["bw_Mbps"], label=scenario_name)

def main(result_dir: str, gpu: str):
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
    csv_files = [f for f in os.listdir(result_dir) if f.endswith(".csv")]
    
    dfs = {}
    tensor_size = 0
    world_size = 0
    for f in csv_files:
        df = pd.read_csv(os.path.join(result_dir, f))
        tensor_size = df["num_bytes"][0]
        world_size = df["world_size"][0]
        plot_scenario(ax, f[:-4], df)

    ax.set_xticks(np.arange(16))
    ax.set_xticklabels([str(i) for i in range(1, 17)])
    ax.set_ylabel("95p Bandwidth [Mbps]")
    ax.set_xlabel("NCCL_SOCKET_NTHREADS")
    plt.legend()
    plt.title(
        f"95p Bandwidth doing AllReduce on tensor of {tensor_size / 1e9} GB with 4 {gpu} GPUs\nusing PyTroch distributed on top of NCCL backend",
        fontdict={ "weight": "bold", "size": "medium" }
    )
    plt.tight_layout()
    plt.savefig(f"{result_dir}/p95_bandwidth.png", format="png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_dir', type=str, help="Directory containing csv files")
    parser.add_argument('--gpu', type=str, help="GPU type")

    args = parser.parse_args()
    main(args.result_dir, args.gpu)