""" Script for plotting the bandwidth measurements and the expected bandwidth in AllReduce
when running PyTorch on top of NCCL. Assumes that the respective data is in
pytorch/all_reduce/results/diff_data_size/V100 and
pytorch/send_recv/results/diff_data_size/V100 """

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_send_recv():
    fig, ax = plt.subplots(3, 1, figsize=(9,9))
    
    input_dir = f"send_recv/results/diff_data_size/V100"
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    for f in csv_files:
        df = pd.read_csv(os.path.join(input_dir, f))

        # iterate over all threads and plot them on their respective subplot
        num_threads = [1,2,4]
        for i, n in enumerate(num_threads):
            threads_df = df[df["socket_nthreads"] == n].drop(columns=["gpu"])
            #threads_df = threads_df[threads_df["num_bytes"] >= 65536000] # drop noisy measures
            grouped_df = threads_df.groupby(by=["num_bytes"]).quantile(0.95).reset_index()
            x = np.arange(grouped_df.shape[0])
            x_tick_labels = [round(l / 1e6, 2) for l in grouped_df["num_bytes"]]
            ax[i].plot(x, grouped_df["bw_Mbps"], label=f[:-4])
            ax[i].set_xticks(np.arange(len(x_tick_labels)), x_tick_labels, fontsize="x-small")
            ax[i].set_xlabel("Data size [MB]")
            ax[i].set_ylabel("Bandwidth [Mbps]")
            ax[i].set_title(f"NCCL_SOCKET_NTHREADS = {n}")
            ax[i].legend(fontsize="small")
    
    fig.suptitle("Send/Recv bandwidth measurements using PyTorch on top of NCCL")
    plt.tight_layout()
    plt.savefig(f"{input_dir}/bandwidth.png")
    plt.show()


def calculate_expected_bandwidth(df: pd.DataFrame) -> list[float]:
    """given the results of a Send/Recv measurement, calculates expected AllRed BW
    for the case of 4 GPUs, assumes that each row sends double amount of data of
    previous row
    returns the expected bandwidths in Mbps/s"""
    expected_bw = []
    for i in range(2, df.shape[0]):
        expected_latency = 2 * (4-1) * df.loc[i-2, "p2p_times"]
        bw = 2 * (4-1) * df.loc[i-2, "num_bytes"] / expected_latency
        expected_bw.append(bw * 8 / 1e6) # convert to Mpbs
    return expected_bw


def plot_all_reduce():
    fig, ax = plt.subplots(3, 1, figsize=(12,9))
    
    allred_input_dir = f"all_reduce/results/diff_data_size/V100"
    sendrecv_input_dir = f"send_recv/results/diff_data_size/V100"
    allred_csv_files = [f for f in os.listdir(allred_input_dir) if f.endswith(".csv")]
    sendrecv_csv_files = [f for f in os.listdir(sendrecv_input_dir) if f.endswith(".csv")]
    
    for f in allred_csv_files:
        df = pd.read_csv(os.path.join(allred_input_dir, f))

        # iterate over all threads and plot them on their respective subplot
        num_threads = [1,2,4]
        for i, n in enumerate(num_threads):
            threads_df = df[df["socket_nthreads"] == n].drop(columns=["gpu"])
            #threads_df = threads_df[threads_df["num_bytes"] >= 65536000] # drop noisy measures
            grouped_df = threads_df.groupby(by=["num_bytes"]).quantile(0.95).reset_index()
            x = np.arange(grouped_df.shape[0])
            x_tick_labels = [round(l / 1e6, 2) for l in grouped_df["num_bytes"]]
            ax[i].plot(x, grouped_df["bw_Mbps"], label=f[:-4])
            ax[i].set_xticks(np.arange(len(x_tick_labels)), x_tick_labels, fontsize="x-small")
            ax[i].set_xlabel("Data size [MB]")

    for f in sendrecv_csv_files:
        df = pd.read_csv(os.path.join(sendrecv_input_dir, f))

        # iterate over all threads and plot them on their respective subplot
        num_threads = [1,2,4]
        for i, n in enumerate(num_threads):
            threads_df = df[df["socket_nthreads"] == n].drop(columns=["gpu"])
            #threads_df = threads_df[threads_df["num_bytes"] >= 65536000] # drop noisy measures
            grouped_df = threads_df.groupby(by=["num_bytes"]).quantile(0.95).reset_index()
            x = np.arange(grouped_df.shape[0] - 2) + 2 # shift by 2
            expected_bw = calculate_expected_bandwidth(grouped_df)
            ax[i].plot(x, expected_bw, label=f"expected_bw_{f[:-4]}")
            ax[i].set_xlabel("Data size [MB]")
            ax[i].set_ylabel("Bandwidth [Mbps]")
            ax[i].set_title(f"NCCL_SOCKET_NTHREADS = {n}")
            ax[i].legend(fontsize="small", bbox_to_anchor=(1.05,1))
    
    fig.suptitle("PyTorch AllReduce measured and expected bandwidth")
    plt.tight_layout()
    plt.savefig(f"{allred_input_dir}/bandwidth.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--send_recv', action='store_true')
    parser.add_argument('--all_reduce', action='store_true')

    args = parser.parse_args()
    if args.send_recv:
        plot_send_recv()
    if args.all_reduce:
        plot_all_reduce()
