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

    num_threads = [1,2,4]
    for i, n in enumerate(num_threads): # create a subplot for 1, 2 and 4 threads
        input_dir = f"results/send_recv/{n}_threads"
        csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
        
        for f in csv_files: # line plot each scenario
            df = pd.read_csv(os.path.join(input_dir, f), delimiter="\s+")
            x = np.arange(df.shape[0])
            y = df["busbw_ip"] * 1e3 * 8
            ax[i].plot(x, y, label=f[:-4])

        x_tick_labels = [round(s / 1e6, 2) for s in df["size"]]
        ax[i].set_xticks(np.arange(len(x_tick_labels)), x_tick_labels, fontsize="x-small")
        ax[i].set_xlabel("Data size [MB]")
        ax[i].set_ylabel("Bandwidth [Mbps]")
        ax[i].set_title(f"NCCL_SOCKET_NTHREADS = {n}")
        ax[i].legend()
    
    plt.tight_layout()
    plt.savefig(f"results/send_recv/bandwidth.png")
    plt.show()


def calculate_expected_bandwidth(df: pd.DataFrame) -> list[float]:
    """given the results of a Send/Recv measurement, calculates expected AllRed BW
    for the case of 4 GPUs, assumes that each row sends double amount of data of
    previous row
    returns the expected bandwidths in Mbps/s"""
    expected_bw = []
    for i in range(2, df.shape[0]):
        expected_latency = 2 * (4-1) * (df.loc[i-2, "time_ip"] / 1e6)
        bw = 2 * (4-1) * df.loc[i-2, "size"] / expected_latency
        expected_bw.append(bw * 8 / 1e6) # convert to Mpbs
    return expected_bw


def plot_all_reduce():
    fig, ax = plt.subplots(3, 1, figsize=(12,9))

    num_threads = [1,2,4]
    for i, n in enumerate(num_threads): # create a subplot for 1, 2 and 4 threads
        allred_input_dir = f"results/all_reduce/{n}_threads"
        sendrecv_input_dir = f"results/send_recv/{n}_threads"
        allred_csv_files = [f for f in os.listdir(allred_input_dir) if f.endswith(".csv")]
        sendrecv_csv_files = [f for f in os.listdir(sendrecv_input_dir) if f.endswith(".csv")]
        
        for f in allred_csv_files: # line plot each AllReduce scenario
            allred_df = pd.read_csv(os.path.join(allred_input_dir, f), delimiter="\s+")
            x = np.arange(allred_df.shape[0])
            y = allred_df["busbw_ip"] * 1e3 * 8
            ax[i].plot(x, y, label=f[:-4])

        for f in sendrecv_csv_files: # line plot expected AllReduce bandwidth for each sceanrio
            sendrecv_df = pd.read_csv(os.path.join(sendrecv_input_dir, f), delimiter="\s+")
            expected_bw = calculate_expected_bandwidth(sendrecv_df)
            x = np.arange(sendrecv_df.shape[0] - 2) + 2 # shift by 2
            ax[i].plot(x, expected_bw, label=f"expected_bw_{f[:-4]}")

        x_tick_labels = [round(s / 1e6,2) for s in allred_df["size"]]
        ax[i].set_xticks(np.arange(len(x_tick_labels)), x_tick_labels, fontsize="x-small")
        ax[i].set_xlabel("Data size [MB]")
        ax[i].set_ylabel("Bandwidth [Mbps]")
        ax[i].set_title(f"NCCL_SOCKET_NTHREADS = {n}")
        ax[i].legend(fontsize="small", bbox_to_anchor=(1.05,1))
    
    fig.suptitle("NCCL-test AllReduce measured and expected bus bandwidth")
    plt.tight_layout()
    plt.savefig(f"results/all_reduce/bandwidth.png")
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
