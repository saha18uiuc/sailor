""" Measure bandwidth between 2 GPUs in different geographic locations when seding
a tensor of size 1 GB from a sender (rank 0) to a reveiver (rank 1). We use PyTorch's
MPI framwork on top of the nccl backend. The script allows to customize the
NCCL_SOCKET_NTHREADS variable."""

import torch
import torch.distributed as dist
import pandas as pd
import os
import logging
import time
import argparse

def setup(rank: int, master_ip: str, master_port: str, socket_nthreads: int, min_channels: int):
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = master_port
    os.environ['NCCL_SOCKET_NTHREADS'] = str(socket_nthreads)
    # os.environ['NCCL_MIN_NCHANNELS'] = str(min_channels)
    dist.init_process_group('nccl', rank=rank, world_size=2)

def cleanup():
    dist.destroy_process_group()

def main(args):
    
    # initialize process group
    setup(args.rank, args.master_ip, args.master_port, args.socket_nthreads, args.min_channels)

    # Tensor size of ~1 GB (250 million elements * 4 bytes/element)
    tensor_size = args.data_size // 4
    tensor = torch.randn(tensor_size, dtype=torch.float32).to(0)
    bytes = tensor.element_size() * tensor.nelement()
    dist.barrier()
    idx = 0

    p2p_times = []
    bw_MB_s = []
    bw_Mbps = []

    try:
        for idx in range(args.num_runs):
            # Master node sends data
            torch.cuda.synchronize()
            start_time = time.time()
            if args.rank == 0:
                print(f"Start sending {idx}")
                dist.send(tensor=tensor, dst=1)
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time

                bw = (bytes) / elapsed_time  # Calculate bandwidth
                p2p_times.append(elapsed_time)
                bw_MB_s.append(bw / 1e6)
                bw_Mbps.append(bw * 8 / 1e6)
                print(f"Bandwidth from Master to Worker: {bw / 1e6} MB/s, {bw * 8 / 1e6} Mbps")

            # Worker node receives data
            elif args.rank == 1:
                dist.recv(tensor=tensor, src=0)
                print(f"Received {idx}")


            time.sleep(1)  # Wait for 1 second before next iteration
    except KeyboardInterrupt:
        pass

    dist.barrier()

    # rank 0 to output results to file
    if (args.rank == 0 and args.output_file is not None):
        results_df = pd.DataFrame(data={
            'p2p_times': p2p_times,
            'num_bytes': [bytes] * args.num_runs,
            'bw_MB_s': bw_MB_s,
            'bw_Mbps': bw_Mbps,
            'socket_nthreads': [args.socket_nthreads] * args.num_runs,
            # 'min_channels': [args.min_channels] * args.num_runs,
            'gpu': [args.gpu] * args.num_runs,
        })

        output_file = f"send_recv/results/diff_data_size/{args.gpu}/{args.output_file}.csv"
        if (args.add_header):
            results_df.to_csv(output_file, index=False)
        else:
            results_df.to_csv(output_file, index=False, header=False, mode='a') # append to file without header

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--rank', type=int, default=0, help='rank of the process')
    parser.add_argument('--master_ip', type=str, default='10.128.15.210', help='internal IP of VM running rank 0')
    parser.add_argument('--master_port', type=str, default='29500', help='port of VM running rank 0')
    parser.add_argument('--socket_nthreads', type=int, default=16, help='value for NCCL_SOCKET_NTHREADS')
    parser.add_argument('--data_size', type=int, default=128, help='size of data in MB to perform allReduce operation on')
    parser.add_argument('--min_channels', type=int, default=2, help='value for NCCL_MIN_NCHANELS')
    parser.add_argument('--num_runs', type=int, default=20, help='number of all reduce operations')
    parser.add_argument('--output_file', type=str, default=None, help='csv file for results')
    parser.add_argument('--gpu', type=str, default='V100', help='type of GPU')
    parser.add_argument('--add_header', action='store_true', help='add headr if csv file non existent yet')

    args = parser.parse_args()
    main(args)
