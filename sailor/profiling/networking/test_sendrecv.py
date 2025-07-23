import torch
import torch.distributed as dist
import pandas as pd
import os
import logging
import time
import argparse
import numpy as np
import random

def setup(rank: int, master_ip: str, master_port: str, socket_nthreads: int, world_size: int):
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = master_port
    #os.environ['NCCL_SOCKET_NTHREADS'] = str(socket_nthreads)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    set_deterministic()

    # initialize process group
    setup(args.rank, args.master_ip, args.master_port, args.socket_nthreads, args.world_size)

    local_rank = args.rank % args.gpus_per_node
    torch.cuda.set_device(local_rank)

    # Tensor size of ~1 GB (250 million elements * 4 bytes/element)
    tensor_size_bytes = args.data_size
    tensor_size_floats = int(args.data_size) // 4
    tensor_size_gb = tensor_size_bytes / 1e9
    tensor = torch.randn(tensor_size_floats, dtype=torch.float32).to(local_rank)

    idx = 0
    warmup_iters = 3
    p2p_times = []
    bw_GB_s = []
    bw_Gbps = []
    half_world = args.world_size // 2

    dist.barrier(device_ids=[local_rank])
    torch.cuda.synchronize()

    for idx in range(args.num_runs):
        if (idx == warmup_iters):
            dist.barrier(device_ids=[local_rank])
            torch.cuda.synchronize()
            start_time = time.time()

        if args.rank < half_world:
            dist.send(tensor=tensor, dst=args.rank + half_world)
        # Worker node receives data
        else:
            dist.recv(tensor=tensor, src=args.rank - half_world)

        torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()
    end_time = time.time()

    total_size_gb = half_world * (args.num_runs - warmup_iters) * tensor_size_gb
    print(tensor_size_gb, total_size_gb)
    duration = end_time - start_time
    bandwidth = total_size_gb/duration

    cleanup()

    if args.rank==0:

        results_df = pd.DataFrame(data={
            "bytes": [tensor_size_bytes],
            "bw_GB_s": [bandwidth],
        })

        print(f"Bandwidth from Master to Worker: {bandwidth} GB/s")

        output_file = f"data_{args.world_size}.csv"
        if (args.add_header):
            results_df.to_csv(output_file, index=False)
        else:
            results_df.to_csv(output_file, index=False, header=False, mode='a') # append to file without header




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--rank', type=int, default=0, help='rank of the process')
    parser.add_argument('--master_ip', type=str, default='10.128.15.210', help='internal IP of VM running rank 0')
    parser.add_argument('--master_port', type=str, default='29500', help='port of VM running rank 0')
    parser.add_argument('--socket_nthreads', type=int, default=1, help='value for NCCL_SOCKET_NTHREADS')
    parser.add_argument('--gpus_per_node', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--world_size', type=int, default=1, help='World size')

    parser.add_argument('--data_size', type=int, default=1024, help='size of data in bytes to perform allReduce operation on')
    parser.add_argument('--num_runs', type=int, default=10, help='number of all reduce operations')
    parser.add_argument('--output_file', type=str, default=None, help='csv file for results')
    parser.add_argument('--gpu', type=str, default='V100', help='type of GPU')
    parser.add_argument('--add_header', action='store_true', help='add headr if csv file non existent yet')

    args = parser.parse_args()
    main(args)
