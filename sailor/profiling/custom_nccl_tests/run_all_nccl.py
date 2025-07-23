import os
import argparse

def main(args):
    for i in range(16, 31):
        num_bytes = pow(2,i)
        cmd = f"mpirun --mca btl_tcp_if_include {args.interface} -hostfile hostfile -np {args.world_size} ./test_nccl {i}"
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, required=True, help='World size')
    parser.add_argument('--interface', type=str, required=True, help='Interface for MPI')
    args = parser.parse_args()
    main(args)

