import os
import argparse

def main(args):
    for i in range(26, 31):
        num_bytes = pow(2,i)
        print(f"-------------- Run with {num_bytes} bytes")
        cmd = f"python test_sendrecv.py --rank {args.rank} --master_ip {args.master_ip} --gpus_per_node {args.gpus_per_node} --world_size {args.world_size} --data_size {num_bytes} --num_runs 50"
        if i==16:
            cmd += " --add_header"
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--rank', type=int, default=0, help='rank of the process')
    parser.add_argument('--master_ip', type=str, default='10.128.15.210', help='internal IP of VM running rank 0')
    parser.add_argument('--master_port', type=str, default='29500', help='port of VM running rank 0')

    parser.add_argument('--gpus_per_node', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--world_size', type=int, default=1, help='World size')

    args = parser.parse_args()
    main(args)
