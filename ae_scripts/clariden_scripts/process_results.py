import numpy as np
import sys
import glob
import argparse
import json

def get_time(input_dir, name):
    with open(f"{input_dir}/sailor_log_worker_rank_0", 'r') as f:
        lines = f.readlines()

    durs = []
    for line in lines:
        if name in line:
            dur = float(line.split(" ")[-2])
            durs.append(dur)

    durs = durs[3:]
    avg = round(np.average(durs), 5)
    print(f"Average {name} is {avg}")
    return avg

def get_mem(input_dir):

    max_total = 0
    res_dict = {}
    for file in glob.glob(f"{input_dir}/memory_log_worker_rank_*"):
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                final = lines[-1]
            tokens = final.split(" ")
            total = int(tokens[3])
            reserved = int(tokens[6])
            allocated = int(tokens[9])
            max_total = max(max_total, total)
        except Exception:
            pass
    print(f"Max mem is {max_total}")
    return max_total


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Result parser',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--main_dir', type=str, required=True, help='main_dir')
    parser.add_argument('--nnodes', type=int, required=True)
    parser.add_argument('--gpus_per_node', type=int, required=True)
    parser.add_argument('--gbs', type=int, required=True)
    parser.add_argument('--dp', type=int, required=True)
    parser.add_argument('--pp', type=int, required=True)
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--mbs', type=int, required=True)

    parser.add_argument('--target_dir', type=str, required=True, help='target dir (containing configs to be updated)')

    args=parser.parse_args()

    suffix = f"N{args.nnodes}/N{args.nnodes}_M{args.gpus_per_node}_G{args.gbs}_D{args.dp}_P{args.pp}_T{args.tp}_M{args.mbs}"

    data_dir_mem = f"{args.main_dir}/memory/{suffix}"
    max_total = get_mem(data_dir_mem)

    data_dir_time = f"{args.main_dir}/time/{suffix}"
    it_time = get_time(data_dir_time, "Iteration")

    if args.nnodes==1:
        target_file = f"{args.target_dir}/N{args.nnodes}/plan_config_N{args.nnodes}_D{args.dp}_M{args.gpus_per_node}_G{args.gbs}.json"
    else:
        target_file = f"{args.target_dir}/N{args.nnodes}/plan_config_N{args.nnodes}_D{args.dp}.json"
    with open(target_file, 'r') as f:
        config = json.load(f)

    config["real"] = it_time
    config["max_mem"] = max_total

    with open(target_file, 'w') as f:
        json.dump(config, f, indent=2)
