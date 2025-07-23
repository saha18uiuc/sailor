import os
import argparse
import json
from sailor.Planner.baselines.Aceso.aceso_cost_model import cost_model_init, predict_time_breakdown_beta, read_profiled_time
from sailor.Planner.baselines.Aceso.aceso_utils import *
from sailor.Planner.baselines.Aceso.aceso_search import search_init

parser = argparse.ArgumentParser()
parser.add_argument('--plan-path', type=str, required=True, help='Path to Aceso config')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--profiling-file-dir', type=str, required=True)
parser.add_argument('--training-config-path', type=str, required=True)
parser.add_argument('--cluster-config-path', type=str, required=True)
parser.add_argument('--gpu-type', type=str, choices=['A100-40', 'V100-16', 'RTX-3090', 'GH-96', 'RTX-2080', 'Titan-RTX'], required=True)
args = parser.parse_args()

with open(args.cluster_config_path, 'r') as f:
    cluster_config = json.load(f)[args.gpu_type]
with open(args.training_config_path, 'r') as f:
    training_config = json.load(f)

#### Hardware info ####
num_nodes = 1 # dummy input, will be overwritten by plan
gpus_per_node=cluster_config['gpus_per_node']
memory_limit=cluster_config['mem_per_gpu'] // (1024 * 1024) # in MB

if args.fp16:
    profiling_file_dir = os.path.join(args.profiling_file_dir, "profiled-data-fp16/")
else:
    profiling_file_dir = os.path.join(args.profiling_file_dir, "profiled-data-fp32/")

argString = f"""--micro-batch-size 1 2 4 8 \
--num-nodes {num_nodes} \
--num-gpus-per-node {gpus_per_node} \
--memory-limit {memory_limit} \
--profiled-time-path {profiling_file_dir} \
--initial-point {args.plan_path}"""
if args.fp16:
    argString += " --fp16"

search_init(argString, training_config)
aceso_args = parse_args(argString)
config, config_dict = read_config_from_json(aceso_args, return_config_dict=True)
max_time, max_memory = predict_time_breakdown_beta(config, print_time=True, print_memory=True)
max_time /= 1000
print("max_time,max_memory")
print(f"{max_time},{max_memory}")
