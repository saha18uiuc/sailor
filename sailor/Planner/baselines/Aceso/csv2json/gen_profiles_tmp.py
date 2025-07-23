import re
import sys
import math
from collections import defaultdict
import csv
import json
import argparse
import os
sys.path.append("../")

parser = argparse.ArgumentParser()

parser.add_argument('--gpu-type-list', type=str, required=True, help='GPU type')
parser.add_argument('--fp16', action='store_true', help='Convert fp16 profiled data')
parser.add_argument('--profile-dir', type=str, required=True, help='directory of profile data')

args = parser.parse_args()


microsec_per_unit = 1000000
gpu_type_list = str(args.gpu_type_list).split(',')
print(gpu_type_list)
if args.fp16:
    output_name = "profiles_tmp_aceso_fp16.json"
    dir_suffix = "profiled-data-fp16"
else: 
    output_name = "profiles_tmp_aceso.json"
    dir_suffix = "profiled-data-fp32"

mb_size_list = [1, 2, 4, 8]
tp_size_list = [1, 2, 4, 8]
algo_list = [0, 1]
model_list = ["GPT-Neo-2.7", "OPT-30", "OPT-350"]
llm_info = {}
num_layers = {"GPT-Neo-2.7": 32, 
              "OPT-30": 48, 
              "OPT-350": 24, }

for model_name in model_list:
    llm_info[model_name] = {}
    for gpu_type in gpu_type_list: 
        profile_dir = os.path.join(args.profile_dir, gpu_type, dir_suffix)
        llm_info[model_name][gpu_type] = {}
        for mb in mb_size_list:
            llm_info[model_name][gpu_type][str(mb)] = {}
            for tp in tp_size_list:
                if not os.path.exists(os.path.join(profile_dir, model_name + f"_mbs{mb}_tp{tp}_algo0.csv")):
                    continue
                llm_info[model_name][gpu_type][str(mb)][str(tp)] = {}
                for algo_index in algo_list:
                    llm_info[model_name][gpu_type][str(mb)][str(tp)][str(algo_index)] = {}
                    src_data_file = os.path.join(profile_dir, model_name + f"_mbs{mb}_tp{tp}_algo{algo_index}.csv")
                    data_info = []
                    with open(src_data_file) as f:
                        src_data = csv.reader(f)
                        row_index = 0
                        for row in src_data:
                            row_index += 1
                            if row_index == 1:
                                continue
                            data_info.append([row[1], row[2]])
                    op = data_info[0]
                    llm_info[model_name][gpu_type][str(mb)][str(tp)][str(algo_index)]["0"] = [float(op[0]) / microsec_per_unit, float(op[1]) / microsec_per_unit, 0]
                    for i in range(num_layers[model_name]):
                        for j in range(13):
                            op = data_info[j + 1]
                            op_index = 1 + i * 13 + j
                            llm_info[model_name][gpu_type][str(mb)][str(tp)][str(algo_index)][str(op_index)] = [float(op[0]) / microsec_per_unit, float(op[1]) / microsec_per_unit, 0]
                    for j in range(2):
                        op = data_info[-2 + j]
                        op_index = 1 + num_layers[model_name] * 13 + j
                        llm_info[model_name][gpu_type][str(mb)][str(tp)][str(algo_index)][str(op_index)] = [float(op[0]) / microsec_per_unit, float(op[1]) / microsec_per_unit, 0]


json.dump(llm_info, open(output_name, 'w'), indent=2, sort_keys=False)