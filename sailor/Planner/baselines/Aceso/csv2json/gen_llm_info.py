import re
import sys
import os
import math
from collections import defaultdict
import csv
import json
import argparse
sys.path.append("../")

parser = argparse.ArgumentParser()
parser.add_argument('--profile-dir', type=str, required=True, help='directory of profile data')
args = parser.parse_args()

data_per_mb = 1024 * 1024 / 4
profile_dir = args.profile_dir

tp_size_list = [1, 2, 4, 8]
algo_list = [0, 1]
model_list = ["GPT-Neo-2.7", "OPT-30", "OPT-350"]
llm_info = {}
num_layers = {"GPT-Neo-2.7": 32, 
              "OPT-30": 48, 
              "OPT-350": 24, }

for model_name in model_list:
    llm_info[model_name] = {}
    for tp in tp_size_list:
        if not os.path.exists(profile_dir + model_name + f"_mbs{1}_tp{tp}_algo0.csv"):
            continue
        llm_info[model_name][str(tp)] = {}
        for algo_index in algo_list:
            llm_info[model_name][str(tp)][str(algo_index)] = {}
            src_data_file = profile_dir + model_name + f"_mbs{1}_tp{tp}_algo{algo_index}.csv"
            data_info = []
            with open(src_data_file) as f:
                src_data = csv.reader(f)
                row_index = 0
                for row in src_data:
                    row_index += 1
                    if row_index == 1:
                        continue
                    op_name, input, output, param, activation = row[0], float(row[3]), float(row[4]), float(row[5]), float(row[6])
                    # print(op_name, activation)
                    if op_name == "enc-attention-softmax":
                        activation = 0
                    data_info.append([param, output, input, activation])
            op = data_info[0]
            llm_info[model_name][str(tp)][str(algo_index)]["0"] = {
                "params_floats": float(op[0]) * data_per_mb,
                "act_output_floats": float(op[1]) * data_per_mb,
                "act_input_floats": float(op[2]) * data_per_mb,
                "act_mem_floats": float(op[3]) * data_per_mb,
                "params_bytes": float(op[0]) * 1024 * 1024,
                "act_output_bytes": float(op[1]) * 1024 * 1024,
                "act_input_bytes": float(op[2]) * 1024 * 1024,
                "act_mem_bytes": float(op[3]) * 1024 * 1024
            }
            for i in range(num_layers[model_name]):
                for j in range(13):
                    op = data_info[j + 1]
                    op_index = 1 + i * 13 + j
                    llm_info[model_name][str(tp)][str(algo_index)][str(op_index)] = {
                        "params_floats": float(op[0]) * data_per_mb,
                        "act_output_floats": float(op[1]) * data_per_mb,
                        "act_input_floats": float(op[2]) * data_per_mb,
                        "act_mem_floats": float(op[3]) * data_per_mb,
                        "params_bytes": float(op[0]) * 1024 * 1024,
                        "act_output_bytes": float(op[1]) * 1024 * 1024,
                        "act_input_bytes": float(op[2]) * 1024 * 1024,
                        "act_mem_bytes": float(op[3]) * 1024 * 1024
                    }
            for j in range(2):
                op = data_info[-2 + j]
                op_index = 1 + num_layers[model_name] * 13 + j
                llm_info[model_name][str(tp)][str(algo_index)][str(op_index)] = {
                    "params_floats": float(op[0]) * data_per_mb,
                    "act_output_floats": float(op[1]) * data_per_mb,
                    "act_input_floats": float(op[2]) * data_per_mb,
                    "act_mem_floats": float(op[3]) * data_per_mb,
                    "params_bytes": float(op[0]) * 1024 * 1024,
                    "act_output_bytes": float(op[1]) * 1024 * 1024,
                    "act_input_bytes": float(op[2]) * 1024 * 1024,
                    "act_mem_bytes": float(op[3]) * 1024 * 1024
                }
file_name = "llm_info_aceso.json"                          
json.dump(llm_info, open(file_name, 'w'), indent=4, sort_keys=False)