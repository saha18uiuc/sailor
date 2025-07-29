#!/bin/bash

mkdir -p /root/sailor/ae_results/planner/fig12

python ae_scripts/planner/run_all_sim.py \
--model-name OPT-350 \
--gpu-type A100-40 \
--trace_file /root/sailor/sailor/Planner/simulations/configs/gpu_trace_2zones_2gpus.csv \
--basic_cluster_config_json /root/sailor/sailor/Planner/simulations/configs/basic_cluster_config.json \
--simulator_profile_file /root/sailor/sailor/Planner/simulations/profiles_tmp.json \
--quotas_dict /root/sailor/sailor/Planner/sailor_planner/dummy_quotas_dict.json \
--gpus-per-node 4 \
--sailor_path /root \
--res_dir ae_results/planner/fig12 \
--objective throughput \
--max_cost_file ae_scripts/planner/max_budget_file.json \
--baselines cost