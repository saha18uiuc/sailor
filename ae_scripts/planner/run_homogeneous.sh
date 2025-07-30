#!/bin/bash

mkdir -p /root/sailor/ae_results/planner/fig7

python ae_scripts/planner/run_all_sim.py \
--model-name OPT-350 \
--gpu-type A100-40 \
--trace_file /root/sailor/sailor/Planner/simulations/configs/gpu_trace_scaled.csv \
--basic_cluster_config_json /root/sailor/sailor/Planner/simulations/configs/basic_cluster_config.json \
--simulator_profile_file /root/sailor/sailor/Planner/simulations/profiles_tmp.json \
--quotas_dict /root/sailor/sailor/Planner/sailor_planner/dummy_quotas_dict.json \
--gpus-per-node 4 \
--sailor_path /root \
--res_dir ae_results/planner/fig7 \
--objective throughput \
--baselines homogeneous