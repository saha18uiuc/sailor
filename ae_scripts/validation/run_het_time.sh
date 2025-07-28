#!/bin/bash

base_path=/root/sailor/sailor/Planner

nodes=(2 4 6)

mkdir -p /root/sailor/ae_results/validation/fig6

# SAILOR

for n in "${nodes[@]}"
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/simulations/profiles_tmp.json \
    --simulate_time \
    --simulator SAILOR \
    --output_path /root/sailor/ae_results/validation/fig6 \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type RTX-3090 \
    --input_path $base_path/simulations/validation/mit/N$n
done

# Piper

for n in "${nodes[@]}"
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/baselines/Piper/profiles/OPT-350/RTX-3090/profile.json \
    --simulate_time \
    --simulator Piper \
    --output_path /root/sailor/ae_results/validation/fig6 \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type RTX-3090 \
    --input_path $base_path/simulations/validation/mit/N$n
done

# Varuna

# first for n=2
python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/baselines/Varuna/profiles/OPT-350/RTX-3090/profile_2.json \
    --simulate_time \
    --simulator Varuna \
    --output_path /root/sailor/ae_results/validation/fig6 \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type RTX-3090 \
    --input_path $base_path/simulations/validation/mit/N2

for n in 4 6
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/baselines/Varuna/profiles/OPT-350/RTX-3090/profile_8.json \
    --simulate_time \
    --simulator Varuna \
    --output_path /root/sailor/ae_results/validation/fig6 \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type RTX-3090 \
    --input_path $base_path/simulations/validation/mit/N$n
done


# Metis

for n in "${nodes[@]}"
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/baselines/Metis/profiles/OPT-350 \
    --simulate_time \
    --simulator Metis \
    --output_path /root/sailor/ae_results/validation/fig6 \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type RTX-3090 \
    --input_path $base_path/simulations/validation/mit/N$n
done



# FlashFlex

for n in "${nodes[@]}"
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/baselines/FlashFlex/src/machine_amounts.json \
    --simulate_time \
    --simulator FlashFlex \
    --output_path /root/sailor/ae_results/validation/fig6 \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type RTX-3090  \
    --input_path $base_path/simulations/validation/mit/N$n
done
