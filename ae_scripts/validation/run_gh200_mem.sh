#!/bin/bash

base_path=/root/sailor/sailor/Planner

nodes=(1 2 4 8 16 32)

mkdir -p /root/sailor/ae_results/validation/fig5a

# SAILOR

for n in "${nodes[@]}"
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/simulations/profiles_tmp.json \
    --simulate_mem \
    --simulator SAILOR \
    --output_path /root/sailor/ae_results/validation/fig5a \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type GH-96 \
    --input_path $base_path/simulations/validation/clariden/OPT-350/N$n
done

# Piper

for n in "${nodes[@]}"
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/baselines/Piper/profiles/OPT-350/GH-96/profile.json \
    --simulate_mem \
    --simulator Piper \
    --output_path /root/sailor/ae_results/validation/fig5a \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type GH-96 \
    --input_path $base_path/simulations/validation/clariden/OPT-350/N$n
done

# Varuna

for n in "${nodes[@]}"
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/baselines/Varuna/profiles/OPT-350/GH-96/profile_4.json \
    --simulate_mem \
    --simulator Varuna \
    --output_path /root/sailor/ae_results/validation/fig5a \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type GH-96 \
    --input_path $base_path/simulations/validation/clariden/OPT-350/N$n
done


# Metis

for n in "${nodes[@]}"
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/baselines/Metis/profiles/OPT-350 \
    --simulate_mem \
    --simulator Metis \
    --output_path /root/sailor/ae_results/validation/fig5a \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type GH-96 \
    --input_path $base_path/simulations/validation/clariden/OPT-350/N$n
done



# # FlashFlex

for n in "${nodes[@]}"
do
	python $base_path/simulations/validation/run_val.py \
    --training_config_json $base_path/simulations/configs/training_config_opt_350.json \
    --simulator_profile_path $base_path/baselines/FlashFlex/src/machine_amounts.json \
    --simulate_mem \
    --simulator FlashFlex \
    --output_path /root/sailor/ae_results/validation/fig5a \
    --sailor_path /root \
    --basic_cluster_config_json $base_path/simulations/configs/basic_cluster_config.json \
    --gpu_type GH-96 \
    --input_path $base_path/simulations/validation/clariden/OPT-350/N$n
done
