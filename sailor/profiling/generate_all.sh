#!/bin/bash

PLANNER="Metis"
GPU="A100-80"
GPUS_PER_NODE=4
echo $PLANNER

python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner $PLANNER --model OPT-350 --gpu-type $GPU --num-layers 26 --optimizer Adam --gpus-per-node $GPUS_PER_NODE

# python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner $PLANNER --model OPT-350 --gpu-type A100-40 --num-layers 26 --optimizer Adam --gpus-per-node $GPUS_PER_NODE

# python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner $PLANNER --model OPT-350 --gpu-type V100-16 --num-layers 26 --optimizer Adam --gpus-per-node $GPUS_PER_NODE

# python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner $PLANNER --model OPT-30 --gpu-type A100-40 --num-layers 50 --optimizer Adam --gpus-per-node $GPUS_PER_NODE

# python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner $PLANNER --model OPT-30 --gpu-type V100-16 --num-layers 50 --optimizer Adam --gpus-per-node $GPUS_PER_NODE

# python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner $PLANNER --model LLAMA-3-8 --gpu-type A100-40 --num-layers 35 --optimizer Adam --gpus-per-node $GPUS_PER_NODE

# python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner $PLANNER --model LLAMA-3-8 --gpu-type V100-16 --num-layers 35 --optimizer Adam --gpus-per-node $GPUS_PER_NODE

# python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner $PLANNER --model GPT-Neo-2.7 --gpu-type A100-40 --num-layers 34 --optimizer Adam --gpus-per-node $GPUS_PER_NODE

# python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner $PLANNER --model GPT-Neo-2.7 --gpu-type V100-16 --num-layers 34 --optimizer Adam --gpus-per-node $GPUS_PER_NODE
