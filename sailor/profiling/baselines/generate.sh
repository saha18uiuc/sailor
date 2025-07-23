#!/bin/bash

GPU="A100-40"
tp_degrees=(1 2 4)
planner="AMP"
model="OPT-350"
layers=26

for tp in "${tp_degrees[@]}"
do
    python generate_baseline_profs.py --path-sim ../../Planner/simulations/profiles_tmp.json --path-mem ../../Planner/llm_info.json --network-coeff-path ../../providers/gcp/network_coeffs.json --planner $planner --model $model --gpu-type $GPU --num-layers $layers --optimizer Adam --gpus-per-node $tp
done