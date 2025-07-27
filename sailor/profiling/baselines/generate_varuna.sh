#!/bin/bash

GPU="SGS-RTX"
tp_degrees=(1 2 4)
model=OPT-350
nlayers=26

for tp in "${tp_degrees[@]}"
do
    python generate_baseline_profs.py --path-sim ../../Planner/simulations/profiles_tmp.json --path-mem ../../Planner/llm_info.json --network-coeff-path ../../providers/network_coeffs_all_homo.json --planner Varuna --model $model --gpu-type $GPU --num-layers $nlayers --optimizer Adam --gpus-per-node $tp --extra-mem-file $1
done