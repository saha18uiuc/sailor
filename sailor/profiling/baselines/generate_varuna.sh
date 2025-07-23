#!/bin/bash

GPU="A100-40"
tp_degrees=(1 2 4)

for tp in "${tp_degrees[@]}"
do
    python generate_baseline_profs.py --path-sim ../../Planner/simulations/profiles_tmp.json --path-mem ../../Planner/llm_info.json --network-coeff-path ../../providers/gcp/network_coeffs.json --planner Varuna --model OPT-350 --gpu-type $GPU --num-layers 26 --optimizer Adam --gpus-per-node $tp --extra-mem-file ../baseline_mem_prof/varuna_profile_OPT.json
done