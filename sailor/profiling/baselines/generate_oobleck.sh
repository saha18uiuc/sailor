#!/bin/bash

GPU="RTX-3090"
tp_degrees=(1 2 4 8)

for tp in "${tp_degrees[@]}"
do
    echo $GPU $tp
    python generate_baseline_profs.py --path-sim ../..//Planner/simulations/profiles_tmp.json --path-mem ../../Planner/llm_info.json --network-coeff-path ../../providers/gcp/multizone_bandwidths.json --planner Oobleck --model OPT-350 --gpu-type $GPU --num-layers 26 --optimizer Adam --gpus-per-node $tp --extra-mem-file ../baseline_mem_prof/oobleck_profile_OPT.json
done