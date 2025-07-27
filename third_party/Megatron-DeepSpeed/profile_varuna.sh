#!/bin/bash

# profiles up to a large batch size - if OOM, it will just fail and do not generate the profile

tp_degrees=(1 2 4)

mkdir -p logs
export SAILOR_LOGS_DIR=logs

mkdir -p $2

for tp in "${tp_degrees[@]}"
do
    python profile_all.py --tp $tp --pp 1 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 1 --profile varuna --use-embedding
    python profile_all.py --tp $tp --pp 1 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 1 --profile varuna --use-transformer
    python profile_all.py --tp $tp --pp 1 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 1 --profile varuna --use-last
done
