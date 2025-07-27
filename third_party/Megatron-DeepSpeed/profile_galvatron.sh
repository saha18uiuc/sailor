#!/bin/bash

#profiles up to a large batch size - if OOM, it will just fail and do not generate the profile

mkdir -p logs
export SAILOR_LOGS_DIR=logs

mkdir -p $2

python profile_all.py --tp 1 --pp 1 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 1 --profile galvatron
python profile_all.py --tp 1 --pp 1 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile galvatron
python profile_all.py --tp 1 --pp 2 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 1 --profile galvatron
python profile_all.py --tp 1 --pp 2 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile galvatron

python profile_all.py --tp 2 --pp 1 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 1 --profile galvatron
python profile_all.py --tp 2 --pp 1 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile galvatron
python profile_all.py --tp 2 --pp 2 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 1 --profile galvatron
python profile_all.py --tp 2 --pp 2 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile galvatron

python profile_all.py --tp 4 --pp 1 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 1 --profile galvatron
python profile_all.py --tp 4 --pp 1 --max_bs 1 --model_name $1 --results_dir $2 --num_prof_layers 2 --profile galvatron