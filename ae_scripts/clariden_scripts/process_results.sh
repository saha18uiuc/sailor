#!/bin/bash

# process results for each scenario
BASE_DIR=$1
TARGET_DIR=$2

################################## 1 node

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 1 \
--gpus_per_node 4 \
--gbs 1 \
--dp 1 \
--pp 1 \
--tp 4 \
--mbs 1 \
--target_dir $TARGET_DIR

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 1 \
--gpus_per_node 4 \
--gbs 32 \
--dp 1 \
--pp 1 \
--tp 4 \
--mbs 1 \
--target_dir $TARGET_DIR


################################## 2 nodes

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 2 \
--gpus_per_node 4 \
--gbs 32 \
--dp 1 \
--pp 2 \
--tp 4 \
--mbs 8 \
--target_dir $TARGET_DIR

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 2 \
--gpus_per_node 4 \
--gbs 32 \
--dp 2 \
--pp 1 \
--tp 4 \
--mbs 2 \
--target_dir $TARGET_DIR

################################## 4 nodes

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 4 \
--gpus_per_node 4 \
--gbs 64 \
--dp 1 \
--pp 4 \
--tp 4 \
--mbs 1 \
--target_dir $TARGET_DIR

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 4 \
--gpus_per_node 4 \
--gbs 64 \
--dp 2 \
--pp 2 \
--tp 4 \
--mbs 1 \
--target_dir $TARGET_DIR

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 4 \
--gpus_per_node 4 \
--gbs 128 \
--dp 4 \
--pp 1 \
--tp 4 \
--mbs 4 \
--target_dir $TARGET_DIR

################################## 8 nodes

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 8 \
--gpus_per_node 4 \
--gbs 512 \
--dp 2 \
--pp 4 \
--tp 4 \
--mbs 8 \
--target_dir $TARGET_DIR

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 8 \
--gpus_per_node 4 \
--gbs 512 \
--dp 4 \
--pp 2 \
--tp 4 \
--mbs 8 \
--target_dir $TARGET_DIR

################################## 16 nodes

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 16 \
--gpus_per_node 4 \
--gbs 1024 \
--dp 16 \
--pp 1 \
--tp 4 \
--mbs 8 \
--target_dir $TARGET_DIR

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 16 \
--gpus_per_node 4 \
--gbs 1024 \
--dp 4 \
--pp 4 \
--tp 4 \
--mbs 8 \
--target_dir $TARGET_DIR

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 16 \
--gpus_per_node 4 \
--gbs 1024 \
--dp 8 \
--pp 2 \
--tp 4 \
--mbs 8 \
--target_dir $TARGET_DIR

################################## 32 nodes

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 32 \
--gpus_per_node 4 \
--gbs 1024 \
--dp 16 \
--pp 2 \
--tp 4 \
--mbs 8 \
--target_dir $TARGET_DIR

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 32 \
--gpus_per_node 4 \
--gbs 1024 \
--dp 32 \
--pp 1 \
--tp 4 \
--mbs 8 \
--target_dir $TARGET_DIR

python3 process_results.py --main_dir $BASE_DIR \
--nnodes 32 \
--gpus_per_node 4 \
--gbs 1024 \
--dp 8 \
--pp 4 \
--tp 4 \
--mbs 8 \
--target_dir $TARGET_DIR