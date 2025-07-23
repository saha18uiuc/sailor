#! /bin/bash

if [ $# -lt 1 ]; then
    echo "Please provide a save path"
    exit 1
fi

RUNTIME_PATH=$(pwd)/
PROFILING_PATH=${RUNTIME_PATH}$1/
mkdir -p ${PROFILING_PATH}
FILE_NAME=${PROFILING_PATH}p2p_intra_node.csv

MASTER_ADDR=localhost \
MASTER_PORT=7000 \
NNODES=1 \
GPUS_PER_NODE=2 \
NODE_RANK=0 \
FILE_NAME=$FILE_NAME \
python3 p2p_band_profiler.py