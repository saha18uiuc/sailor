#! /bin/bash

if [ $# -lt 4 ]; then
    echo "Please provide model name [GPT-Neo-2.7/OPT-30/OPT-350], data type [fp16/fp32], training config path, GPU type and max #GPU"
    exit 1
fi

MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=1
NODE_RANK=0

MAX_NUM_GPUS=$5
MODEL_NAME=$1
DATA_TYPE=$2
TRAINING_CONFIG_JSON=$3

RUNTIME_PATH=$(pwd)/$4/
if [ ${DATA_TYPE} == "fp32" ]; then
    PROFILING_PATH=${RUNTIME_PATH}profiled-data-fp32/
    mkdir -p ${PROFILING_PATH}

    for ((tp_size=1; tp_size<=$MAX_NUM_GPUS; tp_size=tp_size*2))
    do
    GPUS_PER_NODE=${tp_size}
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK "

    echo [fp32] [TIME] before profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

    python torch.distributed.launch $DISTRIBUTED_ARGS \
        op_profiler.py \
        --training-config-json $TRAINING_CONFIG_JSON \
        --prof-tp-size $tp_size \
        --prof-path $PROFILING_PATH \
        --prof-cache-file ${PROFILING_PATH}${MODEL_NAME}_op_profile.pkl \
        --prof-model-name $MODEL_NAME \
        --prof-warmup-times 10 \
        --prof-repeat-times 40 \
        2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_op_tp${tp_size}.log

    echo [fp32] [TIME] after profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
    done

    for ((num_gpus=2; num_gpus<=$MAX_NUM_GPUS; num_gpus=num_gpus*2))
    do
    echo [fp32] [TIME] before profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

    python3 comm_profiler.py \
        --prof-path $PROFILING_PATH \
        --prof-cache-file ${PROFILING_PATH}comm_profile.pkl \
        --prof-op-time-path $PROFILING_PATH \
        --prof-tp-size $num_gpus \
        --prof-model-name $MODEL_NAME \
        --prof-warmup-times 5 \
        --prof-repeat-times 20 \
        --max-data-size 4096 \
        2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_comm${num_gpus}gpus.log

    echo [fp32] [TIME] after profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

    done
elif [ ${DATA_TYPE} == "fp16" ]; then
    PROFILING_PATH=${RUNTIME_PATH}profiled-data-fp16/
    mkdir -p ${PROFILING_PATH}

    for ((tp_size=1; tp_size<=$MAX_NUM_GPUS; tp_size=tp_size*2))
    do
    GPUS_PER_NODE=${tp_size}
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK "

    echo [fp16] [TIME] before profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

    torchrun $DISTRIBUTED_ARGS \
        op_profiler.py \
        --training-config-json $TRAINING_CONFIG_JSON \
        --prof-tp-size $tp_size \
        --prof-path $PROFILING_PATH \
        --prof-cache-file ${PROFILING_PATH}${MODEL_NAME}_op_profile.pkl \
        --prof-model-name $MODEL_NAME \
        --prof-warmup-times 10 \
        --prof-repeat-times 40 \
        --fp16 \
        2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_op_tp${tp_size}.log

    echo [fp16] [TIME] after profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
    done

    for ((num_gpus=2; num_gpus<=$MAX_NUM_GPUS; num_gpus=num_gpus*2))
    do
    echo [fp16] [TIME] before profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

    python3 comm_profiler.py \
        --prof-path $PROFILING_PATH \
        --prof-cache-file ${PROFILING_PATH}comm_profile.pkl \
        --prof-op-time-path $PROFILING_PATH \
        --prof-tp-size $num_gpus \
        --prof-model-name $MODEL_NAME \
        --prof-warmup-times 5 \
        --prof-repeat-times 20 \
        --max-data-size 4096 \
        --fp16 \
        2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_comm${num_gpus}gpus.log

    echo [fp16] [TIME] after profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

    done
else
    echo "Data type not recognized"
    exit 1
fi
