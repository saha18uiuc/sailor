#!/bin/bash

# mainly for testing

CONSTANT_ARGS=" \
    --loss-scale 12 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1000 \
    --data-path /root/sailor/third_party/Megatron-DeepSpeed/data/meg-gpt2-oscar-en-10k_text_document \
    --vocab-file /root/sailor/third_party/Megatron-DeepSpeed/data/gpt2-vocab.json \
    --merge-file /root/sailor/third_party/Megatron-DeepSpeed/data/gpt2-merges.txt \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
"

NO_CONTROLLER_ARGS="
    --global_batch_size 1 \
    --micro_batch_size 1 \
    --num_stages 1 \
    --rank $1 \
    --world_size 4 \
    --master_ip 127.0.0.1 \
    --master_port 1234 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --data-parallel-size 1 \
"

NO_CONTROLLER_HET_ARGS="
    --global_batch_size 2 \
    --micro_batch_size 1 \
    --num_stages 1 \
    --rank $1 \
    --world_size 1 \
    --master_ip 127.0.0.1 \
    --master_port 1234 \
    --tensor-model-parallel-size $2 \
    --pipeline-model-parallel-size 1 \
    --data-parallel-size 1 \
    --max-tensor-model-parallel-size 1 \
    --distributed-config-file dist_config.json \
"

# MODEL ARGS
MODEL_ARGS=" \
    --num-layers 16 \
    --num-transformer-layers-original 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

python elastic_worker_agent.py \
    --with_controller \
    --use_megatron \
    --deepspeed \
    --model-name OPT \
    --gpu-type V100 \
    --train-iters 100 \
    --results-dir tests \
    --ds_config_file /root/sailor/third_party/Megatron-DeepSpeed/ds_config.json \
    $MODEL_ARGS \
    $CONSTANT_ARGS
