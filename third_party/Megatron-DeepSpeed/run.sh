#!/bin/bash

if [ "$#" -ne 9 ]; then
    echo "Please provide 9 arguments: NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, DP_SIZE, PP_SIZE, TP_SIZE, GLOBAL_BATCH_SIZE, MICRO_BATCH_SIZE"
    exit 1
fi

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3
MASTER_PORT=$4

DP_SIZE=$5
PP_SIZE=$6
TP_SIZE=$7
GLOBAL_BATCH_SIZE=$8

MICRO_BATCH_SIZE=$9
TRAIN_ITERS=10
MODEL_NAME=OPT

GA_STEPS=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP_SIZE)))

config_json="/root/sailor/third_party/Megatron-DeepSpeed/ds_config.json"
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_accumulation_steps": $GA_STEPS,
  "zero_optimization": {},
  "optimizer": {
      "type": "AdamW",
      "params": {
          "lr": 1.0e-5,
          "betas": [0.9, 0.999],
          "eps": 1.0e-8,
          "weight_decay": 4.0e-5
      }
   },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

# MODEL ARGS
GPT_ARGS=" \
    --num-layers 24 \
    --num-transformer-layers-original 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

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

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $config_json \
"


LAUNCHER="torchrun \
    --nproc_per_node $TP_SIZE \
    --nnodes=$NNODES \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
"

CMD=" \
    /root/sailor/third_party/Megatron-DeepSpeed/train_llm.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --data-parallel-size $DP_SIZE \
    $GPT_ARGS \
    $CONSTANT_ARGS \
    $DEEPSPEED_ARGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --model-name $MODEL_NAME \
    --gpu-type RTX \
"

$LAUNCHER --node_rank $NODE_RANK $CMD
