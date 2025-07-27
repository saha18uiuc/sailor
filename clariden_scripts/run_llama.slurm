#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --gpus-per-task=4

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# TRAIN
GLOBAL_BATCH_SIZE=$1
DP_SIZE=$2
PP_SIZE=$3
TP_SIZE=$4
MICRO_BATCH_SIZE=$5
TRAIN_ITERS=10
MODEL_NAME=LLAMA
GA_STEPS=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DP_SIZE))) 

mkdir -p $BASE_DIR
export SAILOR_LOGS_DIR=$BASE_DIR/N${SLURM_NNODES}/N${SLURM_NNODES}_M${TP_SIZE}_G${GLOBAL_BATCH_SIZE}_D${DP_SIZE}_P${PP_SIZE}_T${TP_SIZE}_M${MICRO_BATCH_SIZE}

# SYNC
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=1234

echo "MASTER_ADDR="$MASTER_ADDR
echo "GA_STEPS="$GA_STEPS
echo "SAILOR_LOGS_DIR="$SAILOR_LOGS_DIR
mkdir -p $SAILOR_LOGS_DIR

config_json="$HOME/ds_config.$SLURM_JOB_ID.json"
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
    --num-layers 32 \
    --num-transformer-layers-original 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
"

CONSTANT_ARGS=" \
    --loss-scale 12 \
    --lr 6.0e-5 \
    --min-lr 6.0e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 40 \
    --eval-interval 1000 \
    --data-path /root/Megatron-DeepSpeed/data/meg-gpt2-oscar-en-10k_text_document \
    --vocab-file /root/Megatron-DeepSpeed/data/gpt2-vocab.json \
    --merge-file /root/Megatron-DeepSpeed/data/gpt2-merges.txt \
    --save-interval 1000 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
"

set -eo pipefail


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $config_json \
"

LAUNCHER="torchrun \
    --nproc_per_node $TP_SIZE \
    --nnodes=\$SLURM_NNODES \
    --master-addr=\$MASTER_ADDR \
    --master-port=\$MASTER_PORT \
"

CMD=" \
    /root/Megatron-DeepSpeed/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $CONSTANT_ARGS \
    $DEEPSPEED_ARGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --train-iters $TRAIN_ITERS \
    --model-name $MODEL_NAME \
    --gpu-type GH-96 \
    --results-dir /root/Megatron-DeepSpeed/results
"


srun -A a-a09 -t 15:00:00 -ul --container-writable --environment=/capstor/scratch/cscs/$USER/elastic-spot-ml/clariden_scripts/sailor.toml /bin/bash -c "

echo 'hello'

$LAUNCHER --node_rank \$SLURM_PROCID $CMD

"
