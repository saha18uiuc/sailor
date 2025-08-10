#!/bin/bash
#SBATCH --job-name=test
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
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
NNODE=$((DP_SIZE * PP_SIZE))
TRAIN_ITERS=10
model_name=OPT-350

# SYNC
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=1234
DISTRIBUTED_ARGS=""

#### Paths ####
RESULT_PATH=${ROOT_PATH}/aceso_validation_clariden/
if [[ "$PRINT_MEMORY" -eq 0 ]]; then
   LOG_PATH=${RESULT_PATH}runtime/${model_name}/time/
else
   LOG_PATH=${RESULT_PATH}runtime/${model_name}/memory/
fi

CONFIG_SAVE_PATH=${RESULT_PATH}configs/
mkdir -p ${LOG_PATH}csv

config_name=${model_name}_${NNODE}-${GLOBAL_BATCH_SIZE}-${DP_SIZE}-${PP_SIZE}-${MICRO_BATCH_SIZE}
echo $config_name
file_name=${config_name}.json
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')

LAUNCHER="torchrun \
    --nproc_per_node $TP_SIZE \
    --nnodes \$SLURM_NNODES \
    --master_addr \$MASTER_ADDR \
    --master_port \$MASTER_PORT \
"

CMD=" \
    runtime/pretrain_gpt.py \
    --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
    --train-iters $TRAIN_ITERS \
    --eval-iters 0 \
    --lr-decay-iters 320000 \
    --vocab-file /root/sailor/third_party/Megatron-DeepSpeed/data/gpt2-vocab.json \
    --merge-file /root/sailor/third_party/Megatron-DeepSpeed/data/gpt2-merges.txt \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --log-interval 1 \
    --DDP-impl local \
    --log-path $LOG_PATH
"

srun -A a-infra02 -t 05:00:00 -ul --container-writable --environment=/capstor/scratch/cscs/$USER/sailor/ae_scripts/clariden_scripts/sailor.toml /bin/bash -c "

echo 'hello'
cd $ROOT_PATH
$LAUNCHER --node_rank \$SLURM_PROCID $CMD

"
