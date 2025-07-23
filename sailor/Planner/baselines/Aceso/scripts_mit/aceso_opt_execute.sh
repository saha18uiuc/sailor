export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# TRAIN
MASTER_ADDR=$1
NODEID=$2
GLOBAL_BATCH_SIZE=$3
DP_SIZE=$4
PP_SIZE=$5
TP_SIZE=$6
MICRO_BATCH_SIZE=$7

NNODE=$((DP_SIZE * PP_SIZE))
TRAIN_ITERS=10
model_name=OPT-350


ROOT_PATH=/root/elastic-spot-ml/sailor/Planner/baselines/Aceso

# SYNC
export MASTER_PORT=1234
DISTRIBUTED_ARGS=""

#### Paths ####
RESULT_PATH=${ROOT_PATH}/aceso_validation_mit/
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
    --nnodes $NNODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"
echo $LAUNCHER

CMD=" \
    runtime/pretrain_gpt.py \
    --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
    --train-iters $TRAIN_ITERS \
    --eval-iters 0 \
    --lr-decay-iters 320000 \
    --vocab-file runtime/vocabs/gpt2-vocab.json \
    --merge-file runtime/vocabs/gpt2-merges.txt \
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

cd $ROOT_PATH
$LAUNCHER --node_rank $NODEID $CMD
