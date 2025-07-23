#!/bin/bash

RANK=0
WORLD_SIZE=4
MASTER_IP="10.128.15.210"
MASTER_PORT="29500"
OUTPUT_FILE="1Z-us-central1-a"
NUM_RUNS=20
GPU="V100"
NCCL_SOCKET_NTHREADS=4

### All Reduce of a tensor of different size using world_size GPUs

# python3 testb_all_reduce.py --rank $RANK --world_size $WORLD_SIZE --master_ip $MASTER_IP --master_port $MASTER_PORT --socket_nthreads $NCCL_SOCKET_NTHREADS --data_size 64000 --num_runs $NUM_RUNS --output_file $OUTPUT_FILE --gpu $GPU

# for ((i = 128000; i <= 1048576000; i *= 2)); do
#     python3 testb_all_reduce.py --rank $RANK --world_size $WORLD_SIZE --master_ip $MASTER_IP --master_port $MASTER_PORT --socket_nthreads $NCCL_SOCKET_NTHREADS --data_size $i --num_runs $NUM_RUNS --output_file $OUTPUT_FILE --gpu $GPU;
# done

### Sending 1 GB from sender to receiver

python3 test_sendrecv.py --rank $RANK --master_ip $MASTER_IP --master_port $MASTER_PORT --socket_nthreads $NCCL_SOCKET_NTHREADS --data_size 64000 --num_runs $NUM_RUNS --output_file $OUTPUT_FILE --gpu $GPU

for ((i = 128000; i <= 1048576000; i *= 2)); do
    python3 test_sendrecv.py --rank $RANK --master_ip $MASTER_IP --master_port $MASTER_PORT --socket_nthreads $NCCL_SOCKET_NTHREADS --data_size $i --num_runs $NUM_RUNS --output_file $OUTPUT_FILE --gpu $GPU;
done
