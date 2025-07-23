#!/bin/bash

wsize=$1
hworld=$((wsize/2))
start_rank=$2
end_rank=$((start_rank+hworld-1))

gpus_per_node="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
MASTER_IP=$3
MASTER_PORT=$4

echo $wsize
echo $hworld
echo $start_rank
echo $end_rank
echo $gpus_per_node

echo "--------------------------------"

for i in $(seq $start_rank $end_rank)
do
        echo "Start with rank $i"
        python3 run_all.py --rank $i --master_ip $MASTER_IP --master_port $MASTER_PORT --world_size $wsize --gpus_per_node $gpus_per_node &
done
