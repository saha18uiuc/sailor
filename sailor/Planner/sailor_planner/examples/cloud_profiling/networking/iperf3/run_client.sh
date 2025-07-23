#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 --num_runs"
    exit 1
fi

OUTPUT_FILE="results/us-central1-a-us-central1-a.json"
MAX_NUM_STREAMS=128
INCREMENT=4

# Add an opening square bracket to start list for facilitating parsing in python
echo "[" > $OUTPUT_FILE &&

for ((i = 1; i <= $MAX_NUM_STREAMS; i += $INCREMENT)); do
    for ((j = 1; j <= $1; j++)); do
        iperf3 -c 10.128.15.210 -p 7575 -P $i -J >> $OUTPUT_FILE &&
        if [ $j -lt $1 ]; then
            echo "," >> $OUTPUT_FILE
        fi
    done;
    
    # Add a comma if it's not the last iteration
    if (( $i + $INCREMENT < $MAX_NUM_STREAMS )); then
        echo "," >> $OUTPUT_FILE
    fi
done

# Add a closing square bracket to end of list
echo "]" >> $OUTPUT_FILE
