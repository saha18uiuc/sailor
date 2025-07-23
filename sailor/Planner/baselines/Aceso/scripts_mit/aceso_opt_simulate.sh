#! /bin/bash
ROOT_PATH=$(pwd)
model_name=OPT-350
validation_dir=aceso_validation_mit
GPU_TYPE=RTX-3090

#### Paths ####
RESULT_PATH=${ROOT_PATH}/${validation_dir}/
LOG_PATH=${RESULT_PATH}aceso/${model_name}/
CONFIG_SAVE_PATH=${RESULT_PATH}configs/
mkdir -p ${LOG_PATH}

for file_name in $(ls $CONFIG_SAVE_PATH)
do
    config_name=`basename $file_name .json`

    python aceso_simulate_plan.py --gpu-type $GPU_TYPE \
        --profiling-file-dir profiler/$GPU_TYPE \
        --training-config-path ../../simulations/tests/training_config_opt_350.json \
        --cluster-config-path ../../simulations/tests/basic_cluster_config.json \
        --plan-path $CONFIG_SAVE_PATH${file_name} \
        2>&1 | tee ${LOG_PATH}aceso_${config_name}.log
done