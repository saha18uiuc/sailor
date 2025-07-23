#! /bin/bash

if [ $# -lt 2 ]; then
    echo "Please provide the GPU type and #GPU per node"
    exit 1
fi

PROFILING_PATH=$1/profiled-data-fp16
bash scripts/profile_p2p.sh ${PROFILING_PATH}
bash scripts/profile_model.sh OPT-30 fp16 ../../../simulations/tests/training_config_opt_30.json $1 $2
bash scripts/profile_model.sh OPT-350 fp16 ../../../simulations/tests/training_config_opt_350.json $1 $2
bash scripts/profile_model.sh GPT-Neo-2.7 fp16 ../../../simulations/tests/training_config_gpt_neo27.json $1 $2

PROFILING_PATH=$1/profiled-data-fp32
bash scripts/profile_p2p.sh ${PROFILING_PATH}
bash scripts/profile_model.sh OPT-30 fp32 ../../../simulations/tests/training_config_opt_30.json $1 $2
bash scripts/profile_model.sh OPT-350 fp32 ../../../simulations/tests/training_config_opt_350.json $1 $2
bash scripts/profile_model.sh GPT-Neo-2.7 fp32 ../../../simulations/tests/training_config_gpt_neo27.json $1 $2

