GPU_TYPE=RTX-3090
BASIC_CLUSTER_CONFIG=../../simulations/tests/basic_cluster_config.json
TRAINING_CONFIG_JSON=../../simulations/tests/training_config_opt_350.json
SIMULATOR_PROFILE_FILE=../../simulations/profiles_tmp_aceso.json
SIMULATOR_LLM_INFO=../../llm_info_aceso.json

PLAN_DIR=../../simulations/validation/mit_v2
for NNODES in $(ls $PLAN_DIR)
do
    for PLAN in $(ls $PLAN_DIR/$NNODES)
    do
    echo $PLAN
    PLAN_PATH=$PLAN_DIR/$NNODES/$PLAN
    python scripts_mit/create_plan.py \
        --plan_path $PLAN_PATH \
        --gpu_type $GPU_TYPE \
        --basic_cluster_config_json $BASIC_CLUSTER_CONFIG \
        --training_config_json $TRAINING_CONFIG_JSON \
        --simulator_profile_file $SIMULATOR_PROFILE_FILE \
        --simulator_llm_info $SIMULATOR_LLM_INFO
    echo ""
    
    done
done