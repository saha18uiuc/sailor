# PROFILING CONSTANTS

# Checkpoint overhead for model
# [GPT2, OPT1.3B] with 4 and 8 V100 nodes respectively
CKPT_DIR_GPT2 = "model_profiling/results/overhead_sailor_V100_4_gpt2-medium.csv"
CKPT_DIR_OPT = "model_profiling/results/overhead_sailor_V100_8_OPT-1.3B.csv"

# Execution layer time
# [OPT-1.3B]x[V100,A100-40]x[fp16,fp32]
EXEC_DIR_OPT = "model_profiling/results"
NAME_FILE_FP16 = "sailor_fp16"
NAME_FILE_FP32 = "sailor_fp32"

# Bandwidth
# [bytes_sizes]x[V100,A100-40]x[1zone]
BW_DIR_OPT = "cloud_profiling/networking/pods"
NAME_FILE_BW = "bw_profiling"
