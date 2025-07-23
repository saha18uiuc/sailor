# To generate llm_info_op.json and profiles_tmp_op.jsons 
python gen_llm_info.py --profile-dir ../profiler/RTX-3090/profiled-data-fp32/
GPU_TYPE_LIST="V100-16,GH-96,RTX-3090,A100-40,Titan-RTX,RTX-2080"
python gen_profiles_tmp.py --gpu-type-list $GPU_TYPE_LIST --profile-dir ../profiler/
python gen_profiles_tmp.py --gpu-type-list $GPU_TYPE_LIST --fp16 --profile-dir ../profiler/
cp llm_info_aceso.json ../../../
cp profiles_tmp_aceso.json ../../../simulations
cp profiles_tmp_aceso_fp16.json ../../../simulations
rm llm_info_aceso.json profiles_tmp_aceso.json profiles_tmp_aceso_fp16.json