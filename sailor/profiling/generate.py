import os

models = [("OPT-350", 26), ("OPT-30", 50), ("LLAMA-3-8", 34), ("GPT-Neo-2.7", 34)]

for model, nlayers in models:
    cmd = f"python generate_baseline_profs.py --path-sim ../Planner/simulations/profiles_tmp.json --path-mem ../Planner/llm_info.json --network-coeff-path ../providers/gcp/network_coeffs.json --planner AMP --model {model} --gpu-type A100-40 --num-layers {nlayers}"
    print(cmd)
    os.system(cmd)

