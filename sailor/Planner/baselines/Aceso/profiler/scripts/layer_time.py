import csv 
import os

os.makedirs("layer_time", exist_ok=True)
gpu_list=["RTX-3090", "A100-40", "V100-16", "GH-96", "RTX-2080", "Titan-RTX"]
for gpu_type in gpu_list:
    results = ["mbs,tp,algo,fwd,bwd,fwd+bwd\n"]
    for mbs in [1,2,4,8]:
        for tp in [1,2,4,8]:
            for algo in [0,1]:
                profile_path = f'{gpu_type}/profiled-data-fp32/OPT-350_mbs{mbs}_tp{tp}_algo{algo}.csv'
                if os.path.exists(profile_path):
                    fwd_time = 0
                    bwd_time = 0
                    with open(profile_path, 'r') as f:
                        src_data = csv.reader(f)
                        row_index = 0
                        for row in src_data:
                            if row_index <= 1 or row_index >= 15:
                                row_index += 1
                                continue
                            row_index += 1
                            # print(f"add {row[0]}")
                            fwd_time += float(row[1])
                            bwd_time += float(row[2])
                    fwd_time /= 1000000
                    bwd_time /= 1000000
                    total_time = fwd_time + bwd_time
                    results.append(f"{mbs},{tp},{algo},{fwd_time:.6f},{bwd_time:.6f},{total_time:.6f}\n")
    with open(f"layer_time/{gpu_type}.csv", 'w') as f:
        f.writelines(results)