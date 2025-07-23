import os
import sys

validation_path = sys.argv[1]
log_path = os.path.join(validation_path, "slurm")
for file_name in os.listdir(log_path):
    file_path = os.path.join(log_path, file_name)
    with open(file_path, 'r') as f:
        log = f.readlines()
        config = log[0]
        print(config)
        save_name = "slurm_" + config.strip().split('_')[1] + ".log"
        print_memory = False
        for line in log:
            if "==> Memory " in line:
                print_memory = True
        if print_memory:
            subfolder = "memory"
        else:
            subfolder = "time"
        save_path = os.path.join(validation_path, "runtime/OPT-350", subfolder, save_name)
    with open(save_path, 'w') as f:
        f.writelines(log)