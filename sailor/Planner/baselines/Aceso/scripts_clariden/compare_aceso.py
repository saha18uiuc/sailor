import sys
import os 
import csv 

print_time = False
print_memory = False

model_name = 'OPT-350'
src_data_path = 'aceso_validation_clariden'
# N,gbs,dp,pp,tp,mbs
target_configs = [
    [1,32,1,1,4,1],
    [2,32,1,2,4,8],
    [2,32,2,1,4,2],
    [4,64,1,4,4,1],
    [4,64,2,2,4,1],
    [4,128,4,1,4,4],
    [8,512,2,4,4,8],
    [8,512,8,1,4,8],
    [8,512,4,2,4,8],
    [16,1024,16,1,4,8],
    [16,1024,4,4,4,8],
    [16,1024,8,2,4,8],
    [32,1024,16,2,4,8],
    [32,1024,32,1,4,8],
    [32,1024,8,4,4,8]
]

real_time_list = []
real_memory_list = []
aceso_time_list = []
aceso_memory_list = []
for __target_config in target_configs:
    N,gbs,dp,pp,tp,mbs = __target_config
    target_config = f'{N}-{gbs}-{dp}-{pp}-{mbs}'
    best_result = [target_config,0,0]

    file_dir = os.path.join(src_data_path, "runtime", model_name, "time/csv")
    if os.path.exists(file_dir):
        files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
        
        max_time = 0
        min_time = 1e9
        for file_name in files:
            config = file_name.split('_')[1]
            if config != target_config:
                continue

            # print(config)
            with open(os.path.join(file_dir, file_name), 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                for row in csv_reader:
                    time = float(row[1])/1000
                    if time > max_time:
                        best_result[2] = time
                        max_time = time
                    if time < min_time:
                        best_result[1] = time
                        min_time = time
        real_time_list.append(max_time)
    else:
        print(f"{file_dir} not exist")

    if print_time:
        print(f"-------- {model_name} End-to-end throughput --------")
        print(f"Configuration\t Min_time(s)\t Max_time(s)\t Diff")
        diff = best_result[2] - best_result[1]
        print(f"{best_result[0]}\t {best_result[1]:.2f}\t\t {best_result[2]:.2f}\t\t {diff:.6f}")

    min_memory = ["Min",1e9,1e9,1e9,1e9,1e9]
    max_memory = ["Max",0,0,0,0,0]

    file_dir = os.path.join(src_data_path, "runtime", model_name, "memory/csv")
    if os.path.exists(file_dir):
        files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
        
        max_time = 0
        min_time = 1e9
        for file_name in files:
            config = file_name.split('_')[1]
            if config != target_config:
                continue

            # print(config)
            with open(os.path.join(file_dir, file_name), 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                for row in csv_reader:
                    total_mb = float(row[-5])
                    if total_mb > max_memory[1]:
                        max_memory[1] = total_mb
                    if total_mb < min_memory[1]:
                        min_memory[1] = total_mb

                    allocated = float(row[-4])
                    if allocated > max_memory[2]:
                        max_memory[2] = allocated
                    if allocated < min_memory[2]:
                        min_memory[2] = allocated

                    max_allocated = float(row[-3])
                    if max_allocated > max_memory[3]:
                        max_memory[3] = max_allocated
                    if max_allocated < min_memory[3]:
                        min_memory[3] = max_allocated

                    reserved = float(row[-2])
                    if reserved > max_memory[4]:
                        max_memory[4] = reserved
                    if reserved < min_memory[4]:
                        min_memory[4] = reserved

                    max_reserved = float(row[-1])
                    if max_reserved > max_memory[5]:
                        max_memory[5] = max_reserved
                    if max_reserved < min_memory[5]:
                        min_memory[5] = max_reserved
        real_memory_list.append(max_memory[1])
    else:
        print(f"{file_dir} not exist")
    if print_memory:
        print(f"-------- {model_name} End-to-end memory --------")
        print(f"Type\t Total(s)\t Allocated(s)\t Max_allocated\t Reserved\t Max_reserved")
        # print(f"{min_memory[0]}\t {min_memory[1]:.2f}\t\t {min_memory[2]:.2f}\t\t {min_memory[3]:.2f}\t\t {min_memory[4]:.2f}\t\t {min_memory[5]:.2f}")
        print(f"{max_memory[0]}\t {max_memory[1]:.2f}\t\t {max_memory[2]:.2f}\t\t {max_memory[3]:.2f}\t\t {max_memory[4]:.2f}\t\t {max_memory[5]:.2f}")
        print('')

    # aceso simulation
    file_dir = os.path.join(src_data_path, "aceso", model_name)
    if os.path.exists(file_dir):
        file_name = f"aceso_{model_name}_{target_config}.log"
        
        max_time,max_memory = 0,0
        with open(os.path.join(file_dir, file_name), 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if "max_time,max_memory" in line:
                    max_time,max_memory = lines[i+1].strip().split(',')
                    max_time = float(max_time)
                    max_memory = float(max_memory)
                    break
        aceso_time_list.append(max_time)
        aceso_memory_list.append(max_memory)
    else:
        print(f"{file_dir} not exist")

total_diff_time = 0
total_diff_memory = 0
lines_time = ['N,gbs,dp,pp,tp,mbs,real,estimated\n']
lines_memory = ['N,gbs,dp,pp,tp,mbs,real,estimated\n']
print("config,real_time,aceso_time,diff_time,real_memory,aceso_memory,diff_memory")
for i in range(len(target_configs)):
    N,gbs,dp,pp,tp,mbs = target_configs[i]
    lines_time.append(f'{N},{gbs},{dp},{pp},{tp},{mbs},{real_time_list[i]:.6f},{aceso_time_list[i]:.6f}\n')
    real_mem_b = real_memory_list[i] * 1024 * 1024
    aceso_mem_b = aceso_memory_list[i] * 1024 * 1024
    lines_memory.append(f'{N},{gbs},{dp},{pp},{tp},{mbs},{real_mem_b:.2f},{aceso_mem_b:.2f}\n')
    target_config = f'{N}-{gbs}-{dp}-{pp}-{mbs}'
    diff_time = abs(1 - aceso_time_list[i] / real_time_list[i]) * 100
    diff_memory = abs(1 - aceso_memory_list[i] / real_memory_list[i]) * 100
    total_diff_time += diff_time
    total_diff_memory += diff_memory
    print(f"{target_config},{real_time_list[i]:.6f},{aceso_time_list[i]:.6f},{diff_time:.2f}%,{real_memory_list[i]:.2f},{aceso_memory_list[i]:.2f},{diff_memory:.2f}%")
    
avg_diff_time = total_diff_time / len(target_configs)
avg_diff_memory = total_diff_memory / len(target_configs)
print(f'avg_diff_time: {avg_diff_time}%; avg_diff_memory: {avg_diff_memory}%')

save_dir = os.path.join(src_data_path, "plot_data", model_name)
os.makedirs(save_dir, exist_ok=True)
time_path = os.path.join(save_dir, 'Aceso-time.csv')
print(f'Save time to {time_path}')
with open(time_path, 'w') as f:
    f.writelines(lines_time)
mem_path = os.path.join(save_dir, 'Aceso-mem.csv')
print(f'Save memory to {mem_path}')
with open(mem_path, 'w') as f:
    f.writelines(lines_memory)