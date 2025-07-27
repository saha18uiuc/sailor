import os
import math
import json

from sailor.Planner.baselines.Varuna.auto_config import AutoConfig

class VarunaSimulator():
    def __init__(self, profile_file: str, cluster_config: dict, training_config: dict) -> None:
        home_dir = os.environ.get('SAILOR_PATH')
        varuna_simulator_path = f"{home_dir}/sailor/sailor/Planner/baselines/Varuna/simulator"
        #if not os.path.exists(f"{varuna_simulator_path}/simulate-varuna"):
        compile_cmd = "rm -rf simulate-varuna.bin && g++ -std=c++11 simulate-varuna-main.cc generate_schedule.cc simulate-varuna.cc -o simulate-varuna.bin"
        os.system(f"cd {varuna_simulator_path} && {compile_cmd}")
        self.batch_size = training_config['global_batch_size']
        self.num_pstages = training_config['num_all_layers']
        print(f"-------------- Varuna simulator with gpus_per_node {cluster_config['gpus_per_node']}")
        self.config = AutoConfig(
            cluster_config['num_nodes'],
            cluster_config['gpus_per_node'],
            training_config['global_batch_size'],
            profile_file,
            training_config['optimizer'],
            cluster_config['gpu_type'],
            zone=cluster_config['zone'],
            gpu_memory_capacity=cluster_config["mem_per_gpu"]
        )


    def get_memory(self, mp, dp, pp, mbs, layer_partition=None):
        max_mem = self.config.get_max_mem(mbs, mp, pp, layer_partition, self.config.optimizer)
        return max_mem

    def get_time(self, mp, dp, pp, mbs, layer_partition=None):
        batch_time = self.config.process_config(pp, dp, mbs, layer_partition)
        return batch_time