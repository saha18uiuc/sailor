import os
from os.path import expanduser

from sailor.Planner.baselines.baseline_planner import BaselinePlanner
from sailor.Planner.baselines.Varuna.auto_config import AutoConfig
from sailor.Planner.simulations.constants import GPU_PRICES

class VarunaPlanner(BaselinePlanner):
    def __init__(self, profiling_file, objective) -> None:
        super().__init__()
        self.profiling_file = profiling_file
        self.objective = objective

        assert self.objective in ["throughput", "iteration_cost"]

        # compile Varuna simulator if not already there
        home_dir = os.environ.get('SAILOR_PATH')

        varuna_simulator_path = f"{home_dir}/elastic-spot-ml/sailor/Planner/baselines/Varuna/simulator"
        compile_cmd = "rm -rf simulate-varuna.bin && g++ -std=c++11 simulate-varuna-main.cc generate_schedule.cc simulate-varuna.cc -o simulate-varuna.bin"
        os.system(f"cd {varuna_simulator_path} && {compile_cmd}")


    def get_sorted_plans(self, cluster_config: dict, training_config: dict):

        print(cluster_config)

        num_nodes = cluster_config['num_nodes']
        gpus_per_node = cluster_config['gpus_per_node']
        mem_per_gpu = cluster_config['mem_per_gpu']
        gpu_type = cluster_config['gpu_type']
        zone = cluster_config['zone']

        global_batch_size = training_config['global_batch_size']

        auto = AutoConfig(num_nodes, gpus_per_node, global_batch_size,
                          self.profiling_file, training_config['optimizer'], gpu_type, zone=cluster_config['zone'], gpu_memory_capacity=mem_per_gpu)
        auto.generate_plan()
        gpus_available = num_nodes * gpus_per_node

        sorted_list = auto.get_sorted_list()

        sorted_list_dicts = []
        for plan in sorted_list:
            pp = plan[0]
            dp = gpus_available // pp
            batch_time = plan[2]
            used_gpus = dp*pp
            comp_cost = used_gpus * (GPU_PRICES[gpu_type][zone]/3600)
            total_cost = comp_cost * batch_time
            plan_dict = {
                'mbs': plan[1],
                'D': dp,
                'P': pp,
                'T': 1,
                'gpu_type': gpu_type,
                'num_gpus_per_node': gpus_per_node,
                'used_gpus': {gpu_type: used_gpus},
                'zone': cluster_config['zone'],
                'estimated_throughput': 1/batch_time,
                'estimated_cost': total_cost
            }
            sorted_list_dicts.append(plan_dict)

        if self.objective == "iteration_cost":
            sorted_list_dicts = sorted(sorted_list_dicts, key=lambda x: x['estimated_cost'])

        return sorted_list_dicts
