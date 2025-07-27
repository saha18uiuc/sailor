import os
import json

from sailor.Planner.baselines.FlashFlex.src.cost_modeling import TimeCost
from sailor.Planner.baselines.FlashFlex.src.config import Config
from sailor.Planner.baselines.FlashFlex.src.globals import update_configs
from sailor.Planner.baselines.FlashFlex.src.cost_modeling import MemoryCost

class FlashFlexSimulator():
    def __init__(self, machine_config_path: str, cluster_config: dict, training_config: dict):
        self.machine_config_path = machine_config_path
        self.cluster_config = cluster_config
        self.training_config = training_config

        self.niter = 1
        self.kway = 3

        self.zone = cluster_config.pop("zone")

        with open(machine_config_path, 'r') as machine_config_file:
            self.machine_config_dict = json.load(machine_config_file)

        home_dir = os.environ.get('SAILOR_PATH')
        network_path = f'{home_dir}/sailor/sailor/providers/gcp/multizone_bandwidths.json'
        with open(network_path, 'r') as f:
            self.network_info = json.load(f)

        bandwidths = []
        for sender_gpu_type, send_data in self.cluster_config.items():
            for recv_gpu_type, recv_data in self.cluster_config.items():
                sender_gpu_count = send_data["gpus_per_node"]
                receiver_gpu_count = recv_data["gpus_per_node"]
                bw = self.network_info[self.zone][sender_gpu_type][str(sender_gpu_count)][self.zone][recv_gpu_type][str(receiver_gpu_count)][1]
                bandwidths.append(bw/1e9)

        # we get min for now
        self.inter_bw = min(bandwidths)
        print(f"INTER BW IS {self.inter_bw}")


    def get_memory(self, mp, dp, pp, mbs, tp_configs, layer_partition=None):
        npipelines = dp
        stage_sizes = [len(x) for x in layer_partition]
        configs = Config(
            self.training_config,
            self.niter,
            npipelines,
            self.kway,
            self.inter_bw,
            mbs * npipelines
        )

        for gpu_type, val in self.cluster_config.items():
            num_nodes = val["num_nodes"]
            gpus_per_node = val["gpus_per_node"]
            self.machine_config_dict["machine_amounts"][gpu_type] = {str(gpus_per_node): num_nodes}

        update_configs(configs, self.machine_config_dict)

        max_mem = 0
        start = 0
        for pipeline in range(npipelines):
            strategy = []
            for _ in range(pp):
                strategy.append(list(range(start, start+mp)))
                start += mp

            memories = [sum([configs.devices[strategy[i][j]].memory for j in range(len(strategy[i]))])
                  for i in range(len(strategy))]

            for i in range(pp):
                flags = [1 if i == 0 else 0, 1 if i == pp - 1 else 0 , 1 if i == pp - 1 else 0]
                mem_utils = MemoryCost(device_memory=memories[i], layers=stage_sizes[i], stage_strategy=strategy, configs=configs, flags=flags)
                overall_mem = mem_utils.overall_memory()
                max_mem = max(max_mem, overall_mem)

        max_mem_bytes = max_mem * 1024 * 1024 * 1024
        return max_mem_bytes


    def get_time(self, mp, dp, pp, mbs, tp_configs, layer_partition=None):

        stage_sizes = [len(x) for x in layer_partition]
        npipelines = dp

        all_pipelines = []
        start = 0
        gpu_counts_idx = {}

        configs = Config(
            self.training_config,
            self.niter,
            npipelines,
            self.kway,
            self.inter_bw,
            mbs * npipelines
        )
        for gpu_type, val in self.cluster_config.items():
            num_nodes = val["num_nodes"]
            gpus_per_node = val["gpus_per_node"]
            self.machine_config_dict["machine_amounts"][gpu_type] = {str(gpus_per_node): num_nodes}

        update_configs(configs, self.machine_config_dict)
        print(f"device_machine_map is {configs.device_machine_map}")
        print(f"devices is {[x.name for x in configs.devices]}")

        for i,_ in enumerate(configs.device_machine_map):
            gpu_name = configs.devices[i].name
            if gpu_name not in gpu_counts_idx:
                gpu_counts_idx[gpu_name] = i

        print(f"gpu_counts_idx is {gpu_counts_idx}")
        print(f"tp_configs is {tp_configs}")
        for j in range(npipelines):
            strategy = []
            for i in range(pp):
                config_i_j = tp_configs[i][j]
                gpu_type = config_i_j[0][0][0]
                start_idx = gpu_counts_idx[gpu_type]
                tp_configs_j_stage_i = list(range(start_idx, start_idx+mp))
                print(f"pipeline {j}, stage {i}, config is {config_i_j}, tp_configs_j_stage_i is {tp_configs_j_stage_i}")
                #strategy.append(list(range(start, start+mp)))
                strategy.append(tp_configs_j_stage_i)
                gpu_counts_idx[gpu_type] += mp

            pipeline = [strategy, stage_sizes, []]
            print(f"pipeline is {pipeline}")
            all_pipelines.append(pipeline)



        config_model = TimeCost(all_pipelines, configs)
        iter_time = config_model.overall_cost()
        return iter_time