# Extracted AMP simulator
import copy
import torch
import numpy as np
import json
import os

from sailor.Planner.baselines.AMP.cost_amp import AMP

class AMPSimulator():
    def __init__(self, profile_dir: str, cluster_config: dict, training_config_or: dict) -> None:
        self.training_config = copy.deepcopy(training_config_or)

        home_dir = os.environ.get('SAILOR_PATH')
        with open(f'{home_dir}/elastic-spot-ml/sailor/providers/gcp/multizone_bandwidths_het.json', 'r') as f:
            self.network_info = json.load(f)

        with open(f'{home_dir}/elastic-spot-ml/sailor/Planner/llm_info.json') as f:
            self.llm_info = json.load(f)

        self.global_batch_size = self.training_config['global_batch_size']
        # needed by AMP
        for k, _ in self.training_config.items():
            if k in ["hidden_size", "sequence_length", "num_layers", "vocab_size"]:
                self.training_config[k] = torch.tensor(self.training_config[k], dtype=float)

        zone = cluster_config['zone']
        self.num_nodes = 0
        self.gpus_per_node = 0
        self.network_bandwidth = 500*1e9

        cluster_config.pop('zone')
        for gpu_type,info in cluster_config.items():
            self.num_nodes += info['num_nodes']
            self.gpus_per_node = info['gpus_per_node'] # should be the same for all GPU types
            gpu_bw = self.network_info[zone][gpu_type][str(self.gpus_per_node)][zone][gpu_type][str(self.gpus_per_node)][1]
            self.network_bandwidth = min(self.network_bandwidth, gpu_bw)

        self.cluster_info = {}
        for i in range(self.num_nodes):
            # it gets network bandwidth in floats
            self.cluster_info[i] = [torch.tensor(
                [self.network_bandwidth / 4]).float(), np.inf]

        self.model = AMP(self.training_config, cluster_config, profile_dir, self.llm_info)  # includes the profiling

    def get_memory(self, mp, dp, pp, mbs, layer_partition):
        return 0

    def get_time(self, mp, dp, pp, mbs, tp_configs, layer_partition):
        partition = [len(x) for x in layer_partition]
        print(layer_partition, partition)
        oth = {
            "mp_deg": torch.ones(1,)*mp,
            "dp_deg": torch.ones(1,)*dp,
            "pp_deg": torch.ones(1,)*pp
        }

        print(f"Run with {self.num_nodes} nodes, {self.gpus_per_node} gpus_per_node")
        fake_config = np.ones((self.gpus_per_node, self.num_nodes)) * (-1)
        model_args = (fake_config, self.global_batch_size, mbs,
                              self.cluster_info, self.training_config, oth)

        with torch.no_grad():
            _, _, cost_time = self.model(model_args, partition)
        print(f"Cost time is {cost_time}")
        return cost_time.item()
