import re
import os
import json
from typing import List, Tuple, Dict, Union, TYPE_CHECKING
import math

from sailor.Planner.baselines.Metis.data_loader import ProfileDataLoader
from sailor.Planner.baselines.Metis.utils import ModelConfig
from sailor.Planner.baselines.Metis.model.cost_estimator import HeteroCostEstimator
from sailor.Planner.baselines.Metis.model.activation_parameter import GPTActivationAndParam
from sailor.Planner.baselines.Metis.gpu_cluster import GPUCluster
from sailor.Planner.baselines.Metis.search_space.plan import InterStagePlan, IntraStagePlan
from sailor.Planner.baselines.Metis.model.load_balancer import DataLoadBalancer

class MetisSimulator():
    def __init__(self, profile_file: str, cluster_config: dict, training_config: dict) -> None:

        self.cluster_config = cluster_config

        data_loader = ProfileDataLoader(profile_file)
        self.profile_data, _ = data_loader.load_profile_data_all()
        assert len(self.profile_data.keys()) > 0, 'There is no profiled data at the specified path.'

        home_dir = os.environ.get('SAILOR_PATH')
        with open(f"{home_dir}/sailor/sailor/providers/gcp/intra_node_bandwidths.json", 'r') as f:
            self.intra_network_config = json.load(f)

        with open(f"{home_dir}/sailor/sailor/providers/gcp/multizone_bandwidths.json", 'r') as f:
            self.inter_network_config = json.load(f)


        self.model_config = ModelConfig(model_name=training_config["model"], num_layers=training_config["num_all_layers"],
                                        sequence_length=training_config["sequence_length"], vocab_size=training_config["vocab_size"],
                                        hidden_size=training_config["hidden_size"], attention_head_size=training_config["head_dim"])
        self.model_volume = GPTActivationAndParam(self.model_config, self.profile_data['model']['parameters'])

        self.max_profiled_tp_degree = 1
        self.max_profiled_batch_size = 1

        for key, value in self.profile_data.items():
            if 'DeviceType' in key:
                for key_spec, _ in value.items():
                    tp, mbs = re.findall("\d+", key_spec)
                    tp = int(tp)
                    mbs = int(mbs)
                    self.max_profiled_tp_degree = max(self.max_profiled_tp_degree, tp)
                    self.max_profiled_batch_size = max(self.max_profiled_batch_size, mbs)

        self.global_batch_size = training_config["global_batch_size"]
        self.num_layers = training_config["num_all_layers"]

    def get_stage_memory_demand(self, layer_partition: List[int], strategies: List[Tuple[int, int]],
                                 device_group: List[int], device_types: List[str], gbs: int, batches: int, mbs: int,
                                 mem_coef: float = 5.0):

        stage_memory = []
        print(device_group, strategies, device_types)
        for stage_id, strategy in enumerate(strategies):
            dp_deg, tp_deg = strategy
            start_rank = sum(device_group[:stage_id])
            end_rank = sum(device_group[:stage_id + 1])
            cur_device_types = [device_types[rank] for rank in range(start_rank, end_rank)]

            start_layer_id, end_layer_id = layer_partition[stage_id], layer_partition[stage_id + 1]
            cur_stage_memory_demand = 0.001
            print(f"stage_id is {stage_id}, cur_device_types is {cur_device_types}")
            if len(set(cur_device_types)) == 1:
                #bs = gbs // batches // dp_deg
                profile_memory = self.profile_data[f'DeviceType.{device_types[0]}'][f'tp{tp_deg}_bs{mbs}']['memory']
                print(tp_deg, mbs, profile_memory)
                cur_stage_memory_demand += sum(profile_memory[start_layer_id:end_layer_id]) * mem_coef
            else:
                data_load_balancer = DataLoadBalancer(self.profile_data, self.model_config)
                hetero_bs = data_load_balancer.partition_data(device_types, strategy, gbs // batches)
                for h_mbs in hetero_bs:
                    comb_h_mbs = [2 ** j for j in range(int(math.log2(h_mbs)), -1, -1) if h_mbs & 2 ** j]
                    for slice_h_mbs in comb_h_mbs:
                        profile_memory = self.profile_data[f'DeviceType.{device_types[0]}'][f'tp{tp_deg}_bs{slice_h_mbs}']['memory']
                        cur_stage_memory_demand += sum(profile_memory[start_layer_id:end_layer_id]) * mem_coef
            stage_memory.append(cur_stage_memory_demand)

        return stage_memory

    def get_memory(self, mp, dp, pp, mbs, tp_configs=None, layer_partition=None):
        cluster_zone = self.cluster_config.pop('zone')
        gpu_cluster = GPUCluster(self.cluster_config, self.inter_network_config, self.intra_network_config, cluster_zone)
        device_types = gpu_cluster.get_device_types()

        lp_list = [p[0] for p in layer_partition]
        lp_list.append(self.num_layers)

        stage_memory = self.get_stage_memory_demand(
            layer_partition=lp_list,
            strategies = [(dp,mp) for _ in range(pp)],
            device_group=[mp*dp for _ in range(pp)],
            device_types=[device_types[0].name for _ in range(mp*dp*pp)],
            gbs=self.global_batch_size,
            batches=self.global_batch_size//mbs,
            mbs=mbs,
            mem_coef=1.0
        )
        max_memory_mb = max(stage_memory)

        return max_memory_mb * 1024.0 * 1024.0

    def get_time(self, mp, dp, pp, mbs, tp_configs, layer_partition=None):
        cluster_zone = self.cluster_config.pop('zone')
        gpu_cluster = GPUCluster(self.cluster_config, self.inter_network_config, self.intra_network_config, cluster_zone)
        cost_estimator = HeteroCostEstimator(self.profile_data, self.model_config, self.model_volume, gpu_cluster, self.max_profiled_batch_size)

        device_types = gpu_cluster.get_device_types()
        print(f"----------- DEVICE TYPES IS {device_types}")
        inter_stage_plan = InterStagePlan(
            ns_idx=0,
            node_sequence=tuple(device_types),
            dg_idx=0,
            device_groups=[mp*dp for _ in range(pp)],
            num_stage=pp,
            batches = self.global_batch_size//(mbs*dp),
            gbs = self.global_batch_size
        )
        lp_list = [p[0] for p in layer_partition]
        lp_list.append(self.num_layers)
        intra_stage_plan = IntraStagePlan(
            strategies = [(dp,mp) for _ in range(pp)],
            memory_state= [],
            layer_partition=lp_list,
            num_repartition=1
        )

        # homogeneous
        rank_device_map = dict()
        idx = 0
        for i in range(pp):
            for j in range(dp):
                config_ij = tp_configs[i][j]
                gpu_type = config_ij[0][0][0]
                for k in range(mp):
                    rank_device_map[idx+k] = gpu_type.replace("-", "_")
                idx += mp
        # for device_rank in range(gpu_cluster.get_total_num_devices()):
        #     rank_device_map[device_rank] = device_types[0].name

        print(f"rank_device_map is {rank_device_map}")
        cost_ms = cost_estimator.get_cost(inter_stage_plan, intra_stage_plan.strategies,
                                                   intra_stage_plan.layer_partition, rank_device_map, mbs=mbs)
        return cost_ms/1000.0
