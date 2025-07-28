import json
import re
import time
import os
import math

from sailor.Planner.baselines.baseline_planner import BaselinePlanner
from sailor.Planner.baselines.Metis.data_loader import ProfileDataLoader
from sailor.Planner.baselines.Metis.utils import ModelConfig
from sailor.Planner.baselines.Metis.model.activation_parameter import GPTActivationAndParam
from sailor.Planner.baselines.Metis.model.cost_estimator import HeteroCostEstimator
from sailor.Planner.baselines.Metis.gpu_cluster import GPUCluster
from sailor.Planner.baselines.Metis.model.load_balancer import LayerLoadBalancer
from sailor.Planner.baselines.Metis.search_space.plan import IntraStagePlanGenerator, InterStagePlanGenerator
from sailor.Planner.baselines.Metis.model.device_group import StagePerformance
from sailor.Planner.simulations.constants import GPU_PRICES

TIME_BUDGET_SEC = 300 # added to run experiments within a reasonable time limit

class MetisPlanner(BaselinePlanner):
    def __init__(self, profile_file, training_config_path, objective="throughput", min_group_scale_variance=0.5, max_permute_len=10): # TABLE 4
        super().__init__()

        data_loader = ProfileDataLoader(profile_file, gpu_list=["A100-40", "V100-16"])
        self.profile_data, _ = data_loader.load_profile_data_all()
        self.objective = objective
        assert len(self.profile_data.keys()) > 0, 'There is no profiled data at the specified path.'

        with open(training_config_path, 'r') as f:
            self.training_config = json.load(f)

        home_dir = os.environ.get('SAILOR_PATH')
        with open(f"{home_dir}/sailor/sailor/providers/intra_node_bandwidths.json", 'r') as f:
            self.intra_network_config = json.load(f)

        with open(f"{home_dir}/sailor/sailor/providers/multizone_bandwidths_het.json", 'r') as f:
            self.inter_network_config = json.load(f)


        self.model_config = ModelConfig(model_name=self.training_config["model"], num_layers=self.training_config["num_all_layers"],
                                        sequence_length=self.training_config["sequence_length"], vocab_size=self.training_config["vocab_size"],
                                        hidden_size=self.training_config["hidden_size"], attention_head_size=self.training_config["head_dim"])

        self.model_volume = GPTActivationAndParam(self.model_config, self.profile_data['model']['parameters'])

        self.min_group_scale_variance = min_group_scale_variance
        self.max_permute_len = max_permute_len

        self.max_profiled_tp_degree = 1
        self.max_profiled_batch_size_dict = {}
        self.max_profiled_batch_size = 1024

        for key, value in self.profile_data.items():
            if 'DeviceType' in key:
                if key not in self.max_profiled_batch_size_dict:
                    self.max_profiled_batch_size_dict[key] = 1
                for key_spec, _ in value.items():
                    tp, mbs = re.findall("\d+", key_spec)
                    tp = int(tp)
                    mbs = int(mbs)
                    self.max_profiled_tp_degree = max(self.max_profiled_tp_degree, tp)
                    if tp==1:
                        self.max_profiled_batch_size_dict[key] = max(self.max_profiled_batch_size_dict[key], mbs)

        for _,mbs in self.max_profiled_batch_size_dict.items():
            self.max_profiled_batch_size = min(self.max_profiled_batch_size, mbs)

        print(f"max_profiled_batch_size is {self.max_profiled_batch_size}")

        #time.sleep(1000)

    def _get_device_placement(self, gpu_cluster, plan):
        rank_device_map = dict()

        device_types = []
        for device_type in plan.node_sequence:
            num_dtype_devices = gpu_cluster.get_num_nodes_by_device_type(device_type.name)
            device_types += ([device_type.name] * num_dtype_devices)

        for device_rank in range(gpu_cluster.get_total_num_devices()):
            rank_device_map[device_rank] = device_types[device_rank]

        devices_ranks_all = []
        for stage_id in range(len(plan.device_groups)):
            start_rank = sum(plan.device_groups[:stage_id])
            end_rank = sum(plan.device_groups[:stage_id + 1])
            device_types = [rank_device_map[rank] for rank in range(start_rank, end_rank)]
            devices_ranks_all.append(device_types)
        return devices_ranks_all


    def cost_het_cluster(self, gpu_cluster: GPUCluster, cost_estimator: HeteroCostEstimator, layer_load_balancer: LayerLoadBalancer):

        estimate_costs = []
        start_search_time = time.time()

        for inter_stage_plan in InterStagePlanGenerator(device_types=set(gpu_cluster.get_device_types()),
                                                        num_devices=gpu_cluster.get_total_num_devices(),
                                                        gbs=self.training_config["global_batch_size"],
                                                        num_layers=self.training_config["num_layers"],
                                                        variance=self.min_group_scale_variance,
                                                        max_permute_len=self.max_permute_len):

            #print(f'\n\ninter_stage_plan: {inter_stage_plan}')
            if time.time()-start_search_time > TIME_BUDGET_SEC:
                break

            stage_performance = StagePerformance(self.model_config, self.profile_data, gpu_cluster, inter_stage_plan)
            rank_device_map = stage_performance.get_device_placement()

            intra_stage_plan_generator = IntraStagePlanGenerator(inter_stage_plan, stage_performance, layer_load_balancer,
                                                                 self.max_profiled_tp_degree, self.max_profiled_batch_size)

            while intra_stage_plan_generator.has_next:
                if time.time()-start_search_time > TIME_BUDGET_SEC:
                    break
                intra_stage_plan = intra_stage_plan_generator.next()
                try:
                    #print(f"******************************** inter_stage_plan is {inter_stage_plan}, strategies is {intra_stage_plan.strategies}, rank_device_map is {rank_device_map}")
                    cost = cost_estimator.get_cost(inter_stage_plan, intra_stage_plan.strategies,
                                                   intra_stage_plan.layer_partition, rank_device_map)
                    #print(f'cost: {cost}')
                    rank_map = self._get_device_placement(gpu_cluster, inter_stage_plan)
                    #print(f"------------ rank_map is {rank_map}")
                    estimate_costs.append((inter_stage_plan.node_sequence, inter_stage_plan.device_groups,
                                           intra_stage_plan.strategies, inter_stage_plan.batches,
                                           intra_stage_plan.layer_partition, intra_stage_plan.num_repartition, cost, rank_map))
                except KeyError as e:
                    print(f'KeyError: {e}')
                #break

            #break
        return estimate_costs


    def get_stages_from_partition(self, layer_partition):
        layers_per_stage = []
        num_stages = len(layer_partition)-1
        for idx in range(num_stages):
            stage = list(range(layer_partition[idx], layer_partition[idx+1]))
            layers_per_stage.append(stage)
            if len(stage)==0:
                return []

        return layers_per_stage


    def get_sorted_plans(self, cluster_config: dict, training_config: dict):
        cluster_zone = cluster_config.pop('zone')

        #works only with powers of 2
        if cluster_config["A100-40"]["num_nodes"] == 20:
            for gpu, info in cluster_config.items():
                num_nodes = info["num_nodes"]
                cluster_config[gpu]["num_nodes"] = 16 #2**(math.floor(math.log(num_nodes, 2)))

        #cluster_config["V100-16"]["num_nodes"] = 2**(math.floor(math.log(cluster_config["V100-16"]["num_nodes"], 2)))
        print(cluster_config)

        gpu_cluster = GPUCluster(cluster_config, self.inter_network_config, self.intra_network_config, cluster_zone)

        self.max_profiled_tp_degree = 1
        self.max_profiled_batch_size_dict = {}
        self.max_profiled_batch_size = 1024

        for key, value in self.profile_data.items():
            if 'DeviceType' in key:
                if key not in self.max_profiled_batch_size_dict:
                    self.max_profiled_batch_size_dict[key] = 1
                for key_spec, _ in value.items():
                    tp, mbs = re.findall("\d+", key_spec)
                    tp = int(tp)
                    mbs = int(mbs)
                    self.max_profiled_tp_degree = max(self.max_profiled_tp_degree, tp)
                    if tp==1:
                        self.max_profiled_batch_size_dict[key] = max(self.max_profiled_batch_size_dict[key], mbs)

        for _,mbs in self.max_profiled_batch_size_dict.items():
            self.max_profiled_batch_size = min(self.max_profiled_batch_size, mbs)

        print(f"MAX TP: {self.max_profiled_tp_degree}, MAX BS: {self.max_profiled_batch_size}")

        cost_estimator = HeteroCostEstimator(self.profile_data, self.model_config, self.model_volume, gpu_cluster, self.max_profiled_batch_size)
        layer_load_balancer = LayerLoadBalancer(gpu_cluster, self.profile_data,
                                                self.model_config, self.training_config["global_batch_size"])

        estimate_costs = self.cost_het_cluster(gpu_cluster, cost_estimator, layer_load_balancer)

        print("---------------------------------------- OUTPUT RESULTS!")

        print(f'len(costs): {len(estimate_costs)}')
        sorted_result = sorted(estimate_costs, key=lambda kv: kv[6])

        sorted_list_dicts = []
        for idx, result in enumerate(sorted_result):
            #print(f'{idx + 1}, {result[6]}, {result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}')
            used_gpus = {}

            layers_per_stage = self.get_stages_from_partition(result[4])
            num_stages = len(layers_per_stage)
            if num_stages==0:
                continue
            dp_per_stage = [dp for (dp, _) in result[2]]
            tp_per_stage = [tp for (_, tp) in result[2]]

            tmp_config = []
            rank_map = result[7]
            comp_cost = 0.0
            for stage_idx in range(num_stages):
                ranks_idx = rank_map[stage_idx]
                print(f"stage_idx {stage_idx}, ranks_idx: {ranks_idx}")
                ranks_idx_set = set(ranks_idx)
                for gpu_type in ranks_idx_set:
                    gpu_type_key = gpu_type.replace("_", "-")
                    if gpu_type_key not in used_gpus:
                        used_gpus[gpu_type_key] = 0
                    print(f"{gpu_type}, {ranks_idx.count(gpu_type)}")
                    used_gpus[gpu_type_key] += ranks_idx.count(gpu_type)
                dp = dp_per_stage[stage_idx]
                tp = tp_per_stage[stage_idx]
                tmp_config_per_stage = []
                for i in range(dp):
                    gpus_tp_i = ranks_idx[i*tp:(i+1)*tp]
                    node_tp_list = []
                    tp_covered = 0
                    while tp_covered < tp:
                        for gpu_type in gpus_tp_i:
                            gpu_type = gpu_type.replace("_", "-")
                            node_tp_list.append([gpu_type, cluster_config[gpu_type]["gpus_per_node"], cluster_zone])
                            tp_covered +=  cluster_config[gpu_type]["gpus_per_node"]
                            comp_cost += cluster_config[gpu_type]["gpus_per_node"] * (GPU_PRICES[gpu_type][cluster_zone]/3600)
                    tmp_config_per_stage.append((node_tp_list, tp))
                tmp_config.append(tmp_config_per_stage)

            pipeline_list = [{
                'num_stages': num_stages,
                'layers_per_stage': layers_per_stage,
                'tmp_per_stage': tmp_config,
                'dp': dp_per_stage
            }]

            total_cost = comp_cost * result[6]

            config = {
                'pipeline_list': pipeline_list,
                'mbs': self.training_config["global_batch_size"] // (result[3] * dp_per_stage[0]), # TODO: for now, a unique mbs - change per stage
                'estimated_throughput': 1/result[6],
                'estimated_cost': total_cost,
                'iter_time': result[6],
                'used_gpus': used_gpus
            }

            print(f"****************************88 used_gpus is {used_gpus}")

            sorted_list_dicts.append(config)
            #break

        if self.objective=="iteration_cost":
            sorted_list_dicts = sorted(sorted_list_dicts, key=lambda x: x['estimated_cost'])
        return sorted_list_dicts