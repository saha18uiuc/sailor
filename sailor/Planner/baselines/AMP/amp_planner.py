import torch
import numpy as np
import copy
import json
import os

from sailor.Planner.baselines.baseline_planner import BaselinePlanner
from sailor.Planner.baselines.AMP.cost_amp import AMP
from sailor.Planner.baselines.AMP.sa import amp_no_placement_strategy
from sailor.Planner.simulations.constants import GPU_PRICES

class AMPPlanner(BaselinePlanner):
    def __init__(self, profile_dir, objective="throughput", latest_from_trace=False) -> None:
        super().__init__()
        self.profile_dir = profile_dir
        self.last_valid_M = 0
        self.last_valid_N = 0
        self.last_valid_config = {}
        self.latest_from_trace = latest_from_trace
        self.objective = objective

        assert self.objective in ["throughput", "iteration_cost"]

        home_dir = os.environ.get('SAILOR_PATH')
        with open(f'{home_dir}/sailor/sailor/providers/multizone_bandwidths_het.json', 'r') as f:
            self.network_info = json.load(f)

        with open(f'{home_dir}/sailor/sailor/Planner/llm_info.json') as f:
            self.llm_info = json.load(f)

    def sorted_plans_backend(self, cluster_config: dict, training_config_or: dict):

        training_config = copy.deepcopy(training_config_or)

        global_batch_size = training_config['global_batch_size']
        zone = cluster_config['zone']
        num_nodes = 0
        gpus_per_node = 0
        self.network_bandwidth = 500*1e9

        cluster_config.pop('zone')
        for gpu_type,info in cluster_config.items():
            num_nodes += info['num_nodes']
            gpus_per_node = info['gpus_per_node'] # should be the same for all GPU types
            gpu_bw = self.network_info[zone][gpu_type][str(gpus_per_node)][zone][gpu_type][str(gpus_per_node)][1]
            self.network_bandwidth = min(self.network_bandwidth, gpu_bw)

        configs = []
        if global_batch_size % (num_nodes*gpus_per_node) != 0:
            print("AMP cannot process this scenario - resort to a different solution")
            if self.latest_from_trace:
                print("Go to latest config")
                return self.last_valid_config
            else:
                for i in range(num_nodes):
                    node_count = num_nodes-i
                    if global_batch_size % (gpus_per_node*node_count) == 0:
                        break
                num_nodes = node_count
                print(f"Rerun with M={gpus_per_node}, N={node_count}")

        # needed by AMP
        for k, _ in training_config.items():
            if k in ["hidden_size", "sequence_length", "num_layers", "vocab_size"]:
                training_config[k] = torch.tensor(training_config[k], dtype=float)
        cluster_info = {}
        for i in range(num_nodes):
            # it gets network bandwidth in floats
            cluster_info[i] = [torch.tensor(
                [self.network_bandwidth / 4]).float(), np.inf]

        model = AMP(training_config, cluster_config, self.profile_dir, self.llm_info)  # includes the profiling
        known = None
        iter_count = 0
        #print(model.profile_cost)

        # # Estimating best configurations
        while True:
            ret = amp_no_placement_strategy(
                M=gpus_per_node, N=num_nodes, gbs=global_batch_size, known=known)
            if ret is None:
                break
            else:
                h, w, mbs, known = ret
                if str(h) not in model.profile_cost:
                    continue
                pp = gpus_per_node*num_nodes/(h*w)
                if (pp > training_config["num_all_layers"]):
                        continue
                oth = {"mp_deg": torch.ones(
                    1,)*h, "dp_deg": torch.ones(1,)*w, "pp_deg": torch.ones(1,)*(pp)}
                fake_config = np.ones((gpus_per_node, num_nodes)) * (-1)
                model_args = (fake_config, global_batch_size, mbs,
                              cluster_info, training_config, oth)

                with torch.no_grad():
                    rank_map, ds_partition, cost_time = model(model_args)

                configs.append((mbs, oth, rank_map, ds_partition, cost_time))

            iter_count += 1
            if iter_count % 10 == 0:
                print(f"AMP finish {iter_count} iterations")

        # print(f"Sorted configs is {sorted_configs}")
        return configs

    def get_sorted_plans(self, cluster_config: dict, training_config: dict):

        zone = cluster_config['zone']
        sorted_configs = self.sorted_plans_backend(cluster_config, training_config)
        sorted_list_dicts = []
        for plan in sorted_configs:
            print(plan)

            dp = int(plan[1]['dp_deg'].item()),
            pp = int(plan[1]['pp_deg'].item()),
            tp = int(plan[1]['mp_deg'].item()),
            dp = dp[0]
            pp = pp[0]
            tp = tp[0]
            ds_partition = plan[3]

            layers_per_stage = []
            for i in range(len(ds_partition)-1):
                layers = list(range(ds_partition[i], ds_partition[i+1]))
                layers_per_stage.append(layers)

            #print(pp, ds_partition ,layers_per_stage)

            all_gpus_config = {}
            for gpu_type, info in cluster_config.items():
                all_gpus_config[gpu_type] = info['num_nodes'] * info['gpus_per_node']
            all_gpus_list = list(all_gpus_config.keys())
            cur_gpu = all_gpus_list[0]
            cur_gpu_idx = 0

            config = []
            used_gpus = {
                cur_gpu: 0
            }
            comp_cost = 0.0
            for _ in range(pp):
                stage_config = []
                for _ in range(dp):
                    if all_gpus_config[cur_gpu] < tp:
                        cur_gpu_idx+=1
                        cur_gpu=all_gpus_list[cur_gpu_idx]
                        used_gpus[cur_gpu] = 0

                    tp_config = ([(cur_gpu, cluster_config[cur_gpu]['gpus_per_node'], zone)], tp)
                    comp_cost += (tp * (GPU_PRICES[cur_gpu][zone]/3600))
                    stage_config.append(tp_config)
                    used_gpus[cur_gpu] += tp
                    all_gpus_config[cur_gpu] -= tp
                config.append(stage_config)

            pipeline_list = [
                {
                    'num_stages': pp,
                    'layers_per_stage': layers_per_stage,
                    'tmp_per_stage': config,
                    'dp': [dp for _ in range(pp)]
                }
            ]

            iter_time = plan[4].item()
            iter_cost = comp_cost * iter_time # no comm cost
            plan_dict = {
                'pipeline_list': pipeline_list,
                'mbs': plan[0],
                'iteration_time': iter_time,
                'estimated_cost': iter_cost,
                'estimated_throughput': 1/iter_time,
                'used_gpus': used_gpus,
                'original': plan
            }

            sorted_list_dicts.append(plan_dict)

        if self.objective=="throughput":
            sorted_list_dicts = sorted(sorted_list_dicts, key=lambda kv: kv['iteration_time'])
        elif self.objective=="iteration_cost":
            sorted_list_dicts = sorted(sorted_list_dicts, key=lambda kv: kv['estimated_cost'])

        # sorted_list_cost_only = [(x['estimated_cost'], x['iteration_time']) for x in sorted_list_dicts]
        # print(f"sorted_list_dicts, cost only  is {sorted_list_cost_only}")

        return sorted_list_dicts
