import json
import os
import numpy as np
import math

from sailor.Planner.baselines.baseline_planner import BaselinePlanner
from sailor.Planner.baselines.DTFM.scheduler import GCMA, compute_data_parallel_cost, compute_pipeline_parallel_cost, get_pipelines
from sailor.Planner.sailor_planner.utils import partition_sailor

class DTFMPlanner(BaselinePlanner):
    def __init__(self, profile_file, training_config_file, llm_info, fp16):
        super().__init__()

        sailor_path = os.environ.get('SAILOR_PATH')
        network_path = f'{sailor_path}/sailor/sailor/providers/multizone_bandwidths_het.json'
        with open(network_path, 'r') as f:
            self.network_coeffs = json.load(f)

        intra_network_path = f'{sailor_path}/sailor/sailor/providers/intra_node_bandwidths.json'
        with open(intra_network_path) as f:
            self.intra_network_coeffs = json.load(f)

        with open(training_config_file, 'r') as f:
            self.training_config = json.load(f)

        self.global_batch_size = self.training_config['global_batch_size']
        self.num_layers = self.training_config['num_all_layers']

        self.model = self.training_config['model']
        self.model_mem_info = llm_info[self.model]
        self.float_size = 2 if fp16 else 4

        # dtfm only supports TP 1
        if "1" in self.model_mem_info:
            self.base_activation = self.model_mem_info["1"]["1"]["act_mem_floats"] * self.float_size # transformer
        else:
            self.base_activation = math.inf

    def get_plan_backend_config(self, pp, dp, mbs):

        print(f"Solve with pp {pp}, dp {dp}, mbs {mbs}")
        # they have a unique value for activation/stage, and gradient/stage.
        # for activation, we use the transformer one
        # for stage gradient, we use the max stage
        send_activation_size = self.base_activation * mbs
        layers_per_stage = partition_sailor(self.num_layers, pp)
        send_gradient_size = 0.0
        for stage in layers_per_stage:
            stage_size = 0
            for layer in stage:
                stage_size += self.model_mem_info["1"][str(layer)]["params_floats"]
            stage_size  *= self.float_size
            send_gradient_size = max(send_gradient_size, stage_size)

        print(f"Call GCMA to find best solution")
        candidate_partitions, all_cost_records, min_cost_records = GCMA(
            self.num_devices,
            dp,
            pp,
            self.peer_delay,
            self.peer_bandwidth,
            send_activation_size,
            send_gradient_size,
            nodes=list(range(self.num_devices)),
            population_size=10,
            trails=40,
            mode="default"
        )  # it was: population_size=100, trails: 4900
        candidate_partition_idx = np.argmin(all_cost_records)
        candidate_partition = [candidate_partitions[candidate_partition_idx][i: i + dp]
                               for i in range(0, self.num_devices, dp)]

        print(f"Solution found, compute costs. Candidate_partition is {candidate_partition}")
        data_parallel_cost = compute_data_parallel_cost(
            dp,
            send_gradient_size,
            self.peer_delay,
            self.peer_bandwidth,
            candidate_partition=candidate_partition
        )
        pipeline_parallel_cost, pipeline_parallel_path, pipeline_parallel_match = compute_pipeline_parallel_cost(
            pp,
            dp,
            self.peer_delay,
            send_activation_size,
            self.peer_bandwidth,
            candidate_partition
        )
        min_total_cost = data_parallel_cost + 2 * pipeline_parallel_cost

        candidate_pipeline = get_pipelines(
            pp, dp, self.num_devices,
            candidate_partition, pipeline_parallel_path, pipeline_parallel_match)

        print(f"Candidate pipeline is {candidate_pipeline}")
        return candidate_pipeline, min_total_cost


    def get_sorted_plans(self, cluster_config: dict, training_config: dict):
        print(cluster_config)

        self.gpu_types = cluster_config["gpu_types"]
        self.gpus_per_node = cluster_config["gpus_per_node"]
        regions = []

        cluster_config.pop("gpu_types")
        cluster_config.pop("gpus_per_node")

        # dev_id_info[i]=(zone, gpu_type) of device with id i
        dev_id_info = []
        self.num_devices = 0
        start = 0
        for key, val in cluster_config.items():
            gpu_type, zone = key.split("_")
            region = zone[:-2]
            self.num_devices += val
            for i in range(start, start+val):
                dev_id_info.append((zone, gpu_type))
            start += val
            regions.append(zone) # TODO: put region or zone here?
        print(f"num_devices is {self.num_devices}, dev_id_info is {dev_id_info}")

        self.peer_delay = np.zeros((self.num_devices, self.num_devices)) # TODO: get new prof measurements?
        self.peer_bandwidth = np.ones((self.num_devices, self.num_devices))

        for i in range(self.num_devices):
            for j in range(self.num_devices):
                zone_i, gpu_type_i = dev_id_info[i]
                zone_j, gpu_type_j = dev_id_info[j]

                node_i = i // self.gpus_per_node[gpu_type_i]
                node_j = j // self.gpus_per_node[gpu_type_j]

                if (zone_i != zone_j) or (gpu_type_i != gpu_type_j):
                    bandwidth = self.network_coeffs[zone_i][gpu_type_i]["1"][zone_j][gpu_type_j]["1"][1] # is 1 correct here?
                else:
                    if node_i == node_j:
                        bandwidth = self.intra_network_coeffs[gpu_type_i]["2"][1]
                    else:
                        bandwidth = self.network_coeffs[zone_i][gpu_type_i]["1"][zone_j][gpu_type_j]["1"][1] # is 1 correct here?

                bandwidth_gbps = bandwidth/(8*1e9)
                self.peer_bandwidth[i][j] = bandwidth_gbps

        max_pp = min(self.num_layers, self.num_devices)
        mbs_set = [1,2,4,8]
        all_configs = []

        for pp in range(2, max_pp):
            if max_pp % pp != 0:
                continue
            for mbs in mbs_set:
                if self.global_batch_size % mbs != 0:
                    continue
                max_dp = min(self.global_batch_size // mbs, self.num_devices // pp)
                # TODO: assertions here
                for dp in range(1, max_dp+1):
                    #print(pp, dp, self.num_devices)
                    if pp * dp != self.num_devices:
                        continue
                    best_config, comm_cost = self.get_plan_backend_config(pp, dp, mbs)
                    all_configs.append([pp, dp, mbs, best_config, comm_cost])

        all_configs.sort(key=lambda x: x[-1])
        all_plans = []
        for config in all_configs:
            # TODO: convert and put at plans
            print(config)
            used_gpus = {}
            pp, dp, mbs, partitions, _ = config
            layers_per_stage = partition_sailor(self.num_layers, pp)

            dp_config_updated = []
            for stage in range(pp):
                stage_config = []
                stage_partition = partitions[stage]
                for id in stage_partition:
                    zone, gpu_type = dev_id_info[id]
                    tp = self.gpus_per_node[gpu_type]
                    basic_tmp = ([(gpu_type, tp, zone)], 1)
                    stage_config.append(basic_tmp)
                    if gpu_type not in used_gpus:
                        used_gpus[gpu_type] = 0
                    used_gpus[gpu_type]+=1
                dp_config_updated.append(stage_config)

            pipeline_list = [{
                'num_stages': pp,
                'layers_per_stage': layers_per_stage,
                'tmp_per_stage': dp_config_updated,
                'dp': [dp for _ in range(pp)]
            }]
            plan = {
                'pipeline_list': pipeline_list,
                'mbs': mbs,
                'estimated_throughput': 0.0,
                'iter_time': 0.0,
                'estimated_cost': 0.0,
                'used_gpus': used_gpus
            }
            all_plans.append(plan)

        return all_plans



if __name__=="__main__":
    planner = DTFMPlanner()
    planner.get_sorted_plans()