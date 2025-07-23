import numpy as np
import json
import os

from sailor.Planner.baselines.FlashFlex.src.cost_modeling import TimeCost

from sailor.Planner.baselines.FlashFlex.src.evaluate import throughput
from sailor.Planner.baselines.FlashFlex.src.graph import construct_graph
from sailor.Planner.baselines.FlashFlex.src.initialize import initialize
from sailor.Planner.baselines.FlashFlex.src.partitioner import partition_pipeline, partitioner
from sailor.Planner.baselines.baseline_planner import BaselinePlanner

from sailor.Planner.baselines.FlashFlex.src.config import Config
from sailor.Planner.baselines.FlashFlex.src.globals import update_configs

from sailor.Planner.simulations.constants import GPU_PRICES

class FlashFlexPlanner(BaselinePlanner):
    def __init__(self, machine_config_path, niter=100, npipeline=1, kway=3, objective="throughput"):
        super().__init__()
        self.machine_config_path = machine_config_path
        self.niter = niter
        self.npipeline = npipeline
        self.kway = kway
        self.objective = objective

        assert self.objective in ["throughput", "iteration_cost"]

        with open(machine_config_path, 'r') as machine_config_file:
            self.machine_config_dict = json.load(machine_config_file)

        home_dir = os.environ.get('SAILOR_PATH')
        network_path = f'{home_dir}/elastic-spot-ml/sailor/providers/gcp/multizone_bandwidths_het.json'
        with open(network_path, 'r') as f:
            self.network_info = json.load(f)

    def get_layers(self, partioning):

        layers_per_stage = []
        start = 1
        num_stages = len(partioning)
        for i,part in enumerate(partioning):
            if part==0:
                return []
            if i==0:
                layers_this_stage = [0]
            else:
                layers_this_stage = []
            layers_this_stage += list(range(start, start+part))
            start += part
            if i==num_stages-1:
                layers_this_stage.append(start)
            layers_per_stage.append(layers_this_stage)
        print(f"partitioning is {partioning}, layers_per_stage is {layers_per_stage}")
        return layers_per_stage

    def get_plans_backend(self):

        min_npipeline = 1
        max_npipeline = len(self.configs.devices) // 5 # from: https://github.com/Relaxed-System-Lab/HexiScale/blob/main/experimental/main.py#L52

        all_plans = []

        for npipeline in range(min_npipeline, max_npipeline+1):
            self.npipeline = npipeline

            # always keep MB to 1
            self.configs.B = self.configs.GLB_B // self.npipeline
            self.configs.GLB_MB = self.micro_bsz_base * self.npipeline
            self.configs.MB = self.configs.GLB_MB // self.npipeline
            self.configs.N_MB = self.configs.GLB_B // self.configs.GLB_MB

            print(f"-------------------------------------------- npipeline is {self.npipeline}, per-pipeline B is {self.configs.B}, MB is {self.configs.MB}")
            self.configs.K = initialize(self.configs.param, (1, self.npipeline))

            # ------------- Step 1
            # construct with reverse of bandwidth
            G = construct_graph(self.configs.specs[0], self.configs.specs[3])
            G.options = self.configs.options

            self.parts = partitioner(G, self.npipeline)
            assert self.npipeline == max(self.parts) + 1

            print("Initial partition results:", self.parts)


            # ------------- Step 2
            # reconstruct with bandwidth
            G = construct_graph(self.configs.specs[0], self.configs.specs[1])
            G.options = self.configs.options

            next_parts = None
            optimal = None
            optimal_npipeline = None

            # ------------- Step 2
            all_pipelines, all_sub_recovered_parts = partition_pipeline(G, self.parts, self.npipeline, self.configs)
            print(f"next_parts is {next_parts}")

            if next_parts is not None:
                next_all_pipelines, next_all_sub_recovered_parts = partition_pipeline(G, next_parts, self.npipeline, self.configs)
                if throughput(next_all_pipelines, self.configs) > throughput(all_pipelines, self.configs):
                    self.parts = next_parts
                    all_sub_recovered_parts = next_all_sub_recovered_parts

            if (optimal is None) or (throughput(all_pipelines, self.configs) > throughput(optimal, self.configs)):
                optimal = all_pipelines
                optimal_npipeline = self.npipeline

            print(len(all_pipelines), len(all_pipelines[0]))
            print(f"Throughput is {throughput(all_pipelines, self.configs)}, all_pipelines is {all_pipelines}")

            config_model = TimeCost(all_pipelines, self.configs)
            iter_time = config_model.overall_cost()
            plan_config = [all_pipelines, iter_time]
            all_plans.append(plan_config)

        all_plans.sort(key=lambda x:x[1])
        return all_plans

    def get_sorted_plans(self, cluster_config: dict, training_config: dict):
        print(f"cluster_config is {cluster_config}")
        self.zone = cluster_config.pop("zone")

        bandwidths = []
        for sender_gpu_type, send_data in cluster_config.items():
            for recv_gpu_type, recv_data in cluster_config.items():
                sender_gpu_count = send_data["gpus_per_node"]
                receiver_gpu_count = recv_data["gpus_per_node"]
                bw = self.network_info[self.zone][sender_gpu_type][str(sender_gpu_count)][self.zone][recv_gpu_type][str(receiver_gpu_count)][1]
                bandwidths.append(bw/1e9)

        # we get min for now
        inter_bw = min(bandwidths)
        print(f"INTER BW IS {inter_bw}")

        self.micro_bsz_base = 1
        self.npipeline = 1
        micro_bsz = self.micro_bsz_base * self.npipeline
        self.configs = Config(
            training_config,
            self.niter,
            self.npipeline,
            self.kway,
            inter_bw,
            micro_bsz
        )

        for gpu_type, val in cluster_config.items():
            num_nodes = val["num_nodes"]
            gpus_per_node = val["gpus_per_node"]
            self.machine_config_dict["machine_amounts"][gpu_type] = {str(gpus_per_node): num_nodes}

        update_configs(self.configs, self.machine_config_dict)
        plans = self.get_plans_backend()
        configs = []

        for plan_config in plans:
            plan, estimated_iter_time = plan_config
            pipeline_list = []
            used_gpus = {}
            valid_config = True
            comp_cost = 0.0
            for id, pipeline in enumerate(plan):
                #print(f"********************************************* Pipeline {id}, pipeline is {pipeline}")
                num_stages = len(pipeline[1])
                layers_per_stage = self.get_layers(pipeline[1])
                if len(layers_per_stage)==0:
                    valid_config = False
                    break
                #print(num_stages, layers_per_stage)

                dp = 1
                tmp_configs = []
                for i in range(num_stages):
                    devices_on_stage = pipeline[0][i]
                    tmp_stage = []
                    tmp = len(devices_on_stage)
                    #print(f"--------- devices_on_stage is {devices_on_stage}")
                    vm_list = []
                    for dev in devices_on_stage:
                        gpu_id = dev # TODO: might be diff devices
                        gpu_type = self.configs.devices[gpu_id].name
                        if gpu_type not in used_gpus:
                            used_gpus[gpu_type] = 0
                        used_gpus[gpu_type] += 1
                        comp_cost += (GPU_PRICES[gpu_type][self.zone]/3600)
                        gpus_per_node = int(list(self.machine_config_dict["machine_amounts"][gpu_type].keys())[0])
                        vm_list.append((gpu_type, gpus_per_node, self.zone))
                    tmp_stage = [(vm_list,tmp)]

                    tmp_configs.append(tmp_stage)

                pipeline_def = {
                    'num_stages': num_stages,
                    'tmp_per_stage': tmp_configs,
                    'layers_per_stage': layers_per_stage,
                    'dp': [dp for _ in range(num_stages)]
                }
                pipeline_list.append(pipeline_def)

            if valid_config:
                estimated_iter_cost = comp_cost * estimated_iter_time
                configs.append({
                    'mbs': micro_bsz,
                    'pipeline_list': pipeline_list,
                    'estimated_throughput': 1/estimated_iter_time,
                    'iter_time': estimated_iter_time,
                    'estimated_cost': estimated_iter_cost,
                    'used_gpus': used_gpus
                })

        if self.objective=="iteration_cost":
            configs = sorted(configs, key=lambda kv: kv['estimated_cost'])
        return configs