# Our implementation of the Atlas paper: https://arxiv.org/pdf/2411.14458
# Since there is no info about how memory or iteration time is estimated, we use the methodology followed by SAILOR

import json
import os
import math
import copy

from sailor.Planner.baselines.baseline_planner import BaselinePlanner
from sailor.Planner.sailor_planner.python_src.utils import partition_sailor
from sailor.Planner.simulations.constants import GPU_MEMORY_GB
from sailor.Planner.simulations.runtime_simulator import Simulator

class AtlasPlanner(BaselinePlanner):
    def __init__(self, profile_file, training_config_file, llm_info, fp16, objective):
        super().__init__()
        self.profile_file = profile_file
        with open(training_config_file, 'r') as f:
            self.training_config = json.load(f)

        self.global_batch_size = self.training_config['global_batch_size']
        self.num_layers = self.training_config['num_all_layers']
        self.optimizer = self.training_config['optimizer']
        self.model = self.training_config['model']
        self.model_mem_info = llm_info[self.model]
        self.objective = objective

        assert self.objective in ["throughput", "iteration_cost"]

        self.float_size = 2 if fp16 else 4
        sailor_path = os.environ.get('SAILOR_PATH')

        network_path = f'{sailor_path}/elastic-spot-ml/sailor/providers/gcp/multizone_bandwidths_het.json'
        with open(network_path, 'r') as f:
            self.network_coeffs = json.load(f)

        self.fwd_times = {}
        self.bwd_times = {}
        self.update_times = {}

        with open(profile_file, 'r') as f:
            self.profile = json.load(f)

        self.simulator = Simulator(sailor_path, self.training_config, llm_info, fp16, self.profile, zone='')


    def generate_plan(self, dp, pp, tp, mbs, regions, regions_zones, gpu_type):
        # Algorithm 1 from the paper
        part_left = pp
        region_mapping = []
        used_gpus = 0

        for region, num_gpus in regions:
            pp_gpu = num_gpus // (tp * dp)
            part_assigned = min(part_left, pp_gpu)
            for _ in range(part_assigned):
                region_mapping.append(region)
            part_left -= part_assigned
            used_gpus += part_assigned * (tp * dp)
            if part_left == 0:
                break

        if part_left==0:
            # get iteration time:
            layers_per_stage = partition_sailor(self.num_layers, pp)
            config_updated = []
            for stage in range(pp):
                region = region_mapping[stage]
                nodes = 0
                stage_config = []
                zones = list(regions_zones[region].keys())
                cur_zone_id = 0
                while nodes < dp:
                    zone = zones[cur_zone_id]
                    if regions_zones[region][zone] >= tp:
                        full_zone = f"{region}-{zone}"
                        basic_tmp = ([(gpu_type, tp, full_zone)], tp)
                        stage_config.append(basic_tmp)
                        nodes+=1
                        regions_zones[region][zone]-=tp
                    else:
                        cur_zone_id+=1
                        # TODO: any checks here?

                config_updated.append(stage_config)

            pipeline_list = [{
                'num_stages': pp,
                'layers_per_stage': layers_per_stage,
                'tmp_per_stage': config_updated,
                'dp': [dp for _ in range(pp)]
            }]

            # generate plan:
            plan = {
                'pipeline_list': pipeline_list,
                'mbs': mbs,
                'estimated_throughput': 0.0,
                'iter_time': 0.0,
                'estimated_cost': 0.0,
                'used_gpus': {gpu_type: used_gpus}
            }

            total_time, total_cost, _ = self.simulator.simulate_iteration_time(plan)
            plan['iter_time'] = total_time
            plan['estimated_throughput'] = 1.0/total_time
            plan['estimated_cost'] = total_cost

            return plan

        return {}

    def can_fit(self, pp, tp, mbs, gpu_type):
        # we follow the SAILOR logic

        # extra memory for kernel loading
        megatron_mem = 4.76 * 1e9 + 300*1e6 # the first term is for fused kernel loading, depends on GPU + platform
        if self.optimizer == 'sgd':
            # sgd saves only a copy of model parameters in fp32
            memory_multiplier_optim = 4*1  # bytes
        else:
            # this works for fp16
            memory_multiplier_optim = 4*2  # bytes - only 2 keys in state dict
        model_copy = 4  # keep model in fp32
        additional_ds_copies = 4  # Deepspeed creates 2 additional copies of the model (start of the training)
        gradients = 4
        comm = 4

        model_multiplier = memory_multiplier_optim + model_copy + gradients + comm + additional_ds_copies
        all_fit = True

        layers_per_stage = partition_sailor(self.num_layers, pp)
        for i, stage in enumerate(layers_per_stage):
            # 1. Compute mem needed for parameters
            num_params = 0
            for layer in stage:
                num_params += self.model_mem_info[str(tp)][str(layer)]['params_floats']
            mf = num_params * model_multiplier

            af_stage = 0
            for layer in stage:
                af_stage_layer = self.model_mem_info[str(tp)][str(layer)]['act_mem_floats']
                af_stage += af_stage_layer

            af_stage = af_stage * mbs * self.float_size
            af_factor = pp - i

            reserved_mem = mf + af_stage * af_factor
            mem_used = reserved_mem + megatron_mem

            gpu_mem = GPU_MEMORY_GB[gpu_type] * 1024 * 1024 * 1024
            all_fit &= (mem_used <= gpu_mem)

        return all_fit

    def get_plans_backend(self, regions, regions_zones):
        # PP, DP, TP, MBS not found!
        # we assume TP==num_gpus_per_VM + iterate through the rest

        all_plans = []
        tp = self.gpus_per_node

        for pp in range(1, self.num_layers):
            print(f"-------------------------------------- PP IS {pp}")
            for mbs in [1,2,4,8]:
                if self.can_fit(pp, tp, mbs, self.gpu_type):
                    max_dp = self.global_batch_size // mbs
                    for dp in range(1, max_dp):
                        regions_zones_copy = copy.deepcopy(regions_zones)
                        plan = self.generate_plan(dp, pp, tp, mbs, regions, regions_zones_copy, self.gpu_type)
                        if bool(plan):
                            all_plans.append(plan)
                else:
                    break
        return all_plans


    def get_sorted_plans(self, cluster_config: dict, training_config: dict):

        # 1. Do grouping

        # it is not clear if the 'Datacenters' mentioned in the paper correspond to zones or regions
        # We assume they correspond to regions, and merge all different zones of the same region
        regions_dict = {}
        regions_dict_all = {}
        self.gpus_per_node = cluster_config['gpus_per_node']
        self.gpu_type = cluster_config['gpu_type']
        cluster_config.pop('gpus_per_node')
        cluster_config.pop('gpu_type')

        for key, val in cluster_config.items():
            zone_region = key.split("_")[1]
            region = zone_region[:-2]
            zone = zone_region[-1]

            if region not in regions_dict:
                regions_dict[region] = {}
                regions_dict_all[region] = 0

            regions_dict[region][zone] = val
            regions_dict_all[region] += val

        # 2. Sort regions based on availability

        regions_sorted = [(x,y) for x,y in regions_dict_all.items()]
        regions_sorted.sort(key=lambda x:-x[1])

        print(regions_dict_all, regions_sorted)

        # 3. Get plans
        plans = self.get_plans_backend(regions_sorted, regions_dict)
        if self.objective=="throughput":
            plans.sort(key=lambda x: x['iter_time'])
        elif self.objective=="iteration_cost":
            plans.sort(key=lambda x: x['estimated_cost'])
        return plans