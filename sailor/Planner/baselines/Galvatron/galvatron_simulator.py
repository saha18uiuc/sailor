import os

from sailor.Planner.baselines.Galvatron.core.cost_model import pipeline_costmodel, TimeCostModel, MemoryCostModel
from sailor.Planner.baselines.Galvatron.config_utils import model_layer_configs
from sailor.Planner.baselines.Galvatron.core.search_engine import optimal_chunk_func_default
from sailor.Planner.baselines.Galvatron.utils import (
    read_allreduce_bandwidth_config,
    read_json_config,
    read_p2p_bandwidth_config
)


class GalvatronSimulator:
    def __init__(self, profile_file: str, cluster_config: dict, training_config: dict) -> None:
        self.global_batch_size = training_config["global_batch_size"]
        self.num_layers = training_config["num_layers"]

        self.use_pipeline_costmodel = 1
        self.optimal_chunk_func = optimal_chunk_func_default
        self.costmodel_coe = 1.0
        self.num_gpus = cluster_config['num_nodes'] * cluster_config['gpus_per_node']
        self.profile_file = profile_file

        model_configs = model_layer_configs(training_config)
        self.set_model_layer_configs(model_configs)
        self.get_profiled_model_configs()

        bandwidth_config_path = "hw_prof.json"
        self.allreduce_bandwidth, self.allreduce_comm_coe = read_allreduce_bandwidth_config(
            os.path.join(self.profile_file, bandwidth_config_path), gpu_num=self.num_gpus)
        self.p2p_bandwidth, self.p2p_comm_coe = read_p2p_bandwidth_config(os.path.join(self.profile_file, bandwidth_config_path))
        self.overlap_coe = read_json_config(os.path.join(self.profile_file, bandwidth_config_path))['overlap_coe']

        self.timecost_model_args_list = []
        for i in range(self.num_layertype):
            self.timecost_model_args_list.append({
                'parameter_size': self.param_sizes[i],
                'microbatch': False if self.use_pipeline_costmodel else True,
                'optimal_chunk_func': self.optimal_chunk_func,
                'sequence_length': self.seqlen_list[i],
                'hidden_size': self.hiddensize_list[i],
                'forward_computation_time': self.time_profiled_list[i],
                'bct_fct_coe': 2,
                'extra_overhead': 0,
                'comm_coe_dict': self.allreduce_comm_coe,
                'dp_overlap_coe': self.overlap_coe,
                'bct_overlap_coe': self.overlap_coe,
                'p2p_comm_coe_dict': self.p2p_comm_coe,
                'layer_num': self.layernum_list[i],
                'use_zero2_for_dp': 0,
                'mixed_precision': False,
                'costmodel_coe': self.costmodel_coe,
            })

    def set_model_layer_configs(self, model_layer_configs):
        if model_layer_configs is None:
            return
        self.hiddensize_list = [config['hidden_size'] for config in model_layer_configs]
        self.layernum_list = [config['layer_num'] for config in model_layer_configs]
        self.seqlen_list = [config['seq_len'] for config in model_layer_configs]
        self.num_layertype = len(self.layernum_list)

    def get_profiled_model_configs(self):
        self.time_config = read_json_config(os.path.join(self.profile_file, "compute_profile.json"))
        self.memory_config = read_json_config(os.path.join(self.profile_file, "memory_profile.json"))
        self.time_profiled_list = [self.time_config['layertype_%d' % i] for i in range(self.num_layertype)]
        self.param_sizes = [0] * self.num_layertype
        self.act_sizes = [{} for _ in range(self.num_layertype)]
        for i in range(self.num_layertype):
            layer_mem_config = self.memory_config['layertype_%d' % i]
            parameter_size = layer_mem_config['parameter_size']
            tp_activation_per_bsz_dict = layer_mem_config['tp_activation_per_bsz_dict'].copy()  # keys are diff batch sizes?
            for key, val in layer_mem_config['tp_activation_per_bsz_dict'].items():
                if len(key) < 5:
                    tp_activation_per_bsz_dict[int(key)] = val
                    del tp_activation_per_bsz_dict[key]
            self.param_sizes[i] = parameter_size
            self.act_sizes[i] = tp_activation_per_bsz_dict

        self.other_memory_pp_off = self.memory_config['other_memory_pp_off']
        self.other_memory_pp_on = {
            'first_stage': self.memory_config['other_memory_pp_on_first'], 'last_stage': self.memory_config['other_memory_pp_on_last']}
        return self.time_config, self.memory_config

    def get_memory(self, mp, dp, pp, mbs, layer_partition):
        strategy = [pp, mp, dp, {}]
        mem = MemoryCostModel(
                strategy,
                self.global_batch_size,
                self.param_sizes[0],
                self.act_sizes[0],
                self.other_memory_pp_off,
                self.other_memory_pp_on,
                mbsz=mbs,
                microbatch=True,
                optimal_chunk_func=self.optimal_chunk_func,
                model_type='gpt',
                checkpoint=0,
                use_zero2_for_dp=0,
                use_zero3_for_embed=0,
                mixed_precision=False,
                pipeline_type='pipedream_flush'
            ).get_memory_cost()
        print(mem)
        num_pp_layers = self.num_layers // pp
        mem_state = mem['enc_total'] * num_pp_layers
        max_mem_other = max(mem['other'])
        total_mem_mb = mem_state + max_mem_other
        total_mem = total_mem_mb * 1024.0 * 1024.0
        return total_mem

    def get_time(self, mp, dp, pp, mbs, layer_partition):
        chunks = [self.global_batch_size/mbs]
        partition = [self.num_layers//pp for _ in range(pp)]  # TODO: replace
        strategies = [[pp, mp, dp, {}] for _ in range(self.num_layers)]  # TODO
        batch_time = pipeline_costmodel(
            TimeCostModel,
            [self.num_layers],
            self.timecost_model_args_list,  # TODO
            strategies,
            partition,
            chunks,
            self.global_batch_size,
        )
        return batch_time
