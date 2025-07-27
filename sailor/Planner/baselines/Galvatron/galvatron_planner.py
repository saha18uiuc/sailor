from argparse import Namespace
import json

from sailor.Planner.baselines.Galvatron.core.search_engine import GalvatronSearchEngine, strategy2config
from sailor.Planner.baselines.baseline_planner import BaselinePlanner
from sailor.Planner.baselines.Galvatron.config_utils import model_layer_configs, model_name
from sailor.Planner.simulations.constants import GPU_PRICES

class GalvatronPlanner(BaselinePlanner):
    def __init__(self, training_config, planner_config_dict, objective):
        super().__init__()
        path = "/home/fstrati/sailor/sailor/Planner/baselines/Galvatron/test"
        with open(training_config, 'r') as f:
            config = json.load(f)

        self.objective = objective
        assert self.objective in ["throughput", "iteration_cost"]

        args = Namespace(
            set_model_config_manually=0,
            set_layernum_manually=0,
            num_nodes=0, # set during eval
            num_gpus_per_node=0, # set during eval
            memory_constraint=0, # set during eval
            min_bsz=config["global_batch_size"],
            max_bsz=config["global_batch_size"],
            settle_bsz=config["global_batch_size"],
            settle_chunk=-1,
            search_space='full',
            disable_dp=0,
            disable_tp=0,
            disable_pp=0,
            disable_sdp=1,
            disable_ckpt=1,
            disable_tp_consec=0,
            max_tp_deg=4,
            max_pp_deg=config['num_layers'],
            default_dp_type='ddp',
            embed_sdp=0,
            mixed_precision='fp32',
            pipeline_type='pipedream_flush',
            use_pipeline_costmodel=1,
            costmodel_coe=1.0)

        self.search_engine = GalvatronSearchEngine(args)
        self.search_engine.set_search_engine_info(planner_config_dict, model_layer_configs(config), model_name(config))
        #self.search_engine.set_model_type('gpt')


    def get_sorted_plans(self, cluster_config: dict, training_config: dict):

        num_nodes = cluster_config["num_nodes"]
        gpus_per_node = cluster_config["gpus_per_node"]
        num_gpus = num_nodes * gpus_per_node
        new_num_gpus = num_gpus

        if training_config["global_batch_size"] % num_gpus != 0:
            for i in range(num_nodes):
                new_num_gpus -= gpus_per_node
                if training_config["global_batch_size"] % new_num_gpus == 0:
                    break

            num_nodes = new_num_gpus // gpus_per_node
            print(f"Invalid number of GPUs, Rerun with {num_nodes} nodes")

        self.search_engine.set_cluster_info(
            num_nodes,
            gpus_per_node,
            int(cluster_config["mem_per_gpu"]/(1024 * 1024)) # needs to be int
        )

        self.search_engine.initialize_search_engine()
        results = self.search_engine.parallelism_optimization()
        results = sorted(results, key=lambda d: d['throughput'], reverse=True)

        sorted_list_dicts = []
        gpu_type = cluster_config['gpu_type']
        zone = cluster_config['zone']

        # Convert
        for result in results:
            # Assume homogeneous for now:
            dp = result["dp_sizes_enc"][0]
            tp = result["tp_sizes_enc"][0]
            pp = result["pp_deg"]
            print(dp, tp, result['throughput'])
            mbs = training_config["global_batch_size"]//(result["chunks"] * dp)
            iter_time = 1/result['throughput']

            comp_cost = (dp*pp*tp)*(GPU_PRICES[gpu_type][zone]/3600)
            total_cost = comp_cost * iter_time
            res_dict = {
                'mbs': mbs,
                'D': dp,
                'P': pp,
                'T': tp,
                'gpu_type': cluster_config['gpu_type'],
                'num_gpus_per_node': min(tp, 4), # up to 4 GPUs per node
                'used_gpus': {gpu_type: dp * pp * tp},
                'zone': zone,
                'estimated_throughput': result['throughput'],
                'estimated_cost': total_cost
            }
            sorted_list_dicts.append(res_dict)
            print("-----------------------------------")

        if self.objective == "iteration_cost":
            sorted_list_dicts = sorted(sorted_list_dicts, key=lambda x: x['estimated_cost'])

        return sorted_list_dicts
