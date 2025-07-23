from sailor.Planner.baselines.baseline_planner import BaselinePlanner

from sailor.Planner.baselines.Aceso.aceso_cost_model import read_profiled_time, predict_time_breakdown, update_recompute, get_reserved_memory_list, cost_model_init
from multiprocessing import Process, Queue
from sailor.Planner.baselines.Aceso.aceso_utils import *
from sailor.Planner.baselines.Aceso.aceso_prims import prims_init, finetune_dim_stage_level, finetune, get_explored_cases, reset_explored_cases
from sailor.Planner.baselines.Aceso.aceso_policy import *
from sailor.Planner.baselines.Aceso.aceso_search import *
import sailor.Planner.baselines.Aceso.aceso_search as aceso_search
import os


class AcesoPlanner(BaselinePlanner):
    def __init__(self, profiling_file_dir:str, fp16:bool = False) -> None:
        super().__init__()

        if fp16:
            self.profiling_file_dir = os.path.join(profiling_file_dir, "profiled-data-fp16/")
        else:
            self.profiling_file_dir = os.path.join(profiling_file_dir, "profiled-data-fp32/")
        self.fp16 = fp16
        aceso_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_save_path = os.path.join(aceso_dir, "aceso_logs/configs/")
        self.search_info_path = os.path.join(self.config_save_path, "csv/")
        self.top_config_path = os.path.join(self.config_save_path, "top_configs/")
        self.log_path = os.path.join(aceso_dir, "aceso_logs/search/")
        self.trend_path = os.path.join(self.log_path, "trends/")
        os.system(f"mkdir -p {self.config_save_path}")
        os.system(f"mkdir -p {self.search_info_path}")
        os.system(f"mkdir -p {self.top_config_path}")
        os.system(f"mkdir -p {self.trend_path}")

    def get_plan(self, cluster_config: dict, training_config: dict, multi_process: bool = True):
        plans = self.get_sorted_plans(cluster_config, training_config, multi_process)
        if plans:
            return plans[0]
        else:
            return None
    
    def get_sorted_plans(self, cluster_config: dict, training_config: dict, multi_process: bool = True):
        #### Hardware info ####
        num_nodes = cluster_config['num_nodes']
        gpus_per_node=cluster_config['gpus_per_node']
        memory_limit=cluster_config['mem_per_gpu'] // (1024 * 1024) # in MB

        #### Search algo parameters ####
        # time budget for searching
        budget=200
        # maximum depth of adjustments to the plan in each iteration
        max_num_hops=7
        # initial plan configuration
        if gpus_per_node == 1:
            init_config="imbalance_gpu"
        else:
            init_config="balance"

        #### Model info ####
        global_batch_size = training_config["global_batch_size"]
        model_name = training_config["model"]

        argString = f"""--model-name {model_name} \
        --global-batch-size {global_batch_size} \
        --micro-batch-size 1 2 4 8 \
        --num-nodes {num_nodes} \
        --num-gpus-per-node {gpus_per_node} \
        --memory-limit {memory_limit} \
        --config-save-path {self.config_save_path} \
        --log-path {self.log_path} \
        --profiled-time-path {self.profiling_file_dir} \
        --max-num-hops {max_num_hops} \
        --time-budget-total {budget} \
        --initial-point {init_config} \
        --do-not-use-flex-recompute \
        --num-of-saved-configs 5 """ # --print-debug-info --start-num-stages  --end-num-stages

        if not multi_process:
            argString += " --no-multi-process"
        if self.fp16:
            argString += " --fp16"

        search_init(argString, training_config)
        result_dict = search_plan()
        opt_dict = save_and_print_top_configs(result_dict, aceso_search.args)
        for plan in opt_dict:
            plan['name'] = 'Aceso'
            plan['gpu_type'] = cluster_config["gpu_type"]
            plan['num_gpus_per_node'] = cluster_config["gpus_per_node"]
            plan['used_gpus'] = {cluster_config["gpu_type"]: sum(plan["num_gpus"])}
        destory_args()
        return opt_dict
        
