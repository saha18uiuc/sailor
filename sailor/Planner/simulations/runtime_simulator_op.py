import json
import math
from os.path import expanduser

from sailor.Planner.simulations.constants import GPU_MEMORY_GB
from sailor.Planner.simulations.utils import models

from sailor.Planner.sailor_planner.python_src.utils import calculate_exec_per_stage
from sailor.Planner.sailor_planner.python_src.utils import partition_uniform
from sailor.Planner.sailor_planner.constants import MEMORY_BUCKET_DEEPSPEED_SIZE
from sailor.profiling.profile_utils import estimate_send_time, estimate_ar_time

class Pipeline:
    def __init__(self, gpu_type: str, gpus_per_node: int, num_stages: int, ops_per_stage: list[list[int | str]], num_dp: list[list[int]], num_tp: list[list[int]], algo: list[list[int | str]]):
        self.gpu_type = gpu_type
        self.gpus_per_node = gpus_per_node
        self.num_stages = num_stages
        self.ops_per_stage = ops_per_stage
        self.num_dp = num_dp
        self.num_tp = num_tp
        self.algo = algo

    def print_info(self):
        print(f"GPU type: {self.gpu_type}")
        print(f"GPUs_per_node: {self.gpus_per_node}")
        print(f"Num stages: {self.num_stages}")
        print(f"Operators_per_stage: {self.ops_per_stage}")
        print(f"TP size: {self.num_tp}")
        print(f"DP size: {self.num_dp}")
        print(f"Algorithm type: {self.algo}")


class Plan:
    def __init__(self, pipeline: Pipeline, mbs: int, homogeneous_pipelines: bool) -> None:
        self.pipeline = pipeline
        self.mbs = mbs # base batch size
        self.homogeneous_pipelines = homogeneous_pipelines

    def to_dict(self):
        plan_to_dict = {}
        plan_to_dict["mbs"] = self.mbs

        pipeline_dict = {
            "gpu_type": self.pipeline.gpu_type,
            "gpus_per_node": self.pipeline.gpus_per_node,
            "num_stages": self.pipeline.num_stages,
            "ops_per_stage": self.pipeline.ops_per_stage,
            "dp_degrees": self.pipeline.num_dp,
            "tp_degrees": self.pipeline.num_tp,
            "algo": self.pipeline.algo
        }

        plan_to_dict['pipeline'] = pipeline_dict

        return plan_to_dict


def convert_homogeneous(plan: dict, training_config: dict):
    # Homogeneous Plans: Aceso, nnScaler
    num_stages = plan['num_stages']
    if plan["name"] == "Aceso":
        num_ops_per_stages = plan['num_ops_in_each_stage']
        ops_per_stage = []
        start_op_index = 0
        for i in range(num_stages):
            ops_per_stage.append(list(range(start_op_index, start_op_index + num_ops_per_stages[i])))
            start_op_index += num_ops_per_stages[i]
    elif plan["name"] == "nnScaler":
        ops_per_stage = plan['ops_per_stage']
    else:
        raise NotImplementedError

    tp_degree_list = plan['model_parallel_size_of_each_op']
    dp_degree_list = plan['data_parallel_size_of_each_op']
    algo_list = plan['algo_of_each_op']

    pipeline = Pipeline(plan['gpu_type'], plan['num_gpus_per_node'], num_stages, ops_per_stage, dp_degree_list, tp_degree_list, algo_list)
    homogeneous_pipelines = True

    return Plan(pipeline, plan['micro_batch_size'], homogeneous_pipelines)


class SimulatorOP():
    def __init__(self, training_config: dict, llm_info: dict, fp16: bool, profiles: dict) -> None:
        self.training_config = training_config
        self.optimizer = self.training_config["optimizer"]

        self.model = training_config['model']
        self.model_mem_info = llm_info[self.model] # index: tp, algo, op
        self.model_config = models[self.model]

        self.fp16 = fp16
        self.float_size = 2 if fp16 else 4
        self.global_batch_size = training_config['global_batch_size']

        self.num_layers = training_config['num_all_layers']
        self.profiles = profiles[self.model] # index: gpu_type, mb, tp, algo, op

        # single-zone
        home = expanduser("~")
        with open(f'{home}/sailor/sailor/providers/gcp/network_coeffs.json', 'r') as f:
            network_coeffs_dict = json.load(f)["inter"]

        self.network_coeffs = {}
        for gpu_type, coeffs_per_gpu in network_coeffs_dict.items():
            for num_gpus, coeffs in coeffs_per_gpu.items():
                self.network_coeffs[(gpu_type, int(num_gpus))] = coeffs

    def check_config_fits(self, plan_dict: dict):
        plan = convert_homogeneous(plan_dict, self.training_config)

        base_batch_size = plan.mbs
        gpu_type = plan.pipeline.gpu_type
        megatron_mem = 3 * (1024**3)  # extra mem needed by megatron, shown in nvidia-smi, but not as part of torch mem
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

        model_multiplier = memory_multiplier_optim + model_copy + gradients + comm
        all_fit = True

        # For the pipeline, make sure it fits in memory:
        for i in range(plan.pipeline.num_stages):
            ops = plan.pipeline.ops_per_stage[i]
            dp = plan.pipeline.num_dp[i]
            tp = plan.pipeline.num_tp[i]
            algo = plan.pipeline.algo[i]
            mbs = [base_batch_size // dp[j] for j in range(len(ops))]
            # 1. Compute mem needed for parameters
            num_params = 0
            for j in range(len(ops)):
                num_params += self.model_mem_info[str(tp[j])][str(algo[j])][str(ops[j])]['params_floats']
            mf = num_params * model_multiplier

            # 2. Compute mem needed for activations
            af_stage = mbs[0] * self.model_mem_info[str(tp[0])][str(algo[0])][str(ops[0])]['act_input_floats']
            for j in range(len(ops)):
                af_stage += mbs[j] * self.model_mem_info[str(tp[j])][str(algo[j])][str(ops[j])]['act_mem_floats']

            af_stage = af_stage * self.float_size
            mem_used = mf + af_stage * (plan.pipeline.num_stages - i) + megatron_mem

            gpu_mem = GPU_MEMORY_GB[gpu_type] * 1024 * 1024 * 1024
            all_fit &= (mem_used <= gpu_mem)
            bytes_per_mb = 1024*1024
            mem_used_mb = mem_used / bytes_per_mb
            gpu_mem_mb = gpu_mem / bytes_per_mb
            mf_mb = mf / bytes_per_mb
            af_stage_mb = af_stage / bytes_per_mb
            total_af_stage = af_stage * (plan.pipeline.num_stages - i) / bytes_per_mb
            print(f"stage: {i}; mf: {mf_mb}MB; af_stage: {af_stage_mb}MB; total_af: {total_af_stage}")
            print(f"mem_used: {mem_used_mb}MB; gpu_mem: {gpu_mem_mb}; all_fit: {all_fit}")

        # print(f"all fit is {all_fit}")
        return all_fit

    def get_max_memory(self, plan_dict: dict):
        plan = convert_homogeneous(plan_dict, self.training_config)

        base_batch_size = plan.mbs
        gpu_type = plan.pipeline.gpu_type
        megatron_mem = 3 * (1024**3)  # extra mem needed by megatron, shown in nvidia-smi, but not as part of torch mem
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

        model_multiplier = memory_multiplier_optim + model_copy + gradients + comm

        max_memory = 0
        # For the pipeline, make sure it fits in memory:
        for i in range(plan.pipeline.num_stages):
            ops = plan.pipeline.ops_per_stage[i]
            dp = plan.pipeline.num_dp[i]
            tp = plan.pipeline.num_tp[i]
            algo = plan.pipeline.algo[i]
            mbs = [base_batch_size // dp[j] for j in range(len(ops))]
            # 1. Compute mem needed for parameters
            num_params = 0
            for j in range(len(ops)):
                num_params += self.model_mem_info[str(tp[j])][str(algo[j])][str(ops[j])]['params_floats']
            mf = num_params * model_multiplier

            # 2. Compute mem needed for activations
            af_stage = mbs[0] * self.model_mem_info[str(tp[0])][str(algo[0])][str(ops[0])]['act_input_floats']
            for j in range(len(ops)):
                af_stage += mbs[j] * self.model_mem_info[str(tp[j])][str(algo[j])][str(ops[j])]['act_mem_floats']

            af_stage = af_stage * self.float_size
            mem_used = mf + af_stage * (plan.pipeline.num_stages - i) + megatron_mem

            if mem_used > max_memory:
                max_memory = mem_used

        return max_memory / (1024*1024) # in MB

    def get_time_per_pipeline(
        self,
        pipeline: Pipeline,
        base_batch_size: int,
        num_micro_batches: int,
    ):
        profiled_time = self.profiles[pipeline.gpu_type]
        comp_time_per_stage = []
        update_time_per_stage = []

        for i in range(pipeline.num_stages):
            ops = pipeline.ops_per_stage[i]
            dp = pipeline.num_dp[i]
            tp = pipeline.num_tp[i]
            algo = pipeline.algo[i]
            mbs = [base_batch_size // dp[j] for j in range(len(ops))]

            # compute per stage
            update = 0.0
            fwd_bwd = 0.0
            for j in range(len(ops)):
                fwd_bwd += profiled_time[str(mbs[j])][str(tp[j])][str(algo[j])][str(ops[j])][0]
                fwd_bwd += profiled_time[str(mbs[j])][str(tp[j])][str(algo[j])][str(ops[j])][1]
                update += profiled_time[str(mbs[j])][str(tp[j])][str(algo[j])][str(ops[j])][2]

            update_time_per_stage.append(update)
            comp_time_per_stage.append(fwd_bwd)

        ops_per_stage = [f"{ops[0]}-{ops[-1]}"for ops in pipeline.ops_per_stage]
        print(f"Pipeline stages is {ops_per_stage}")
        print(f"comp_time_per_stage is {comp_time_per_stage}")

        update_time = max(update_time_per_stage)
        tot_computation_time = sum(comp_time_per_stage)

        per_stage_comm_times = estimate_p2p_pipeline_times(self.model_mem_info, pipeline, base_batch_size, self.float_size, self.network_coeffs)
        print(f"per_stage_comm_times is {per_stage_comm_times}")
        tot_communication_time = sum(per_stage_comm_times)

        straggler_per_stage = [x+y for x,y in zip(comp_time_per_stage, per_stage_comm_times)]
        straggler = max(straggler_per_stage)
        straggler_overhead = (num_micro_batches - 1) * straggler

        print(f"num_micro_batches {num_micro_batches}, straggler: {straggler}, straggler_overhead: {straggler_overhead}, tot_communication_time: {tot_communication_time}, tot_computation_time: {tot_computation_time}, update_time: {update_time}")

        # total pipeline time
        t_pp = straggler_overhead + tot_communication_time + tot_computation_time + update_time
        return t_pp

    def simulate_iteration_time(self, plan_dict: dict):

        plan = convert_homogeneous(plan_dict, self.training_config)
        base_batch_size = plan.mbs
        num_micro_batches = math.ceil(self.global_batch_size / base_batch_size)

        fwd_times = {}
        bwd_times = {}
        update_times = {}

        # get time for the pipeline
        t_pp = self.get_time_per_pipeline(
                plan.pipeline, base_batch_size, num_micro_batches)
        t_sync = self.estimate_sync_time(plan)

        iteration_time = t_pp + t_sync
        # TODO: Fix iteration cost
        iteration_cost = 0

        print(f"*********** T_pp is {t_pp}, T_sync is {t_sync}, Iteration time is {iteration_time}")

        reformed_plan_dict = plan.to_dict()

        return iteration_time, iteration_cost, reformed_plan_dict

    def estimate_sync_time(self, plan: Plan):

        # TODO: Fix: How to accumulate model size from ops with different tp_size
        pipeline = plan.pipeline

        # 1. time for the pipeline
        t_sync_pipeline = 0.0
        for i in range(pipeline.num_stages):
            ops = pipeline.ops_per_stage[i]
            dp = pipeline.num_dp[i]
            tp = pipeline.num_tp[i]
            algo = pipeline.algo[i]
            stage_size = sum([self.model_mem_info[str(tp[j])][str(algo[j])][str(ops[j])]["params_floats"] * self.float_size * tp[j] for j in range(len(ops))])
            gpu_node_tuple = (pipeline.gpu_type, pipeline.gpus_per_node)
            network_coef = self.network_coeffs[gpu_node_tuple][0]
            t_sync_stage = get_ar_time_with_buckets(stage_size, dp[0], network_coef)
            t_sync_pipeline = max(t_sync_pipeline, t_sync_stage)

        return t_sync_pipeline

def get_ar_time_with_buckets(model_size, D, netw_coef_d):

    dp_time = 0
    remaining = model_size
    # print(f"model_size is {model_size}, remaining is {remaining}")
    while remaining > 0:
        bucket_size = MEMORY_BUCKET_DEEPSPEED_SIZE if remaining > MEMORY_BUCKET_DEEPSPEED_SIZE else remaining
        dp_time += estimate_ar_time(bucket_size, D, netw_coef_d)
        remaining -= bucket_size

    return dp_time

def estimate_p2p_pipeline_times(model_mem_info: dict, pipeline: Pipeline, base_batch_size: int, float_size: int, network_coeffs_dict: dict):
    activation_sending_times = []
    for i in range(pipeline.num_stages - 1):
        op = pipeline.ops_per_stage[i][-1]
        dp = pipeline.num_dp[i][-1]
        mbs = base_batch_size / dp
        tp = pipeline.num_tp[i][-1]
        algo = pipeline.algo[i][-1]
        activation_size = model_mem_info[str(tp)][str(algo)][str(op)]["act_output_floats"] * float_size * mbs

        gpu_node_tuple = (pipeline.gpu_type, pipeline.gpus_per_node)
        network_coeffs = network_coeffs_dict[gpu_node_tuple][0]
        send_time = estimate_send_time(activation_size, network_coeffs)
        activation_sending_times.append(2*send_time)  # send activations, receive gradients
    activation_sending_times.append(0.0)
    return activation_sending_times