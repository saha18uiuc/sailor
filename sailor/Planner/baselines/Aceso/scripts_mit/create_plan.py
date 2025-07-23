import argparse
import json
import copy
import os

from sailor.Planner.baselines.Aceso.aceso_planner import AcesoPlanner
from sailor.Planner.simulations.runtime_simulator_op import SimulatorOP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--basic_cluster_config_json', type=str, default='../../simulations/tests/basic_cluster_config.json',
                        help='Json file containing info for the cluster setup.')
    parser.add_argument('--gpu_type', type=str, choices=['A100-40', 'V100-16', 'RTX-3090', 'RTX-2080', 'Titan-RTX', 'GH-96'], default='RTX-3090')
    parser.add_argument('--training_config_json', type=str, default='../../simulations/tests/training_config_opt_350.json', help='Json file containing training info')
    parser.add_argument('--simulator_profile_file', type=str, default='../../simulations/profiles_tmp_aceso.json',
                        help='JSON file containing fwd, bwd and update time for different models and microbatches')
    parser.add_argument('--simulator_llm_info', type=str, default='../../llm_info_aceso.json',
                        help='JSON file containing memory info for different models, tp sizes and algo types')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--plan_path', type=str, required=True, help='Plan to be created and simulated')
    parser.add_argument('--config_save_path', type=str, default='aceso_logs/validation/')

    args = parser.parse_args()
    fp16 = args.fp16
    with open(args.basic_cluster_config_json, 'r') as f:
        cluster_config = json.load(f)[args.gpu_type]
    with open(args.training_config_json, 'r') as f:
        training_config = json.load(f)
    with open(args.simulator_profile_file, 'r') as f:
        profiles = json.load(f)
    with open(args.simulator_llm_info, 'r') as f:
        llm_info = json.load(f)
    with open(args.plan_path, 'r') as f:
        sailor_plan = json.load(f)

    dp = sailor_plan['pipeline_list'][0]['dp']
    pp = sailor_plan['pipeline_list'][0]['num_stages']
    mbs = sailor_plan['mbs']
    gbs = sailor_plan['gbs']
    nnode = sailor_plan['N']
    tp = [stage[0][-1] for stage in sailor_plan['pipeline_list'][0]['tmp_per_stage']]
    layers_per_stage = sailor_plan['pipeline_list'][0]['layers_per_stage']

    original_plan = {}
    original_plan['model_name'] = training_config['model']
    original_plan['num_layers'] = training_config['num_layers']
    original_plan['seq_length'] = training_config['sequence_length']
    original_plan['max_position_embeddings'] = training_config['max_position_embeddings']
    original_plan['num_attention_heads'] = training_config['heads']
    original_plan['hidden_size'] = training_config['hidden_size']
    original_plan['global_batch_size'] = gbs
    original_plan['micro_batch_size'] = mbs * dp[0]
    original_plan['num_stages'] = pp
    original_plan['num_gpus'] = []
    original_plan['checkpoint_activations'] = [False] * pp
    original_plan['resharding_stages'] = [False] * pp
    original_plan['num_ops_in_each_stage'] = []
    original_plan['model_parallel_size_of_each_op'] = []
    original_plan['data_parallel_size_of_each_op'] = []
    original_plan['recompute_ops'] = []
    original_plan['algo_of_each_op'] = []
    for i in range(pp):
        num_ops = 0
        for j in layers_per_stage[i]:
            if j == 0:
                num_ops += 1
            elif j == training_config['num_all_layers'] - 1:
                num_ops += 2
            else:
                num_ops += 13
        original_plan['num_ops_in_each_stage'].append(num_ops)
        original_plan['model_parallel_size_of_each_op'].append([tp[i]] * num_ops)
        original_plan['data_parallel_size_of_each_op'].append([dp[i]] * num_ops)
        original_plan['recompute_ops'].append([0] * num_ops)
        original_plan['algo_of_each_op'].append([0] * num_ops)
        original_plan["num_gpus"].append(dp[i] * tp[i])
    print(original_plan)
    
    plan = copy.deepcopy(original_plan)
    plan['name'] = 'Aceso'
    plan['gpu_type'] = args.gpu_type
    plan['num_gpus_per_node'] = cluster_config["gpus_per_node"]
    plan['used_gpus'] = {args.gpu_type: sum(plan["num_gpus"])}
    training_config['global_batch_size'] = plan['global_batch_size']
    model_name = plan['model_name']
    
    simulator = SimulatorOP(training_config, llm_info, fp16, profiles)
    print(f"CHECK PLAN {plan}")
    if not simulator.check_config_fits(plan):
        print("OOM!!!")
    else:
        iteration_time, comm_cost, reformed_plan_dict = simulator.simulate_iteration_time(plan)
        throughput = 1/iteration_time
        print(f"Throughput is {throughput}")
    save_path = os.path.join(args.config_save_path, f'{model_name}_{nnode}-{gbs}-{dp[0]}-{pp}-{mbs}.json')
    json.dump(original_plan, open(save_path, 'w'), indent=4)
