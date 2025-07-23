import argparse
import json
import copy

from sailor.Planner.baselines.Aceso.aceso_planner import AcesoPlanner
from sailor.Planner.simulations.runtime_simulator_op import SimulatorOP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--basic_cluster_config_json', type=str, default='../../simulations/tests/basic_cluster_config.json',
                        help='Json file containing info for the cluster setup.')
    parser.add_argument('--gpu_type', type=str, choices=['A100-40', 'V100-16', 'RTX-3090', 'GH-96'], default='GH-96')
    parser.add_argument('--training_config_json', type=str, default='../../simulations/tests/training_config_opt_350.json', help='Json file containing training info')
    parser.add_argument('--simulator_profile_file', type=str, default='../../simulations/profiles_tmp_aceso.json',
                        help='JSON file containing fwd, bwd and update time for different models and microbatches')
    parser.add_argument('--simulator_llm_info', type=str, default='../../llm_info_aceso.json',
                        help='JSON file containing memory info for different models, tp sizes and algo types')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--plan_path', type=str, required=True, help='Plan to be modified and simulated')
    parser.add_argument('--gbs', type=int, required=True)
    parser.add_argument('--dp', type=int, required=True)
    parser.add_argument('--pp', type=int, required=True)
    parser.add_argument('--mbs', type=int, required=True)

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
        original_plan = json.load(f)
        original_plan['global_batch_size'] = args.gbs
        original_plan['micro_batch_size'] = args.mbs * args.dp
        for i in range(args.pp):
            for j in range(len(original_plan['model_parallel_size_of_each_op'][i])):
                original_plan['model_parallel_size_of_each_op'][i][j] = 4
            for j in range(len(original_plan['data_parallel_size_of_each_op'][i])):
                original_plan['data_parallel_size_of_each_op'][i][j] = args.dp
            original_plan["num_gpus"][i] = args.dp * 4
        
        plan = copy.deepcopy(original_plan)
        plan['name'] = 'Aceso'
        plan['gpu_type'] = args.gpu_type
        plan['num_gpus_per_node'] = cluster_config["gpus_per_node"]
        plan['used_gpus'] = {args.gpu_type: sum(plan["num_gpus"])}
        training_config['global_batch_size'] = plan['global_batch_size']
        model_name = plan['model_name']
        nnode = args.dp * args.pp

    simulator = SimulatorOP(training_config, llm_info, fp16, profiles)
    print(f"CHECK PLAN {plan}")
    if not simulator.check_config_fits(plan):
        print("OOM!!!")
    else:
        iteration_time, comm_cost, reformed_plan_dict = simulator.simulate_iteration_time(plan)
        throughput = 1/iteration_time
        print(f"Throughput is {throughput}")
    save_path = f'aceso_validation_clariden/configs/{model_name}_{nnode}-{args.gbs}-{args.dp}-{args.pp}-{args.mbs}.json'
    json.dump(original_plan, open(save_path, 'w'), indent=4)