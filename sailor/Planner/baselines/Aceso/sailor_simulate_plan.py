import argparse
import json

from sailor.Planner.baselines.Aceso.aceso_planner import AcesoPlanner
from sailor.Planner.simulations.runtime_simulator_op import SimulatorOP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--basic_cluster_config_json', type=str, required=True,
                        help='Json file containing info for the cluster setup.')
    parser.add_argument('--gpu_type', type=str, choices=['A100-40', 'V100-16', 'RTX-3090', 'RTX-2080', 'GH-96', 'Titan-RTX'], required=True)
    parser.add_argument('--training_config_json', type=str, required=True, help='Json file containing training info')
    parser.add_argument('--simulator_profile_file', type=str, required=True,
                        help='JSON file containing fwd, bwd and update time for different models and microbatches')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--plan_path', type=str, required=True, help='Plan to be simulated')

    args = parser.parse_args()
    fp16 = args.fp16
    with open(args.basic_cluster_config_json, 'r') as f:
        cluster_config = json.load(f)[args.gpu_type]
    with open(args.training_config_json, 'r') as f:
        training_config = json.load(f)
    with open(args.simulator_profile_file, 'r') as f:
        profiles = json.load(f)
    with open("../../llm_info_aceso.json", 'r') as f:
        llm_info = json.load(f)
    with open(args.plan_path, 'r') as f:
        plan = json.load(f)
        plan['name'] = 'Aceso'
        plan['gpu_type'] = args.gpu_type
        print(f"cluster_config['gpus_per_node']: {cluster_config['gpus_per_node']}")
        plan['num_gpus_per_node'] = cluster_config["gpus_per_node"]
        plan['used_gpus'] = {args.gpu_type: sum(plan["num_gpus"])}
        training_config['global_batch_size'] = plan['global_batch_size']

    simulator = SimulatorOP(training_config, llm_info, fp16, profiles)
    print(f"CHECK PLAN {plan}")
    if not simulator.check_config_fits(plan):
        print("OOM!!!")
    else:
        iteration_time, comm_cost, reformed_plan_dict = simulator.simulate_iteration_time(plan)
        max_memory = simulator.get_max_memory(plan)
        throughput = 1/iteration_time
        print(f"Throughput is {throughput}")
        print("max_time,max_memory")
        print(f"{iteration_time},{max_memory}")