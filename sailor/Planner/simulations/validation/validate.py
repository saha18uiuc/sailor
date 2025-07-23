import argparse
import json
import os
import pandas as pd

from sailor.Planner.baselines.AMP.amp_simulator import AMPSimulator
from sailor.Planner.baselines.Varuna.varuna_simulator import VarunaSimulator
from sailor.Planner.baselines.Galvatron.galvatron_simulator import GalvatronSimulator
from sailor.Planner.baselines.Piper.piper_simulator import PiperSimulator
from sailor.Planner.baselines.Metis.metis_simulator import MetisSimulator
from sailor.Planner.baselines.DTFM.dtfm_simulator import DTFMSimulator
from sailor.Planner.baselines.FlashFlex.flashflex_simulator import FlashFlexSimulator

from sailor.Planner.simulations.runtime_simulator import Simulator


def simulate_plan(simulator, plan, zone):
    iteration_time, _, _ = simulator.simulate_iteration_time(plan)
    throughput = 1/iteration_time
    return throughput


def simulate_sailor(
    sailor_path,
    training_config,
    llm_info,
    fp16,
    zone,
    pipeline_list,
    mbs,
    profile_path,
    simulate_time=False,
    simulate_mem=False
):

    with open(profile_path, 'r') as f:
        profiles = json.load(f)

    simulator = Simulator(sailor_path, training_config, llm_info, fp16, profiles, zone)

    new_pipeline_list = [
        {
            'num_stages': pipeline_list[0]['num_stages'],
            'layers_per_stage': pipeline_list[0]['layers_per_stage'],
            'tmp_per_stage': pipeline_list[0]['tmp_per_stage'],
            'dp': pipeline_list[0]['dp']
        }
    ]
    plan = {
        'pipeline_list': new_pipeline_list,
        'mbs': mbs
    }

    if simulate_time:
        throughput = simulate_plan(
            simulator,
            plan,
            zone
        )
        sim_time = 1/throughput
        print(f"SAILOR-Simulated iteration time is {sim_time}")
        return sim_time

    if simulate_mem:
        max_mem = 0
        num_stages = pipeline_list[0]['num_stages']
        tmp_per_stage = pipeline_list[0]['tmp_per_stage']

        for stage in range(num_stages):
            for tmp_config in tmp_per_stage[stage]:
                tmp = tmp_config[1]
                mem_stage_gpu = simulator.get_memory_footprint_on_gpu(plan, stage, tmp)
                print(f"Stage {stage}, TMP: {tmp}, Mem_stage_gpu: {mem_stage_gpu}")
                max_mem = max(max_mem, mem_stage_gpu)
        return max_mem


def simulate_hetero(
    simulator_name,
    training_config,
    pipeline_list,
    basic_cluster_config,
    mbs,
    profile_path,
    simulate_time=False,
    simulate_mem=False
):

    mp = pipeline_list[0]['tmp_per_stage'][0][0][1]
    dp = pipeline_list[0]['dp'][0]
    pp = pipeline_list[0]['num_stages']
    layer_partition = pipeline_list[0]['layers_per_stage']
    gpu_types = set()
    num_nodes_type = {}
    tp_configs = pipeline_list[0]['tmp_per_stage']
    for tp_config_stage in tp_configs:
        for tp_config in tp_config_stage:
            vms = tp_config[0]
            for vm in vms:
                gpu_type = vm[0]
                gpu_types.add(gpu_type)
                if gpu_type not in num_nodes_type:
                    num_nodes_type[gpu_type] = 0
                num_nodes_type[gpu_type] += 1
    print(gpu_types)

    metis_cluster_config = {}
    for gpu_type in gpu_types:
        metis_cluster_config[gpu_type] = {
            "num_nodes": num_nodes_type[gpu_type],
            "gpus_per_node": basic_cluster_config[gpu_type]['gpus_per_node'],
            "mem_per_gpu": basic_cluster_config[gpu_type]['mem_per_gpu'],
        }
    metis_cluster_config["zone"] = basic_cluster_config["zone"]

    if simulator_name == 'AMP':
        simulator = AMPSimulator(profile_path, metis_cluster_config, training_config)
    elif simulator_name == 'Metis':
        simulator = MetisSimulator(profile_path, metis_cluster_config, training_config)
    elif simulator_name == 'FlashFlex':
        simulator = FlashFlexSimulator(profile_path, metis_cluster_config, training_config)
    else:
        raise NotImplementedError

    if simulate_time:
        sim_time = simulator.get_time(mp, dp, pp, mbs, tp_configs, layer_partition)
        print(f"{simulator_name}-Simulated iteration time is {sim_time}")
        return sim_time

    if simulate_mem:
        max_mem = simulator.get_memory(mp, dp, pp, mbs, tp_configs, layer_partition)
        print(f"{simulator_name}-Simulated max memory is {max_mem} GB")
        return max_mem


def simulate_homo(
    simulator_name,
    training_config,
    pipeline_list,
    basic_cluster_config,
    mbs,
    profile_path,
    gpu_type,
    simulate_time=False,
    simulate_mem=False
):
    mp = pipeline_list[0]['tmp_per_stage'][0][0][1]
    dp = pipeline_list[0]['dp'][0]
    pp = pipeline_list[0]['num_stages']
    layer_partition = pipeline_list[0]['layers_per_stage']

    cluster_config = {}
    cluster_config['gpu_type'] = gpu_type
    cluster_config['num_nodes'] = dp*pp  # TODO: generalize
    cluster_config["gpus_per_node"] = basic_cluster_config[gpu_type]['gpus_per_node']
    cluster_config["mem_per_gpu"] = basic_cluster_config[gpu_type]['mem_per_gpu']
    cluster_config["zone"] = basic_cluster_config["zone"]

    if simulator_name == 'Varuna':
        simulator = VarunaSimulator(profile_path, cluster_config, training_config)
    elif simulator_name == 'Galvatron':
        simulator = GalvatronSimulator(profile_path, cluster_config, training_config)
    elif simulator_name == 'Piper':
        simulator = PiperSimulator(profile_path, cluster_config, training_config)
    elif simulator_name == 'DTFM':
        simulator = DTFMSimulator()
    else:
        raise NotImplementedError

    if simulate_time:
        sim_time = simulator.get_time(mp, dp, pp, mbs, layer_partition)
        print(f"{simulator_name}-Simulated iteration time is {sim_time}")
        return sim_time

    if simulate_mem:
        max_mem = simulator.get_memory(mp, dp, pp, mbs, layer_partition)
        print(f"{simulator_name}-Simulated max memory is {max_mem} GB")
        return max_mem


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--training_config_json', type=str, required=True, help='Json file containing training info')
    parser.add_argument('--basic_cluster_config_json', type=str, required=True,
                        help='Json file containing info for the cluster setup.')
    parser.add_argument('--simulator_profile_path', type=str, required=True,
                        help='JSON file containing fwd, bwd and update time for different models and microbatches')
    parser.add_argument('--plan_config_file', type=str, required=True,
                        help='JSON file containing fwd, bwd and update time for different models and microbatches')
    parser.add_argument('--simulate_time', action='store_true')
    parser.add_argument('--simulate_mem', action='store_true')
    parser.add_argument('--simulator', type=str, required=True, help='Simulator name')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--gpu_type', type=str, default=None, help='GPU type, used for homogeneous simulators')
    parser.add_argument('--sailor_path', type=str, default=None, help='Path to SAILOR repo')

    args = parser.parse_args()
    os.environ['SAILOR_PATH'] = args.sailor_path

    with open(args.training_config_json, 'r') as f:
        training_config = json.load(f)

    with open(f"{args.sailor_path}/elastic-spot-ml/sailor/Planner/llm_info.json", 'r') as f:
        llm_info = json.load(f)

    with open(args.basic_cluster_config_json, 'r') as f:
        basic_cluster_config = json.load(f)
    with open(args.plan_config_file, 'r') as f:
        plan_config = json.load(f)

    zone = plan_config['zone']
    pipeline_list = plan_config['pipeline_list']
    mbs = plan_config['mbs']
    training_config["global_batch_size"] = plan_config['gbs']

    if args.simulator == 'SAILOR':
        sim_res = simulate_sailor(
            args.sailor_path,
            training_config,
            llm_info,
            False,  # TODO
            zone,
            pipeline_list,
            mbs,
            args.simulator_profile_path,
            args.simulate_time,
            args.simulate_mem
        )
    elif (args.simulator == 'Metis') or (args.simulator == 'AMP') or (args.simulator == 'FlashFlex'):
        sim_res = simulate_hetero(
            args.simulator,
            training_config,
            pipeline_list,
            basic_cluster_config,
            mbs,
            args.simulator_profile_path,
            args.simulate_time,
            args.simulate_mem
        )
    else:
        sim_res = simulate_homo(
            args.simulator,
            training_config,
            pipeline_list,
            basic_cluster_config,
            mbs,
            args.simulator_profile_path,
            args.gpu_type,
            args.simulate_time,
            args.simulate_mem
        )

    if args.simulate_time:
        output_file = f"{args.output_path}/{args.simulator}-time.csv"
    elif args.simulate_mem:
        output_file = f"{args.output_path}/{args.simulator}-mem.csv"

    data = pd.DataFrame(data={
        "N": [plan_config['N']],
        "gbs": [plan_config['gbs']],
        "dp": [pipeline_list[0]['dp'][0]],
        "pp": [pipeline_list[0]['num_stages']],
        "tp": [pipeline_list[0]['tmp_per_stage'][0][0][1]],
        "mbs": [mbs],
        "real": [plan_config['real'] if args.simulate_time else plan_config['max_mem']],
        "estimated": [sim_res]
    })
    if not os.path.exists(output_file):
        data.to_csv(output_file, index=False)
    else:
        data.to_csv(output_file, index=False, header=False, mode='a')
