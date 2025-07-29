import os
import argparse

def run_all_for_model(args):
    home_dir = args.sailor_path
    sailor_sim_path = "sailor/sailor/Planner/simulations"

    train_config_files = {
        "OPT-350": f"{home_dir}/{sailor_sim_path}/configs/training_config_opt_350.json",
        "GPT-Neo-2.7": f"{home_dir}/{sailor_sim_path}/configs/training_config_gpt_neo27.json",
    }

    res_dir = args.res_dir

    base_cmd = f"python {home_dir}/{sailor_sim_path}/simulator.py --sailor_path {args.sailor_path} --trace_file {args.trace_file} --basic_cluster_config_json {args.basic_cluster_config_json} "\
        f"--training_config_json {train_config_files[args.model_name]}  --simulator_profile_file {args.simulator_profile_file} --result_dir_path {res_dir} "

    if args.max_cost:
        base_cmd += f"--max_cost {args.max_cost} "
    if args.max_cost_file:
        base_cmd += f"--max_cost_file {args.max_cost_file} "
    if args.min_throughput:
        base_cmd += f"--min_throughput {args.min_throughput} "
    if args.min_throughput_file:
        base_cmd += f"--min_throughput_file {args.min_throughput_file} "

    base_cmd += f"--objective {args.objective} "

    varuna_cmd = base_cmd + f"--planner Varuna --planner_profile_file {home_dir}/sailor/sailor/Planner/baselines/Varuna/profiles/{args.model_name}/{args.gpu_type}/profile_{args.gpus_per_node}.json"
    piper_cmd = base_cmd + f"--planner Piper --planner_profile_file {home_dir}/sailor/sailor/Planner/baselines/Piper/profiles/{args.model_name}/{args.gpu_type}/profile.json"
    amp_cmd = base_cmd + f"--planner AMP --planner_profile_file {home_dir}/sailor/sailor/Planner/baselines/AMP/profiles/{args.model_name}"
    metis_cmd = base_cmd + f"--planner Metis --planner_profile_file {home_dir}/sailor/sailor/Planner/baselines/Metis/profiles/{args.model_name}"
    galvatron_cmd = base_cmd + f"--planner Galvatron --planner_profile_file {home_dir}/sailor/sailor/Planner/baselines/Galvatron/profiles/{args.model_name}/{args.gpu_type}"
    dtfm_cmd = base_cmd + f"--planner DTFM  --planner_profile_file {args.simulator_profile_file} --quotas_dict {args.quotas_dict}"
    sailor_cmd = base_cmd + f"--planner SAILOR  --sailor_profile_file_dir {home_dir}/sailor/sailor/Planner/sailor_planner/profiles/{args.model_name}/ --quotas_dict {args.quotas_dict}"
    flashflex_cmd = base_cmd +  f"--planner FlashFlex  --planner_profile_file {home_dir}/sailor/sailor/Planner/baselines/FlashFlex/src/machine_amounts.json --objective throughput"

    if args.baselines=="all":
        all_cmd = [varuna_cmd, piper_cmd, amp_cmd, galvatron_cmd, dtfm_cmd, flashflex_cmd, metis_cmd, sailor_cmd]
    elif args.baselines=="heterogeneous":
        all_cmd = [amp_cmd, flashflex_cmd, metis_cmd, sailor_cmd]
    elif args.baselines=="geo":
        all_cmd = [dtfm_cmd, sailor_cmd]
    elif args.baselines=="cost":
        all_cmd = [galvatron_cmd, amp_cmd, flashflex_cmd, metis_cmd, dtfm_cmd, sailor_cmd]
    for cmd in all_cmd:
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator eval',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-name", type=str, required=True, help="Model to run simulations for")
    parser.add_argument("--gpu-type", type=str, required=True, help="GPU type")
    parser.add_argument('--trace_file', type=str, required=True, help='GPU availability trace file')
    parser.add_argument('--basic_cluster_config_json', type=str, required=True,
                        help='Json file containing info for the cluster setup.')
    parser.add_argument('--simulator_profile_file', type=str, required=True,
                        help='JSON file containing fwd, bwd and update time for different models and microbatches')
    parser.add_argument('--quotas_dict', type=str, default='',
                        help='Json file containing user quotas')
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Number of GPUs per node. Used by Oobleck and Varuna")
    parser.add_argument('--max_cost', type=float, default=0.0,
                        help='Max cost (USD/iteration)')
    parser.add_argument('--max_cost_file', type=str, default=None,
                        help='A json file corresponding to different cost limits for the given trace')
    parser.add_argument('--min_throughput', type=float, default=0.0,
                        help='Min througput (Iters/sec)')
    parser.add_argument('--min_throughput_file', type=str, default=None,
                        help='A json file corresponding to different throughput for the given trace')
    parser.add_argument('--sailor_path', type=str, required=True,
                        help='Path to the sailor repo')
    parser.add_argument('--res_dir', type=str, required=True, help='Path to results dir')
    parser.add_argument('--objective', type=str, required=True,
                        help="User objective ('throughput' or 'iteration_cost')")
    parser.add_argument('--baselines', type=str, required=True, help='can be either of (all, heterogeneous, geo)')
    args = parser.parse_args()
    run_all_for_model(args)