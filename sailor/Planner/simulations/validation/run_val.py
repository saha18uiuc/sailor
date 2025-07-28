import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--training_config_json', type=str, required=True, help='Json file containing training info')
    parser.add_argument('--basic_cluster_config_json', type=str, required=True,
                        help='Json file containing info for the cluster setup.')
    parser.add_argument('--simulator_profile_path', type=str, required=True,
                        help='JSON file containing fwd, bwd and update time for different models and microbatches')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path containing input profiles')
    parser.add_argument('--simulate_time', action='store_true')
    parser.add_argument('--simulate_mem', action='store_true')
    parser.add_argument('--simulator', type=str, required=True, help='Simulator name')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--gpu_type', type=str, default=None, help='GPU type, used for homogeneous simulators')
    parser.add_argument('--sailor_path', type=str, default=None, help='Path to SAILOR repo')

    args = parser.parse_args()
    all_files = [file for file in os.listdir(args.input_path)]
    all_files.sort()

    for i,file in enumerate(all_files):
        file_path = os.path.join(args.input_path, file)
        cmd = f"python {args.sailor_path}/sailor/sailor/Planner/simulations/validation/validate.py --training_config_json {args.training_config_json} --basic_cluster_config_json {args.basic_cluster_config_json} --simulator_profile_path {args.simulator_profile_path} --plan_config_file {file_path} --simulator {args.simulator} --output_path {args.output_path} --sailor_path {args.sailor_path} "
        if args.simulate_time:
            cmd += "--simulate_time "
        if args.simulate_mem:
            cmd += "--simulate_mem "
        if args.gpu_type:
            cmd += f"--gpu_type {args.gpu_type}"
        os.system(cmd)
