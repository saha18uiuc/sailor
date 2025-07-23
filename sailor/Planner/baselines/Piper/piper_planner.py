import json
import os
from os.path import expanduser
import numpy as np

from sailor.Planner.baselines.baseline_planner import BaselinePlanner


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(JSONEncoder, self).default(obj)


class PiperPlanner(BaselinePlanner):
    def __init__(self, profiling_file) -> None:
        super().__init__()
        self.profiling_file = profiling_file
        with open(self.profiling_file, 'r') as f:
            self.profile = json.load(f)

        home_dir = os.environ.get('SAILOR_PATH')

        self.profile = self.profile["1"] # Piper assumes Mbs is given, so we always assume mbs=1
        self.piper_algo_path = f"{home_dir}/elastic-spot-ml/sailor/Planner/baselines/Piper/src"
        compile_cmd = "rm -rf algo.bin && g++ -O3 algo.cpp -ljsoncpp  -o algo.bin"
        os.system(f"cd {self.piper_algo_path} && {compile_cmd}")

        network_coeff_path = f"{home_dir}/elastic-spot-ml/sailor/providers/gcp/multizone_bandwidths_het.json"
        with open(network_coeff_path, 'r') as f:
            self.network_profile = json.load(f)


    def get_sorted_plans(self, cluster_config: dict, training_config: dict):
        # adjust input file
        num_nodes = cluster_config['num_nodes']
        mbsInBatch = training_config['global_batch_size']  # no mbs is found, we use mbs=1
        self.profile["maxDevices"] = num_nodes * cluster_config['gpus_per_node']
        gpu_type = cluster_config["gpu_type"]
        gpus_per_node = cluster_config["gpus_per_node"]

        print(f"Max Devices is {self.profile['maxDevices']}")

        self.profile["mbsInBatch"] = mbsInBatch
        gpu_count = str(cluster_config['gpus_per_node'])
        zone = cluster_config['zone']
        self.profile["bandwidth"] = self.network_profile[zone][gpu_type][gpu_count][zone][gpu_type][gpu_count][1]

        model_path = f"{self.piper_algo_path}/model.json"

        with open(model_path, "w") as f:
            json.dump(self.profile, f, indent=2, cls=JSONEncoder)
            f.flush()

        output_path = "piper_test.txt"

        run_piper_cmd = f"{self.piper_algo_path}/algo.bin {model_path} 0 {output_path}"
        os.system(run_piper_cmd)
        stages = []
        tp_degrees = []
        dp = 0
        tp = 0
        mbs = 0
        used_gpus = 0

        pipeline_list = []
        result = []
        with open(output_path, "r") as f:
            lines = f.readlines()
            pipeline_def = {}
            dp_degrees = []
            tp_degrees = []
            for line in lines:
                print(line)
                # each line is a stage
                layers, dp, tp = line.split(",")
                dp = int(dp)
                tp = int(tp)
                layers = [int(layer) for layer in layers.split(" ")[:-1]]
                stages.append(layers)
                tmp_config = [[(gpu_type, gpus_per_node, cluster_config['zone'])], tp]
                tmp_configs = [tmp_config for _ in range(dp)]
                tp_degrees.append(tmp_configs)
                dp_degrees.append(dp)
                used_gpus += dp*tp
            pipeline_def = {
                'num_stages': len(stages),
                'tmp_per_stage': tp_degrees,
                'layers_per_stage': stages,
                'dp': dp_degrees,
                'sender_zone': cluster_config['zone'],
                'receiver_zone': cluster_config['zone']
            }
            if used_gpus > 0:
                result.append(
                    {
                        'mbs': 1,
                        'pipeline_list': [pipeline_def],
                        'gpu_type': gpu_type,
                        'num_gpus_per_node': gpus_per_node,
                        'used_gpus': {gpu_type: used_gpus}
                    }
                )
                pipeline_list.append(pipeline_def)

        # for PIPER, if we output all solutions, we will change the algorithm runtime, so we output one solution
        return result
