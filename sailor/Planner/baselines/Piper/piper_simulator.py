import os
import json

from sailor.Planner.baselines.Piper.piper_planner import JSONEncoder

class PiperSimulator():
    def __init__(self, profile_file: str, cluster_config: dict, training_config: dict) -> None:
        with open(profile_file, 'r') as f:
            self.profile = json.load(f)

        home_dir = os.environ.get('SAILOR_PATH')
        self.piper_algo_path = f"{home_dir}/elastic-spot-ml/sailor/Planner/baselines/Piper/src"
        # if not os.path.exists(f"{self.piper_algo_path}/algo"):
        compile_cmd = "g++ -O3 algo.cpp -ljsoncpp -o algo.bin"
        os.system(f"cd {self.piper_algo_path} && {compile_cmd}")

        network_coeff_path = f"{home_dir}/elastic-spot-ml/sailor/providers/gcp/multizone_bandwidths.json"
        with open(network_coeff_path, 'r') as f:
            self.network_profile = json.load(f)

        self.global_batch_size = training_config["global_batch_size"]
        self.num_nodes = cluster_config['num_nodes']
        self.gpu_count = cluster_config['gpus_per_node']
        self.zone = cluster_config['zone']
        self.gpu_type = cluster_config["gpu_type"]

    def get_memory(self, mp, dp, pp, mbs, layer_partition=None):
        self.profile = self.profile[str(mbs)]

        self.profile["maxDevices"] = self.num_nodes * self.gpu_count
        self.mbsInBatch = self.global_batch_size // mbs
        self.profile["mbsInBatch"] = self.mbsInBatch
        self.profile["bandwidth"] = self.network_profile[self.zone][self.gpu_type][str(self.gpu_count)][self.zone][self.gpu_type][str(self.gpu_count)][1]

        model_path = f"{self.piper_algo_path}/model.json"
        self.profile["mp"] = mp
        self.profile["dp"] = dp
        self.profile["pp"] = pp
        self.profile["mbs"] = mbs
        self.profile["layers"] = layer_partition

        with open(model_path, "w") as f:
            json.dump(self.profile, f, indent=2, cls=JSONEncoder)
            f.flush()

        output_path = "piper_test.txt"

        run_piper_cmd = f"{self.piper_algo_path}/algo.bin {model_path} 2 {output_path}"
        os.system(run_piper_cmd)

        max_mem = 0
        with open(output_path, "r") as f:
            line = f.readline()
            max_mem = float(line)
        return max_mem


    def get_time(self, mp, dp, pp, mbs, layer_partition=None):

        self.profile = self.profile[str(mbs)]
        self.profile["maxDevices"] = self.num_nodes * self.gpu_count
        self.mbsInBatch = self.global_batch_size // mbs
        self.profile["mbsInBatch"] = self.mbsInBatch
        self.profile["bandwidth"] = self.network_profile[self.zone][self.gpu_type][str(self.gpu_count)][self.zone][self.gpu_type][str(self.gpu_count)][1]

        model_path = f"{self.piper_algo_path}/model.json"
        self.profile["mp"] = mp
        self.profile["dp"] = dp
        self.profile["pp"] = pp
        self.profile["mbs"] = mbs
        self.profile["layers"] = layer_partition

        with open(model_path, "w") as f:
            json.dump(self.profile, f, indent=2, cls=JSONEncoder)
            f.flush()

        output_path = "piper_test.txt"

        run_piper_cmd = f"{self.piper_algo_path}/algo.bin {model_path} 1 {output_path}"
        os.system(run_piper_cmd)

        batch_time = 0
        with open(output_path, "r") as f:
            line = f.readline()
            batch_time = float(line) * self.mbsInBatch
        return batch_time