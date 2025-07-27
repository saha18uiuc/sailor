import argparse
import time
from concurrent import futures
import grpc
import json
import os
import subprocess

from sailor.Controller.controller import NodeSet
from sailor.protos import orchestration_pb2, orchestration_pb2_grpc
from sailor.protos.orchestration_pb2_grpc import (
    WorkerAgentStub, MasterControllerStub
)
from sailor.Worker.elastic_worker_agent import WORKER_AGENT_PORT, TRAINING_START_PORT
from sailor.Planner.sailor_planner.cpp_src.planner import SailorPlanner

class LocalController(NodeSet):
    def __init__(self, args, node_list) -> None:
        super().__init__()
        self.args = args
        self.node_list = node_list
        self.num_nodes = len(node_list)

    def update_node(self, node):
        pass

    def update_nodes(self, cluster_config, training_config):
        num_stages=cluster_config['pipeline_parallelism']
        all_tp_degrees=cluster_config['tp_degrees']
        dp = cluster_config['data_parallelism']
        node_rank = 0
        ga_steps = training_config['global_batch_size'] // (dp * cluster_config['microbatch_size'])
        node_list_idx = 0
        num_layers_per_stage = cluster_config['num_layers_per_stage']

        # find globals
        world_size = 0
        max_tensor_parallelism = 1
        configs_all_stages = []

        for stage_id in range(num_stages):
            stage_config = []
            for replica_id in range(dp):
                max_tensor_parallelism = max(max_tensor_parallelism, len(all_tp_degrees[stage_id][replica_id]))
                stage_config.append(orchestration_pb2.ReplicaConfig(replica_ranks=all_tp_degrees[stage_id][replica_id]))
                world_size += len(all_tp_degrees[stage_id][replica_id])

            sconf = orchestration_pb2.StageConfig(stage_replicas=stage_config)
            configs_all_stages.append(sconf)

        # send to nodes
        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            for stage_id in range(num_stages):
                for replica_id in range(dp):
                    tmp_degrees = all_tp_degrees[stage_id][replica_id]
                    node = self.node_list[node_list_idx]
                    # SIMPLIFIED VERSION - no fault tolerance
                    tp_node = len(tmp_degrees)
                    worker_configuration = orchestration_pb2.WorkerConfiguration(
                        ranks=tmp_degrees,
                        world_size=world_size,
                        master_ip=self.node_list[0],
                        master_port=TRAINING_START_PORT,
                        pipeline_parallelism=num_stages,
                        tensor_parallelism=tp_node,
                        data_parallelism=dp,
                        max_tensor_parallelism=max_tensor_parallelism,
                        all_stages=configs_all_stages,
                        layers_per_stage=num_layers_per_stage
                    )

                hparams = orchestration_pb2.HyperParams()
                hparams.global_batch_size = training_config['global_batch_size']
                hparams.micro_batch_size = cluster_config['microbatch_size'] # TODO: take this out of the hyperparams
                hparams.num_stages = num_stages # not used in megatron
                hparams.ga_steps = ga_steps

                print(worker_configuration, hparams)

                executor.submit(self.notify_topology_change,
                                node, worker_configuration, hparams, node, WORKER_AGENT_PORT)
                print(f"Sent training info to node: {node}")
                node_rank += tp_node


    def check_ready(self, node):
        request = orchestration_pb2.CheckReadyRequest(is_ready=1)
        grpc_target = f'{node}:{WORKER_AGENT_PORT}'
        with grpc.insecure_channel(grpc_target) as channel:
            stub = WorkerAgentStub(channel)
            stub.CheckReady(request)


    def check_if_ready(self, node):
        try:
            self.check_ready(node)
        except Exception:
            print(f"Worker {node} is not ready")
            return False

        return True


    def wait_all_ready(self):
        for node in self.node_list:
            print(f"Check if node {node} is ready")
            while (not self.check_if_ready(node)):
                pass


def main(args):
    node_list = []
    with open(args.node_list_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
           node_list.append(line[:-1])

    print(node_list)
    controller = LocalController(args, node_list)

    with open(args.training_config_json, 'r') as f:
        training_config = json.load(f)

    # 1. wait all to start
    controller.wait_all_ready()

    # 2. send config
    if args.call_planner:
        os.environ['SAILOR_PATH'] = args.sailor_path
        input_planner_config = {
            "A100-40_us-central1-a": 4,
            "max_cost": args.max_cost,
            "min_throughput": args.min_throughput
        }
        # TODO
        planner = SailorPlanner(
            args.sailor_profile_file_dir,
            args.training_config_json,
            args.quotas_dict,
            args.objective,
            args.fp16
        )
        arch = subprocess.check_output(['uname', '-m']).decode('utf-8')[:-1]

        sorted_plans = planner.get_sorted_plans(
            cluster_config=input_planner_config,
            training_config=training_config
        )

        if len(sorted_plans) == 0:
            raise Exception("Not valid plan found! Sorry")
            exit(1)

        plan = sorted_plans[0]['pipeline_list'][0]
        print(f"plan is {plan}")

        layers_per_stage = list(([list(x) for x in plan['layers_per_stage']]))
        print(layers_per_stage)

        num_layers_per_stage = [len(x) for x in layers_per_stage]
        print(num_layers_per_stage)

        tmp_per_stage = []
        rank = 0
        for stage_config in plan['tmp_per_stage']:
            stage_tmps = []
            for replica in stage_config:
                tp_replica = replica[1]
                tps_this_replica = list(range(rank, rank+tp_replica))
                stage_tmps.append(tps_this_replica)
                rank += tp_replica
            tmp_per_stage.append(stage_tmps)

        cluster_config = {}
        cluster_config['pipeline_parallelism'] = plan['num_stages']
        cluster_config['tp_degrees'] = tmp_per_stage
        cluster_config['data_parallelism'] = plan['dp'][0]
        cluster_config['microbatch_size'] = sorted_plans[0]['mbs']
        cluster_config['num_layers_per_stage'] = num_layers_per_stage

        print(cluster_config)
        # TODO
    else:
        # no planner, for testing
        cluster_config = {
            'pipeline_parallelism': 1,
            'tp_degrees': [[[0,1]]],
            'data_parallelism': 1,
            'microbatch_size': 1,
            'num_layers_per_stage': [26]
        }

    controller.update_nodes(cluster_config, training_config)

    # 3. sleep for now
    while True:
        time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trace controller')
    parser.add_argument('--node_list_file', type=str, required=True, help="file containing IP of the nodes for communication")
    parser.add_argument('--waiting_time', type=int, default=0, help="waiting time before starting the training")
    parser.add_argument('--training_config_json', type=str, required=True, help='Json file containing training info')

    parser.add_argument('--call_planner', default=False, action='store_true')

    # if planner is used
    parser.add_argument('--sailor_profile_file_dir', type=str, default='',
                        help='Directory for SAILOR profiling')
    parser.add_argument('--quotas_dict', type=str, default='',
                        help='Json file containing user quotas')
    parser.add_argument('--objective', type=str, default='throughput',
                        help='User objective (throughput, cost, or value (iters/USD))')
    parser.add_argument('--max_cost', type=float, default=0.0,
                        help='Max cost (USD/iteration)')
    parser.add_argument('--min_throughput', type=float, default=0.0,
                        help='Min througput (iters/sec)')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--sailor_path', type=str, required=True, help='Path to the sailor repo')

    args = parser.parse_args()
    main(args)
