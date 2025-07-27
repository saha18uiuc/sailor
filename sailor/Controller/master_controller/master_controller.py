import argparse
import grpc
import time
import json
from concurrent import futures
from deepspeed.utils import logger
from google.api_core.exceptions import AlreadyExists
from gcp_client import GCPClient
from sailor.Planner.sailor_planner.planner import SailorPlanner
from sailor.Planner.baselines.AMP.amp_planner import AMPPlanner
from sailor.Planner.baselines.Oobleck.oobleck_planner import OobleckPlanner
from sailor.Planner.baselines.Varuna.varuna_planner import VarunaPlanner
from sailor.protos import orchestration_pb2, orchestration_pb2_grpc
from training_configuration import TrainingConfiguration


class MasterControllerServicer(orchestration_pb2_grpc.MasterControllerServicer):

    def __init__(self, args):
        self.args = args
        self.client = GCPClient()
        self.create_planner(args)
        self.current_cluster_config = []  # the most up-to-date cluster config

        print("Get optimal configuration from planner ...")
        time.sleep(10)

        if self.args.do_cleanup:
            self.client.clean_up_clusters()

        try:
            with open(args.training_config_json, 'r') as f:
                self.training_config = json.load(f)

            if args.planner in {'Varuna', 'Oobleck', 'AMP'}:
                with open(args.cluster_config_json, 'r') as f:
                    cluster_config = json.load(f)
                plan = self.planner.get_plan(cluster_config, self.training_config)
                self.current_cluster_config = [cluster_config]
            else:
                cluster_config, plan = self.planner.get_plan(self.training_config)
                self.current_cluster_config = cluster_config

            # create cluster
            self.client.create_clusters_from_config(self.current_cluster_config)

            self.training_config["num_stages"] = plan['P']
            self.training_config["micro_batch_size"] = plan['mbs']
            print(f"Training config is {self.training_config}")
            self.topology = TrainingConfiguration(self.training_config)

        except AlreadyExists:
            print("Cluster already exists or is being created...")

        # check whether to adapt configuration or not
        # self.planner_thread = threading.Thread(target=self.evaluate_planner, daemon=True)
        # self.planner_thread.start()

        # schedule a periodic cleanup job in a speparate thread
        # Commented for now since configuration does not change
        # self.cleanup_thread = threading.Thread(target=self.cleanup_training_configuration, daemon=True)
        # self.cleanup_thread.start()

    def create_planner(self, args):

        if args.planner in {'Varuna', 'Oobleck', 'AMP'}:
            assert args.cluster_config_json, "To use one of the baseline planners, provide a cluster config file"

        if args.planner == 'Varuna':
            assert args.varuna_profile_dir, "Provide profiling directory to use Varuna"
            self.planner = VarunaPlanner(args.varuna_profile_dir)
        elif args.planner == 'Oobleck':
            assert args.oobleck_profile_file, "Provide profiling file to use Oobleck"
            self.planner = OobleckPlanner(args.oobleck_profile_file)
        elif args.planner == 'AMP':
            assert args.amp_profile_dir, "Provide profiling directory to use AMP"
            self.planner = AMPPlanner(args.amp_profile_dir)
        elif args.planner == 'Sailor':
            self.planner = SailorPlanner(
                args.sailor_profile_file_dir,
                args.sailor_profile_file_dir_bw,
                args.quotas_dict,
                args.objective,
                args.max_cost,
                args.min_throughput,
            )
        else:
            raise ValueError("Invalid Planner Name")

    def Heartbeat(self, request, context):
        logger.info(f"Received heartbeat from cluster {request.cluster_id}")
        self.topology.add_cluster(request)
        return orchestration_pb2.HeartbeatResponse()

    def cleanup_training_configuration(self, sleep_time=60, heartbeat_threshold=60):
        while True:
            time.sleep(sleep_time)
            self.topology.remove_outdated_clusters()

    def evaluate_planner(self, sleep_time=60):
        while True:
            time.sleep(sleep_time)
            cluster_config, _ = self.planner.get_plan()
            if cluster_config == self.current_cluster_config:
                print("Clusters remain the same, do nothing!")
            else:
                print("Change topology!")
                raise NotImplementedError  # TODO: UPDATE


def serve(args):
    master_controller = MasterControllerServicer(args)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    orchestration_pb2_grpc.add_MasterControllerServicer_to_server(master_controller, server)
    logger.info('Starting server. Listening on port 50051.')
    server.add_insecure_port('[::]:50051')
    server.start()

    # Keep the server running
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Master Controller',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--planner', type=str, required=True,
                        help='Type of resource planner to use. Can choose between [Varuna, Oobleck,  AMP, Sailor]')
    parser.add_argument('--training_config_json', type=str, required=True, help='Json file containing training info')
    parser.add_argument('--do_cleanup', action="store_true", help='delete all clusters')

    # for baselines
    parser.add_argument('--cluster_config_json', type=str,
                        help='Json file containing info for the cluster setup. Provided when a baseline planner is used.')
    parser.add_argument('--varuna_profile_dir', type=str, default=None, help='If Varuna Planner is used, provide profile dir')
    parser.add_argument('--oobleck_profile_file', type=str, default=None, help='If Oobleck Planner is used, provide profile')
    parser.add_argument('--amp_profile_dir', type=str, default=None, help='If AMP Planner is used, provide profile dir')

    # for Sailor
    parser.add_argument('--sailor_profile_file_dir', type=str, default='sailor/Planner/sailor_planner/examples/',
                        help='Directory for SAILOR profiling')
    parser.add_argument('--sailor_profile_file_dir_bw', type=str,
                        default='sailor/Planner/sailor_planner/examples/cloud_profiling/networking/pods',
                        help='Directory for SAILOR profiling')
    parser.add_argument('--quotas_dict', type=str, default='',
                        help='Json file containing user quotas')
    parser.add_argument('--objective', type=str, default='',
                        help='User objective (throughput or cost)')
    parser.add_argument('--max_cost', type=int, default=0,
                        help='Max cost')
    parser.add_argument('--min_throughput', type=int, default=0,
                        help='MIn througput')

    args = parser.parse_args()
    serve(args)
