import grpc
import threading
import time
import uuid
from concurrent import futures
from deepspeed.utils import logger
from grpc import RpcError
from kubernetes import client, config
from sailor.Controller.controller import NodeSet
from sailor.protos import orchestration_pb2, orchestration_pb2_grpc
from sailor.protos.orchestration_pb2_grpc import (
    WorkerAgentStub, MasterControllerStub
)
from sailor.protos.orchestration_pb2 import ClusterTopology

MASTER_CONTROLLER_IP = "master-controller.us-west1-a.c.ml-elasticity.internal"
MASTER_CONTROLLER_PORT = "50051"
CLUSTER_CONTROLLER_PORT = "50051"
WORKER_AGENT_PORT = "50051"


class ClusterController(NodeSet):
    """ Cluster Controller has two main tasks:
    1. run a gRPC endpoint which receives new topology updates from the master controller
    2. periodically fetch all available nodes in cluster and send them to the master controller

    Each task is run in a separate thread
    """

    def __init__(self) -> None:
        # initialize kubernetes client
        super().__init__()
        config.load_config()
        self.k8s_client = client.CoreV1Api()

        cluster_ctr_pod = self._get_cluster_pods(
            label="elastic-ml-controller")
        cluster_nodes = self._get_cluster_nodes()
        assert len(cluster_ctr_pod) == 1, "More than 1 cluster controller"

        self.cluster_id = str(uuid.uuid4())
        self.cluster_controller_ip = cluster_ctr_pod[0].status.pod_ip
        self.cluster_controller_port = CLUSTER_CONTROLLER_PORT
        self.world_size = 0
        self.training_node_names = {}
        self.node_names = {}
        self.node_objs = []
        self.disk_vm_map = {}
        self.curr_vm_list = [
            node.metadata.name for node in cluster_nodes if node.status.conditions[-1].reason == 'KubeletReady'
        ]

        self.master_controller_grpc_channel = grpc.insecure_channel(
            f"{MASTER_CONTROLLER_IP}:{MASTER_CONTROLLER_PORT}")
        self.master_controller_stub = MasterControllerStub(
            self.master_controller_grpc_channel)

        # schedule job to send periodic heartbeats to master
        self.heartbeat_thread = threading.Thread(
            target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def check_ready(self, node):
        request = orchestration_pb2.CheckReadyRequest(is_ready=1)
        grpc_target = f'{node.ip_addr}:{node.port}'
        with grpc.insecure_channel(grpc_target) as channel:
            stub = WorkerAgentStub(channel)
            stub.CheckReady(request)

    def check_if_ready(self, node):
        try:
            self.check_ready(node)
        except Exception:
            logger.error(f"Worker {node.ip_addr} is not ready")
            return False

        return True

    def _get_cluster_nodes(self):
        return self.k8s_client.list_node().items

    def _get_cluster_pods(self, label):
        pods = self.k8s_client.list_pod_for_all_namespaces(
            label_selector=f"app={label}",
            field_selector="status.phase=Running",
            watch=False
        )
        return pods.items

    def send_heartbeat(self, sleep_time=20):
        """Periodically fetches all ready worker pods in cluster and sends
        them to the master controller"""
        while True:
            time.sleep(sleep_time)

            logger.info("Sending heartbeat to master...")
            pods = self._get_cluster_pods(label="elastic-ml-worker")
            available_pods_in_cluster = []

            for pod in pods:
                pnode = orchestration_pb2.Node(
                    ip_addr=pod.status.pod_ip,
                    port=WORKER_AGENT_PORT,
                    name=pod.metadata.name,
                    namespace=pod.metadata.namespace,
                    cluster_id=self.cluster_id
                )
                if self.check_if_ready(pnode):
                    # inform master controller only about available nodes in cluster
                    available_pods_in_cluster.append(pnode)

            try:
                cluster_topology = ClusterTopology(
                    cluster_id=self.cluster_id,
                    cluster_controller_ip=self.cluster_controller_ip,
                    cluster_controller_port=self.cluster_controller_port,
                    worker_nodes=available_pods_in_cluster,
                )
                self.master_controller_stub.Heartbeat(cluster_topology)
            except RpcError as e:
                print(f"Master Controller not available {e}")

    def TopologyChange(self, request, context):
        # pylint: disable=unused-argument
        logger.info("Push new topology down to worker nodes...")
        topology = request.topology
        hyper_params = request.hparams

        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            for node in topology.worker_nodes:
                logger.info(f"to worker node {node.cluster_id}")
                if node.cluster_id != self.cluster_id:
                    # node is not part of current cluster
                    continue

                worker_configuration = orchestration_pb2.WorkerConfiguration(
                    rank=node.rank,
                    world_size=topology.world_size,
                    master_ip=topology.master_ip,
                    master_port=topology.master_port,
                    gname=node.gname
                )

                hparams = orchestration_pb2.HyperParams()
                hparams.CopyFrom(hyper_params)

                executor.submit(self.notify_topology_change,
                                node, worker_configuration, hparams, node.ip_addr, node.port)
                logger.info(f"Sent training info to node: {node.ip_addr}")

        logger.info("Terminated Topology change")
        return orchestration_pb2.TopologyChangeResponse()


def run_grpc_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    orchestration_pb2_grpc.add_ClusterControllerServicer_to_server(
        ClusterController(), server)
    server.add_insecure_port(f"[::]:{CLUSTER_CONTROLLER_PORT}")
    logger.info("starting server")
    server.start()
    server.wait_for_termination()
    logger.info("server shut down")


if __name__ == '__main__':
    run_grpc_server()
