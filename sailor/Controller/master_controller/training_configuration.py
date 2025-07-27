import grpc
import torch.distributed as dist
from concurrent import futures
from datetime import datetime, timedelta
from typing import List, Dict
from deepspeed.utils import logger
from k8s_cluster import K8sCluster as Cluster
from sailor.protos import orchestration_pb2, orchestration_pb2_grpc


class TrainingConfiguration():
    """
    This class contains the state of the training configuration and helper
    methods to propagate the training configuration to each of the clusters
    """

    def __init__(self, training_args: dict):
        self.clusters: Dict[str, orchestration_pb2.ClusterTopology] = {}

        self.master_ip: str = "master-controller.us-west1-a.c.ml-elasticity.internal"
        self.master_port: str = "50051"

        self.distributed_store = None
        self.store_port = 1234

        self.world_size = 0
        self.restart_count = 0
        self.num_ready_nodes = 0
        self.opt_params: orchestration_pb2.OptParams = None
        self.hyper_params: orchestration_pb2.HyperParams = None
        self.training_args = training_args

    @property
    def cluster_ids(self) -> List[str]:
        return list(self.clusters.keys())

    @staticmethod
    def send_topology_to_cluster_controller(dest_ip, dest_port, topology, hparams, retry=3):
        logger.info(f"Sending new topology to {dest_ip}...")
        request = orchestration_pb2.TopologyChangeRequest(
            topology=topology, hparams=hparams)
        grpc_target = f'{dest_ip}:{dest_port}'

        while True:
            try:
                with grpc.insecure_channel(grpc_target) as channel:
                    stub = orchestration_pb2_grpc.ClusterControllerStub(
                        channel)
                    stub.TopologyChange(request)
            except Exception as e:
                logger.error(f"Exception when calling TopologyChange: {e}")
                if retry > 0:
                    retry -= 1
                    continue
                else:
                    raise e
            else:
                break

    def update_hyper_params(self) -> None:
        world_size = self.num_ready_nodes - \
            (self.num_ready_nodes % self.training_args["num_stages"])
        if world_size != 0:
            data_parallel_size = world_size // self.training_args["num_stages"]
            factor = self.training_args["micro_batch_size"] * data_parallel_size
            global_batch_size = self.training_args["global_batch_size"] // factor * factor
        else:
            global_batch_size = 0

        self.world_size = world_size
        self.opt_params = orchestration_pb2.OptParams(
            lr=self.training_args["lr"],
            momentum=self.training_args["momentum"],
            wd=self.training_args["wd"]
        )

        self.hyper_params = orchestration_pb2.HyperParams(
            num_stages=self.training_args["num_stages"],
            cocktail_sgd=False,
            micro_batch_size=self.training_args["micro_batch_size"],
            global_batch_size=global_batch_size,
            opt_params=self.opt_params,
        )

    def broadcast_configuration_to_clusters(self):
        """
        Calculates the world size and sends each cluster its
        training configuration
        """
        logger.info("Broadcasting configuration to clusters...")
        self.update_hyper_params()
        num_updated_nodes = 0

        self.distributed_store = dist.TCPStore(
            "127.0.0.1", self.store_port, self.world_size, True, timedelta(seconds=30))

        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            for cluster_id, cluster in self.clusters.items():

                # need to make sure that we don't update more nodes than in our world size
                num_nodes_in_cluster = cluster.num_workers
                if num_updated_nodes + num_nodes_in_cluster <= self.world_size:
                    worker_nodes = cluster.worker_nodes
                    num_updated_nodes += num_nodes_in_cluster
                else:
                    remaining_num_nodes = self.world_size - num_updated_nodes
                    worker_nodes = cluster.worker_nodes[:remaining_num_nodes]
                    num_updated_nodes = self.world_size

                if len(worker_nodes) == 0:
                    continue

                topology = orchestration_pb2.Topology(
                    worker_nodes=worker_nodes,
                    master_ip=self.master_ip,
                    master_port=str(self.store_port),
                    world_size=self.world_size,
                )

                executor.submit(
                    self.send_topology_to_cluster_controller,
                    cluster.controller_ip,
                    cluster.controller_port,
                    topology,
                    self.hyper_params,
                )

        self.store_port += 1

    def re_initialize_worker_nodes(self):
        logger.info("Reinitiliazing ranks and group names of worker nodes...")
        rank = 0
        for cluster in self.clusters.values():
            for node in cluster.worker_nodes:
                node.gname = f"group-{self.restart_count}"
                node.rank = rank
                rank += 1

        self.num_ready_nodes = rank
        self.restart_count += 1

    def add_cluster(self, cluster: orchestration_pb2.ClusterTopology):
        """
        Adds a cluster to the topology if the cluster is not already present
        or the worker agent set in the cluster changed
        """
        topology_changed = False

        if cluster.cluster_id not in self.cluster_ids:
            logger.info(
                f"Adding new cluster {cluster.cluster_id} to topology...")
            topology_changed = True
        else:
            old_cluster = self.clusters[cluster.cluster_id]
            old_worker_nodes = {
                node.ip_addr for node in old_cluster.worker_nodes}
            new_worker_nodes = {node.ip_addr for node in cluster.worker_nodes}
            if old_worker_nodes != new_worker_nodes:
                logger.info(f"Updating cluster {cluster.cluster_id}...")
                topology_changed = True

        if topology_changed:
            self.clusters[cluster.cluster_id] = Cluster(
                cluster_id=cluster.cluster_id,
                controller_ip=cluster.cluster_controller_ip,
                controller_port=cluster.cluster_controller_port,
                worker_nodes=cluster.worker_nodes,
                last_heartbeat=datetime.now(),
            )
            self.re_initialize_worker_nodes()
            self.broadcast_configuration_to_clusters()
        else:
            logger.info(f"Cluster {cluster.cluster_id} remains unchanged")

    def remove_cluster(self, cluster_id: str):
        self.remove_clusters([cluster_id])

    def remove_clusters(self, cluster_ids: List[str]):
        logger.info(f"Removing clusters {cluster_ids} from topology...")
        for cluster_id in cluster_ids:
            self.clusters.pop(cluster_id)
        self.re_initialize_worker_nodes()
        self.broadcast_configuration_to_clusters()

    def remove_outdated_clusters(self):
        """
        Checks each cluster's last heartbeat and removes the cluster
        if its too old
        """
        logger.info("Checking for outdated clusters...")
        current_time = datetime.now()
        clusters_to_remove = []
        for cluster in self.clusters.values():
            time_difference = current_time - cluster.last_heartbeat
            if time_difference > timedelta(seconds=60):
                clusters_to_remove.append(cluster.id)

        if len(clusters_to_remove) != 0:
            self.remove_clusters(clusters_to_remove)
