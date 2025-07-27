from datetime import datetime
from typing import List
from sailor.protos import orchestration_pb2


class K8sCluster():
    """
    This class is a wrapper for describing a kubernetes cluster
    """
    def __init__(
            self,
            cluster_id: str,
            controller_ip: str,
            controller_port: str,
            worker_nodes: List[orchestration_pb2.Node],
            last_heartbeat: datetime
    ):
        self.id = cluster_id
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.worker_nodes = worker_nodes
        self.last_heartbeat = last_heartbeat

    @property
    def num_workers(self):
        return len(self.worker_nodes)
