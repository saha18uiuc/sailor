import base64
import google.auth
import google.auth.transport.requests
import time
import yaml

from deepspeed.utils import logger
from google.cloud import container_v1
from kubernetes import client as k8s_client, utils
from tempfile import NamedTemporaryFile

GCP_OP_DONE_CODE = "3"


class GCPClient():
    """
    A wrapper class to create GCP resources. For now we assume that we have
    a service account attached to the VM running the master controller with
    the cloud-platform scope which is necessary such that this wrapper class
    can create/delete resources on behalf of the client

    Information about the google-cloud-container ClusterManager can be found
    https://cloud.google.com/python/docs/reference/container/latest/google.cloud.container_v1.services.cluster_manager
    """

    def __init__(self):
        self.client = container_v1.ClusterManagerClient()
        self.project = "ml-elasticity"

        # prefix for cluster names to distinguish with other clusters
        self.cluster_name_prefix = "mregion-"

    def get_cluster_identifier(self, cluster_name: str, zone: str) -> str:
        return f"projects/{self.project}/locations/{zone}/clusters/{cluster_name}"

    def get_operation_identifier(self, operation_name: str, zone: str) -> str:
        return f"projects/{self.project}/locations/{zone}/operations/{operation_name}"

    def wait_for_operation_to_complete(
        self,
        operation_name: str,
        zone: str,
        sleep_time: int = 30,
        timeout: int = 600
    ) -> bool:
        operation_identifier = self.get_operation_identifier(operation_name, zone)
        for i in range(0, timeout, sleep_time):
            response = self.client.get_operation(name=operation_identifier)
            if str(response.status.value) == GCP_OP_DONE_CODE:
                logger.info(f"{operation_name} in {zone} completed.")
                return True

            logger.info(f"Waiting for {operation_name} in {zone} to complete...")
            time.sleep(sleep_time)

        logger.info(f"Timed out waiting for {operation_name} in {zone} to complete.")
        return False

    def create_cluster(
        self,
        name: str,
        zone: str,
        machine_type: str = "n1-standard-4",
        image_type: str = "COS_CONTAINERD",
        network: str = "multi-region-vpc",
        release_channel: str = "REGULAR",
        workload_metadata: str = "GKE_METADATA",
        workload_pool: str = "ml-elasticity.svc.id.goog"
    ) -> str:

        assert name.startswith(self.cluster_name_prefix), "Cluster name must start with 'mregion-'"
        logger.info(f"Creating cluster {name} in {zone}...")

        cluster = {
            "name": name,
            "release_channel": {"channel": release_channel},
            "network": network,
            "initial_node_count": 1,
            "workload_identity_config": {"workload_pool": workload_pool},
            "node_config": {
                "machine_type": machine_type,  # controller machine type
                "image_type": image_type,
                "workload_metadata_config": {"mode": workload_metadata},
                "gcfs_config": {"enabled": True}  # enable image streaming
            }
        }

        response = self.client.create_cluster(
            cluster=cluster,
            parent=f"projects/ml-elasticity/locations/{zone}",
        )
        logger.info(response)
        return response.name

    def delete_cluster(self, name: str, zone: str):
        logger.info(f"Deleting cluster {name} in {zone}...")
        cluster_id = self.get_cluster_identifier(name, zone)
        response = self.client.delete_cluster(name=cluster_id)
        self.wait_for_operation_to_complete(response.name, zone)
        logger.info(response)

    def create_node_pool(
        self,
        cluster_name: str,
        zone: str,
        cluster_config: dict,
        image_type: str = "COS_CONTAINERD",
        workload_metadata: str = "GKE_METADATA"
    ) -> str:
        logger.info(f"Creating nodepool {cluster_name}-pool in {zone}...")
        node_pool = {
            "name": f"{cluster_name}-pool",
            "initial_node_count": cluster_config['num_nodes'],
            "config": {
                "machine_type": cluster_config['machine_type'],
                "disk_type": cluster_config['disk_type'],
                "disk_size_gb": cluster_config['disk_size'],
                "image_type": image_type,
                "spot": cluster_config['spot'],
                "workload_metadata_config": {"mode": workload_metadata},
                "gcfs_config": {"enabled": True},  # enable image streaming
                "accelerators": [{
                    "accelerator_count": cluster_config['accelerator_count'],
                    "accelerator_type": cluster_config['accelerator_type'],
                    # automatically install gpu driver
                    "gpu_driver_installation_config": {"gpu_driver_version": "LATEST"},
                }],
            },
            "autoscaling": {
                "enabled": True,
                "min_node_count": cluster_config['num_nodes'],
                "max_node_count": cluster_config['num_nodes']
            },
        }

        cluster_id = self.get_cluster_identifier(cluster_name, zone)
        response = self.client.create_node_pool(node_pool=node_pool, parent=cluster_id)
        logger.info(response)
        return response.name

    def clean_up_clusters(self):
        # get all existing cluster in all zones (-) of the project
        project_zones = f"projects/{self.project}/locations/-"
        response = self.client.list_clusters(parent=project_zones)

        # delete only clusters our clusters which have the correspondin prefix
        for cluster in response.clusters:
            if cluster.name.startswith(self.cluster_name_prefix):
                self.delete_cluster(cluster.name, cluster.location)

    def create_cluster_with_node_pool(self, cluster_config: dict):
        name = cluster_config['name']
        zone = cluster_config['zone']

        print("Create cluster ...")

        cluster_name = self.cluster_name_prefix + name
        operation_name = self.create_cluster(cluster_name, zone)
        self.wait_for_operation_to_complete(operation_name, zone)

        print("Create node pool ...")

        operation_name = self.create_node_pool(cluster_name, zone, cluster_config)
        self.wait_for_operation_to_complete(operation_name, zone)

        # # get a kubernetes client that has the cluster kubeconfig loaded
        kubernetes_client = self.get_k8s_client(cluster_name, zone)
        self.attach_default_service_account_to_pods(kubernetes_client)

        print("Create cluster controller and worker pods ...")

        # # # create cluster controller and worker agent pods
        # self.spin_up_worker_agents(kubernetes_client, cluster_config['num_nodes'])
        # time.sleep(20)  # wait for workers to get spawned
        # self.spin_up_cluster_controller(kubernetes_client)

    def create_clusters_from_config(self, clusters_config: list[dict]):

        for cluster_config in clusters_config:
            print(f"cluster config is {cluster_config}")
            self.create_cluster_with_node_pool(cluster_config)

    def get_k8s_client(self, cluster_name: str, zone: str) -> k8s_client.CoreApi:
        """
        https://github.com/googleapis/python-container/issues/6#issuecomment-970746358
        For a given cluster in a given zone, this function returns a k8s_client
        with the corresponding kubeconfig loaded, such that the client is able to
        create/delete pods in the GKE cluster
        """
        cluster_id = self.get_cluster_identifier(cluster_name, zone)
        response = self.client.get_cluster(name=cluster_id)

        creds, _ = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        configuration = k8s_client.Configuration()
        configuration.host = f'https://{response.endpoint}'

        with NamedTemporaryFile(delete=False) as ca_cert:
            ca_cert.write(base64.b64decode(response.master_auth.cluster_ca_certificate))
            configuration.ssl_ca_cert = ca_cert.name
        configuration.api_key_prefix['authorization'] = 'Bearer'
        configuration.api_key['authorization'] = creds.token

        return k8s_client.ApiClient(configuration)

    def attach_default_service_account_to_pods(
        self,
        kubernetes_client: k8s_client.CoreApi,
        service_account_name: str = "default",
        namespace: str = "default",
    ):
        """
        Given a k8s client with a loaded kubeconfig, it attaches the default GCP
        service account of the project to the pods, such that they can access
        necessary resources, e.g buckets
        """
        logger.info("Adding annotation to default service account...")
        v1_api = k8s_client.CoreV1Api(kubernetes_client)
        existing_service_account = v1_api.read_namespaced_service_account(
            name=service_account_name,
            namespace=namespace
        )
        existing_service_account.metadata.annotations = {
            "iam.gke.io/gcp-service-account": "37775916204-compute@developer.gserviceaccount.com"
        }
        v1_api.patch_namespaced_service_account(
            name=service_account_name,
            namespace=namespace,
            body=existing_service_account
        )

    def adapt_yaml(self, base_yaml_file: str, updated_yaml_file: str, num_replicas: int):
        with open(base_yaml_file, 'r') as f:
            base_config = yaml.safe_load_all(f.read())
        config1 = next(base_config)
        config2 = next(base_config)

        config2['spec']['replicas'] = num_replicas
        new_config = (config for config in [config1, config2])

        with open(updated_yaml_file, 'w') as f:
            yaml.safe_dump_all(documents=new_config, stream=f)

    def spin_up_cluster_controller(self, kubernetes_client: k8s_client.CoreApi):
        logger.info("Spinning up Cluster controller...")
        utils.create_from_yaml(
            k8s_client=kubernetes_client,
            yaml_file="config/cluster-controller.yaml",
            verbose=True,
            namespace="default"
        )

    def spin_up_worker_agents(self, kubernetes_client: k8s_client.CoreApi, num_replicas: int):
        logger.info("Adapting Worker yaml...")
        self.adapt_yaml(
            base_yaml_file="config/elastic-worker-ds.yaml",
            updated_yaml_file="config/elastic-worker-ds-from-planner.yaml",
            num_replicas=num_replicas,
        )
        logger.info("Spinning up Worker Agents...")
        utils.create_from_yaml(
            k8s_client=kubernetes_client,
            yaml_file="config/elastic-worker-ds-from-planner.yaml",
            verbose=True,
            namespace="default"
        )
