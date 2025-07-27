# SAILOR: Cross Region ML training on transient resources

This README contains information about the SAILOR framework in the cross region training setting and more specifically about the master controller and all its components.

### Components of the master controller
The master controller is responsible for maintaining a consistent and coherent state and view of the overall training setup when training an ML model across potentially multiple regions. This includes keeping state of the different kubernetes clusters and the worker nodes that are part of it, the geographic location of the clusters as well as its health status. In addition the master controller is responsible for dynamically allocating and de-allocating resources in different regions based on user-defined input criteria as well as availability and price constraints of hardware resources. In order to fullfill the above constraints, the master controller is composed of the below listed components:

#### Trainig configuration
The training configuration can be found in the `training_configuration.py` file. This class keeps the state of all the kubernetes clusters, its cluster controllers and its worker nodes that are involved in the distributed training process. The class in addition contains some helper methods in order to broadcast topology information down to the individual clusters. In addition the training configuration class also hosts the [Distributed TCP Store](https://pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore) that will allow the communication between the worker nodes.

#### Cloud wrapper
The role of the cloud wrapper is to expose an abstract API to the master controller that allows it to dynamically allocate and deallocate resources with a given cloud provider. The method `create_cluster_with_node_pool` for example should spin up a k8s cluster in a specified region and create a node pool with spot GPUs that will host the workers. The idea is that this interface is abstract in the way that it should not matter what the underlying cloud provider is. For each cloud provider that we want to support we want to add a client/wrapper that exposes the same methods, such that the master controller can be intitialized with any cloud wrapper. For now we only provide a GCP client in the `gcp_client.py` file but in the future, one might easily want to add a `aws_wrapper.py` or `azure_wrapper.py` that exposes the exact same interface as everybody else.
TODO: we might want to define the API with an abstract class.

#### Cloud catalog
TODO: The idea of this component is to track the availability and prices of resources of cloud providers all over the world.

#### Helper classes
This repo also contains some helper classes like `k8s_cluster.py` whose purpose is to ease manipulation and operations on kubernetes clusters. For now they remain pretty empty, but we expect to fill them up as we go along in this project.

### How to run the master controller?
In order to run the master controller, navigate to the `config/run` directory and execute the `run_master_controller.sh` bash script.

#### Flow of events
1. Once the master controller is up and running, it will automatically spin up a kubernetes cluster in the `us-west1-a` zone. You can change this and add additional clusters if desired by changing the master controller constructor accordingly. It will automatically also create a node pool that uses spot GPUs once the cluster is up that will later on host the worker nodes.

2. Once the cluster and the node pool is created, the master controller will automatically start a cluster controller pod and worker agent pods in each of the created kubernetes clusters.

3. From here on the usual procedure takes of. The cluster controller will register to the master controller by sending regular heartbeats containing the topology of their cluster. The master controller on its turn will assign ranks to each of the nodes and send them along with the hyperparameters back to the respective cluster. The distributed training process will then begin.