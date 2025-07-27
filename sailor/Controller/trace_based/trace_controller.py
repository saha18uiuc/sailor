import json
import time
import argparse
import grpc
from kubernetes import client, config
from sailor.protos import orchestration_pb2
from sailor.protos.orchestration_pb2_grpc import WorkerAgentStub
from sailor.Worker.elastic_worker_agent import WORKER_AGENT_PORT
from sailor.Controller.controller import NodeSet
from datetime import datetime
from deepspeed.utils import logger
from concurrent import futures

from sailor.examples.deepspeed.train_utils import TIMEOUT_NCCL

# This is the window time we will wait between 2 changes in the cluster
# to make sure the cluster is stable before sending the new configuration
# It will also be increased by the checkpointing frequency to make sure we get to the first iteration of checkpointing
# For now we do not support failure while reconfiguration is ongoing. We will add this in the future.
SAFE_PERIOD_MULTI_CHANGE = TIMEOUT_NCCL*5  # seconds
SAVE_EVERY = 25  # cehckpointing frequency (tierations)
# This is the time to kill the pods after a change in the cluster
GRACEFUL_PERIOD = 30


class TraceEndException(Exception):
    pass


class TraceNodeSet(NodeSet):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.store_port = 1234
        self.ckpt_port = 15000
        self.ondem_port = 16000
        self.initial_time = None
        self.next_trace_change_id = 0
        self.last_trace_change_timestamp = 0
        self.data_parallel_size = 1
        self.current_world_size = 0
        self.waiting_time = args.waiting_time
        # self.ondemand_global_ranks = []
        with open(args.trace_json_path, 'r') as f:
            self.trace = json.load(f)

    def update_nodes(self, pods, pods_cpu=None):
        """
        The available pods stay the same but we only use a subset of them
        according to the trace.
        """
        # TODO: OPTIMIZER
        # For now, every time there is an update, we will maintain world size = num_stages
        # with constant global batch size = 16 and micro batch size = 4
        # However, this might not be the most optimal configuration for the new cluster size,
        # i.e. minimize iteration process
        # In next steps we will add the optimizer algorithm to choose the most efficient combination
        # before sharing the new arguments with worker nodes here.
        current_timestamp = datetime.now().timestamp()

        if self.initial_time is None:
            # begin the first update
            self.initial_time = current_timestamp
            assert self.trace[self.next_trace_change_id]['GPUs'] >= self.args.num_stages, \
                "We are asking for a number of stages greater than available GPUs"
        elif current_timestamp - self.initial_time < self.trace[self.next_trace_change_id]['Time']:
            return

        num_training_nodes = self.trace[self.next_trace_change_id]['GPUs']
        if self.args.ckpt_cpu:
            num_ckpt_groups = self.args.num_ckpt_groups
            # NOTE: we need to have a number of training nodes divisible by the number of checkpointing groups
            num_training_nodes = num_training_nodes // num_ckpt_groups * num_ckpt_groups
            logger.info(f"Making sure the training nodes are divisible by number of checkpointing groups: {num_training_nodes}")

        if self.next_trace_change_id > 0:
            if num_training_nodes != self.current_world_size:
                pass
                # TODO: we do not wait anymore, we adapt the trace !
            else:
                self.next_trace_change_id += 1
                logger.info("The cluster size is the same, we do not send anything")
                return

            logger.info("Training already exists and new change is coming: update number of stages")
            start_graceful_period = time.time()
            self.args.num_stages = num_training_nodes * self.data_parallel_size
            logger.info(f"Current criteria for testing: we maintain the same data parallel size = {self.data_parallel_size} \
                        The updated number of stages is {self.args.num_stages}")
        else:
            start_graceful_period = time.time()
            self.args.num_stages = num_training_nodes * self.data_parallel_size
        # num_training_nodes = num_training_nodes // self.args.num_stages * self.args.num_stages
        training_pods = pods[:num_training_nodes]
        training_nodes = [orchestration_pb2.Node(ip_addr=p.status.pod_ip, port=WORKER_AGENT_PORT)
                          for p in training_pods]
        assert not self.args.ckpt_cpu or (len(training_nodes) > 1 and self.args.ckpt_cpu), \
            f"We need at least 2 nodes for training if we do checkpointing in CPU, we have {len(training_nodes)} nodes"
        if num_training_nodes > len(training_pods):
            num_training_nodes = len(training_pods)
            self.args.num_stages = num_training_nodes * \
                self.data_parallel_size  # keep same dp
        killing_pods = pods[num_training_nodes:]

        # List all CPU ondemand pods if applies
        training_cpu_nodes = []
        if pods_cpu is not None:
            training_cpu_pods = pods_cpu[:self.args.num_ckpt_groups]
            training_cpu_nodes = [orchestration_pb2.Node(ip_addr=p.status.pod_ip, port=WORKER_AGENT_PORT)
                                  for p in training_cpu_pods]

        self.next_trace_change_id += 1
        logger.info(
            f"At {self.next_trace_change_id-1} Update to {num_training_nodes} pods")
        if self.next_trace_change_id >= len(self.trace):
            logger.info("No more trace changes; kill the worker nodes")
            all_nodes = [orchestration_pb2.Node(ip_addr=p.status.pod_ip, port=WORKER_AGENT_PORT)
                         for p in pods]
            with futures.ThreadPoolExecutor(max_workers=20) as executor:
                for node in all_nodes:
                    executor.submit(self.send_kill_request, node)

            raise TraceEndException()
        if num_training_nodes == 0:
            logger.info("No training nodes available")
            return

        hparams = orchestration_pb2.HyperParams()

        # global_batch_size has to be divided by data_parallel_size; however this is not always possible
        # when indivisible, varuna uses global_batch_size // data_parallel_size * data_parallel_size
        # we apply the same logic
        self.data_parallel_size = num_training_nodes // self.args.num_stages
        micro_batch_size = self.args.micro_batch_size

        factor = micro_batch_size * self.data_parallel_size
        hparams.global_batch_size = self.args.global_batch_size // factor * factor
        hparams.micro_batch_size = micro_batch_size
        hparams.num_stages = self.args.num_stages
        hparams.cocktail_sgd = self.args.cocktail_sgd
        hparams.ckpt_cpu = self.args.ckpt_cpu

        self.store_port += 2
        self.ckpt_port += 1
        self.ondem_port += 1
        logger.info(f"training nodes contain {training_nodes}")
        logger.info(f"training cpu nodes contain {training_cpu_nodes}")
        if self.args.ckpt_cpu:
            from sailor.checkpoint import CheckpointPolicy
            assert len(
                training_nodes) == self.args.num_stages, "need to adapt ckpt groups to DP > 1"
            assert self.args.num_stages % args.num_ckpt_groups == 0, \
                "need to adapt ckpt groups to PP not divisible by num_ckpt_groups"
            policy_name = "basic"
            if self.data_parallel_size > 1:
                logger.info("DP > 1:we assign same PP stage to same group")
                policy_name = "dp"
            ckpt_policy = CheckpointPolicy(
                policy_name=policy_name,
                training_nodes=training_nodes, training_cpu_nodes=training_cpu_nodes,
                num_ckpt_groups=args.num_ckpt_groups,
                num_stages=self.args.num_stages,
                ckpt_port=self.ckpt_port,
                ondem_port=self.ondem_port,
            )
            logger.info(f"resulted ckpt_groups {ckpt_policy.ckpt_groups}")
        rank = 0
        with futures.ThreadPoolExecutor(max_workers=20) as executor:
            for node in training_nodes:
                # Worker configuration in the training distribution
                worker_configuration = orchestration_pb2.WorkerConfiguration(
                    rank=rank,
                    world_size=len(training_nodes),
                    master_ip=training_nodes[0].ip_addr,
                    master_port=str(self.store_port),
                    gname=f"group-{self.next_trace_change_id}"
                )
                if self.args.ckpt_cpu:
                    # Remote checkpointing distribution
                    my_ckpt_group = [
                        ckpt_group for ckpt_group in ckpt_policy.ckpt_groups if rank in ckpt_group][0]
                    remote_ckpt = orchestration_pb2.RemoteCkpt(
                        rank=str(my_ckpt_group.index(rank)),
                        master_ip=str(training_cpu_nodes[ckpt_policy.ckpt_groups.index(
                            my_ckpt_group)].ip_addr if training_cpu_nodes else training_nodes[my_ckpt_group[0]].ip_addr),
                        master_port=str(ckpt_policy.masters_port),
                        world_size=str(len(my_ckpt_group)),
                        group_id=str(
                            ckpt_policy.ckpt_groups.index(my_ckpt_group)),
                        num_ckpt_groups=str(args.num_ckpt_groups),
                        # always the first one
                        ondem_master_ip=training_cpu_nodes[0].ip_addr if training_cpu_nodes else training_nodes[0].ip_addr,
                        ondem_master_port=str(ckpt_policy.master_ondem_port)
                    )
                    logger.info(
                        f"""GLOBAL RANK {rank}: the ckpt group with id {remote_ckpt.group_id} is: rank {remote_ckpt.rank},
                        master_ip {remote_ckpt.master_ip}, master_port {remote_ckpt.master_port},
                        world_size: {remote_ckpt.world_size}"""
                    )
                    executor.submit(self.notify_topology_change, node,
                                    worker_configuration, hparams, node.ip_addr, node.port, remote_ckpt)
                else:
                    executor.submit(self.notify_topology_change,
                                    node, worker_configuration, hparams, node.ip_addr, node.port)
                rank += 1
            self.current_world_size = len(training_nodes)
            dem_idx = 0
            for cpu_node in training_cpu_nodes:
                worker_configuration = orchestration_pb2.WorkerConfiguration(
                    world_size=len(training_nodes),
                    master_ip=training_nodes[0].ip_addr,
                    master_port=str(self.store_port),
                    gname=f"group-{self.next_trace_change_id}"
                )
                assert self.args.ckpt_cpu, "CPU ondemand nodes are only used when ckpt_cpu is set"
                my_ckpt_group = ckpt_policy.ckpt_groups[dem_idx]
                remote_ckpt = orchestration_pb2.RemoteCkpt(
                    rank=str(0),
                    master_ip=str(cpu_node.ip_addr),
                    master_port=str(ckpt_policy.masters_port),
                    world_size=str(len(my_ckpt_group)),
                    group_id=str(ckpt_policy.ckpt_groups.index(my_ckpt_group)),
                    num_ckpt_groups=str(self.args.num_ckpt_groups),
                    ondem_master_ip=str(training_cpu_nodes[0].ip_addr),  # always the first one
                    ondem_master_port=str(ckpt_policy.master_ondem_port)
                )
                logger.info(f"for cpu node {cpu_node} the ckpt group is: \
                            rank {remote_ckpt.rank}, master_ip {remote_ckpt.master_ip}, \
                            master_port {remote_ckpt.master_port}, world_size: {remote_ckpt.world_size}")
                if self.args.num_ckpt_groups > 1:
                    logger.info(f"the on demand ckpt group is: \
                                world size {num_ckpt_groups}, my rank {ckpt_policy.ckpt_groups.index(my_ckpt_group)}, \
                                port and ip addr: {training_cpu_nodes[0].ip_addr}, {ckpt_policy.master_ondem_port}")
                executor.submit(self.notify_topology_change, cpu_node, worker_configuration, hparams, cpu_node.ip_addr, cpu_node.port, remote_ckpt)
                dem_idx += 1

        if len(killing_pods) != 0 and self.next_trace_change_id > 0:
            logger.info(f"Killing {len(killing_pods)} pods before graceful period ")
            print(time.time() - start_graceful_period, start_graceful_period, GRACEFUL_PERIOD)
            while (time.time() - start_graceful_period) < GRACEFUL_PERIOD:
                time.sleep(1)
                logger.info("Waiting to kill")
                continue
            logger.info(f"Killing {len(killing_pods)} pods after graceful period ")
            killing_nodes = [orchestration_pb2.Node(ip_addr=p.status.pod_ip, port=WORKER_AGENT_PORT)
                             for p in killing_pods]
            with futures.ThreadPoolExecutor(max_workers=20) as executor:
                for node in killing_nodes:
                    logger.info(f"Killing node {node} because cluster changed")
                    executor.submit(self.send_kill_request, node)
        return

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

    def check_nodes_ready(self, pnode_list, port):
        pnode_ready = [False]*len(pnode_list)
        training_nodes = [orchestration_pb2.Node(ip_addr=p.status.pod_ip, port=port)
                          for p in pnode_list]
        print(training_nodes)
        while True:
            for i in range(len(pnode_list)):
                if not pnode_ready[i]:
                    if self.check_if_ready(training_nodes[i]):
                        pnode_ready[i] = True
                    time.sleep(3)

            if pnode_ready.count(True) == len(training_nodes):
                return True

        return False


def main(args):
    config.load_config()
    v1 = client.CoreV1Api()
    worker_nodes = TraceNodeSet(args)

    # Check if all pods are ready when starting for the first time
    while True:
        ret = v1.list_pod_for_all_namespaces(
            label_selector="app=elastic-ml-worker",
            field_selector="status.phase=Running",
            watch=False)
        pods = ret.items
        if len(pods) < args.num_nodes:
            logger.info(f"Waiting for {args.num_nodes} pods to be ready")
            time.sleep(3)
        else:
            break

    # Check if all worker agents have started in each pod
    worker_nodes.check_nodes_ready(pods, WORKER_AGENT_PORT)

    # Check if all cpu ondemand pods are ready when starting for the first time (if applies)
    pods_cpu = None
    # if args.ondem_is_not_worker_agent:
    ret = v1.list_pod_for_all_namespaces(
        label_selector="app=ondem-ckpt-cpu",
        watch=False)
    if ret.items:
        while True:
            ret = v1.list_pod_for_all_namespaces(
                label_selector="app=ondem-ckpt-cpu",
                field_selector="status.phase=Running",
                watch=False)
            pods_cpu = ret.items
            if len(pods_cpu) < args.num_ckpt_groups:
                logger.info(
                    f"Waiting for {args.num_ckpt_groups} pods to be ready")
                time.sleep(3)
            else:
                break
        # Check if all worker agents have started in each pod
        worker_nodes.check_nodes_ready(pods_cpu, WORKER_AGENT_PORT)

    # Loop for updating nodes when configuration changes
    while True:
        try:
            worker_nodes.update_nodes(pods, pods_cpu)
        except TraceEndException:
            break
        time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trace controller')
    parser.add_argument('--num_nodes', type=int, required=True, help="number of nodes in the cluster")
    parser.add_argument('--num_stages', type=int, required=True, help="number of pipeline stages")
    parser.add_argument('--cocktail_sgd', action='store_true', help="use cocktail sgd")
    parser.add_argument('--global_batch_size', type=int, required=True, help="global batch size")
    parser.add_argument('--micro_batch_size', type=int, required=True, help="micro batch size")
    parser.add_argument('--ckpt_cpu', action='store_true', help="use checkpointing in cpu")
    parser.add_argument('--num_ckpt_groups', type=int, required=False, help="number of checkpointing groups")
    parser.add_argument('--trace_json_path', type=str,
                        default='Controller/trace_based/static_trace.json', help="path to the trace json file")
    parser.add_argument('--stateful_sampler', default=True, action='store_false', help="use stateful sampler")
    parser.add_argument('--waiting_time', type=int, default=0, help="waiting time before starting the training")
    args = parser.parse_args()
    main(args)
