# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
from typing import List
import time

from sailor.Planner.baselines.Metis.utils import GPUNode, DeviceType

ONE_GB = 1024 * 1024 * 1024

# Adapted to match SAILOR config - minimal adaptation
class GPUCluster:
    def __init__(self, cluster_config: dict, inter_network_config: dict, intra_network_config: dict, zone: str):

        print(cluster_config)
        # 1. just different types of GPUs, and how many they are available - gpus_per_node is given
        self.host_entries = {}
        idx = 0
        for gpu_type, gpu_type_info in cluster_config.items():
            for node_id in range(gpu_type_info["num_nodes"]):
                self.host_entries[idx] = {}
                self.host_entries[idx]["ip"] = str(idx) # not sure why this is needed, but they have it like that
                self.host_entries[idx]["gpu_type"] = gpu_type
                self.host_entries[idx]["num_device"] = gpu_type_info["gpus_per_node"]
                idx += 1

        idx = 0
        self.nodes_info = {}
        for gpu_type, gpu_type_info in cluster_config.items():
            for node_id in range(gpu_type_info["num_nodes"]):
                intra_bandwidth = 2**40 # just a big number
                gpus_per_node = gpu_type_info["gpus_per_node"]
                if gpus_per_node > 1:
                    intra_bandwidth = intra_network_config[gpu_type][str(gpus_per_node)][1]
                print(zone, gpu_type, gpus_per_node, inter_network_config[zone][gpu_type][str(gpus_per_node)][zone][gpu_type][str(gpus_per_node)][1])
                self.nodes_info[str(idx)] = {
                    "instance_type": gpu_type,
                    "inter_bandwidth": inter_network_config[zone][gpu_type][str(gpus_per_node)][zone][gpu_type][str(gpus_per_node)][1],
                    "intra_bandwidth": intra_bandwidth,
                    "memory": gpu_type_info["mem_per_gpu"] / ONE_GB
                }
                idx += 1

        self.nodes = {}
        for node_id, node_info in self.host_entries.items():
            self.nodes[node_id] = GPUNode(device_type=DeviceType.from_string(node_info["gpu_type"]),
                                          num_devices=node_info["num_device"])


        print(self.host_entries)


    def get_num_nodes(self) -> int:
        return len(self.nodes.keys())

    def get_num_nodes_by_device_type(self, device_type: str) -> int:
        return sum([self.nodes[node_id].num_devices for node_id in self.nodes.keys() if self.nodes[node_id].device_type.name == device_type])

    def get_num_devices_per_node(self) -> int:
        return self.nodes[0].num_devices

    def get_total_num_devices(self) -> int:
        num_devices_list = [self.nodes[node_id].num_devices for node_id in self.nodes.keys()]
        return sum(num_devices_list)

    def get_device_types(self) -> List[DeviceType]:
        return [self.nodes[node_id].device_type for node_id in self.nodes.keys()]

    def get_str_device_types(self) -> str:
        return '_'.join([device_type.name for device_type in set(self.get_device_types())])

    def get_device_memory(self, node_id: int) -> int:
        """
        returns the total memory size of a single GPU within node
        :param node_id:
        :return: Memory size in bytes
        """
        node_ip = self.host_entries[node_id]['ip']
        return self.nodes_info[node_ip]['memory'] * 1024

    def get_device_memory_for_device_type(self, device_type: str) -> int:
        device_type = device_type.replace("_", "-")
        for node_ip in self.nodes_info.keys():
            if device_type == self.nodes_info[node_ip]['instance_type']:
                return self.nodes_info[node_ip]['memory'] * 1024

    def get_intra_bandwidth(self, node_id: int) -> int:
        node_ip = self.host_entries[node_id]['ip']
        return self.nodes_info[node_ip]['intra_bandwidth']

    def get_inter_bandwidth(self, node_id: int) -> int:
        node_ip = self.host_entries[node_id]['ip']
        return self.nodes_info[node_ip]['inter_bandwidth'] # Fixed from original code
