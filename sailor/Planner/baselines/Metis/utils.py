# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union, List


def parse_hostfile(file_path: str) -> Dict[int, Dict[str, Union[str, int]]]:
    num_node, hostfile_info = 0, dict()
    with open(file_path, 'rt') as hostfile:
        line = hostfile.readline()
        while line:
            splitted_data = line.split(' ')
            ip = splitted_data[0]
            num_device = int(splitted_data[1][:1])

            print(ip, num_device)

            hostfile_info[num_node] = dict()
            hostfile_info[num_node]["ip"] = ip
            hostfile_info[num_node]["num_device"] = num_device

            line = hostfile.readline()
            num_node += 1

    return hostfile_info


def parse_nodefile(file_path: str) -> Dict[str, Dict[str, Union[str, int]]]:
    with open(file_path, 'r') as content:
        clusters = json.loads(content.read())

    return clusters


def factor(N: int, upper: int = None, lower: int = None) -> List:
    if upper is None:
        upper = N

    ret = []
    for i in range(1, upper + 1):
        if N % i == 0:
            if lower is None or i >= lower:
                ret.append(i)
    return ret


class DeviceType(Enum):
    A100_40 = "a100-40"
    V100_16 = "v100-16"
    P100_16 = "p100-16"
    T4_16 = "t4-16"
    A100_80 = "a100-80"
    RTX_2080 = "rtx-2080"
    RTX_3090 = "rtx-3090"
    Titan_RTX = "titan-rtx"
    GH_96 = "gh-96"

    @staticmethod
    def from_string(s: str) -> 'DeviceType':
        try:
            return DeviceType[s.replace("-", "_")]
        except KeyError:
            raise ValueError


@dataclass
class ResourceConfig:
    device_type: DeviceType
    inter_bw: int
    intra_bw: int
    num_nodes: int
    num_devices: int
    total_devices: int
    device_memory: int


@dataclass
class ModelConfig:
    num_layers: int
    hidden_size: int
    sequence_length: int
    vocab_size: int
    hidden_size: int
    attention_head_size: int
    model_name: str


@dataclass
class GPUNode:
    device_type: DeviceType
    num_devices: int
