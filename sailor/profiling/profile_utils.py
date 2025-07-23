from functools import partial
import torch
import json
from os.path import expanduser
import time
from math import ceil

import threading
from sailor.providers.gcp.model_networking import get_time_on_config, get_bw_on_config
from typing import Literal

# Supported Zones and GPU types
Zone = Literal["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f", "us-west1-b"]
GPU_Type = Literal["A100-40", "V100-16", "P100-16"]

# hold profile info per-layer

take_time_fwd_start_events = {}
take_time_fwd_end_events = {}
time_fwd = {}

take_time_bwd_start_events = {}
take_time_bwd_end_events = {}
time_bwd = {}

mem_per_layer_floats = {}  # mem needed for each layer in floats
params_per_layer_floats = {} # param size in floats
input_output_act_floats = {}  # tuple of (input activation size, output_activation size) in floats

mem_per_layer_bytes = {}  # mem needed for each layer in floats
params_per_layer_bytes = {} # param size in floats
input_output_act_bytes = {}  # tuple of (input activation size, output_activation size) in floats

alloc_res_mem = {} # tuple of (allocated mem, reserved mem)

intra_node_ar = {}
inter_node_ar = {}

class HookParams():
    def __init__(self) -> None:
        pass

    def set_params(self, first_layer, last_layer):
        self.first_layer = first_layer
        self.last_layer = last_layer

hook_params = HookParams()

def collect_mem_info(layers_list, params_mf, act_mf):
    mem_dict = {
        "params_floats": sum(params_per_layer_floats[l]*params_mf for l in layers_list),
        "act_output_floats": sum(input_output_act_floats[l][1] for l in layers_list),
        "act_input_floats": sum(input_output_act_floats[l][0] for l in layers_list),
        "act_mem_floats": sum(mem_per_layer_floats[l]*act_mf for l in layers_list),

        "params_bytes": sum(params_per_layer_bytes[l]*params_mf for l in layers_list),
        "act_output_bytes": sum(input_output_act_bytes[l][1] for l in layers_list),
        "act_input_bytes": sum(input_output_act_bytes[l][0] for l in layers_list),
        "act_mem_bytes": sum(mem_per_layer_bytes[l]*act_mf for l in layers_list),
    }
    return mem_dict


def get_bytes(object):
    if isinstance(object, torch.Tensor):
        return object.numel() * object.element_size()
    else:
        res = 0
        for t in object:
            res += get_bytes(t)
        return res

def get_numel(object):
    if isinstance(object, torch.Tensor):
        return object.numel()
    else:
        res = 0
        for t in object:
            res += get_numel(t)
        return res


def take_mem_oobleck_pre(layer, name, module, input):
    alloc_res_mem[name][0] = torch.cuda.memory_allocated()

def take_mem_oobleck(layer, name, module, input, output):
    cur = torch.cuda.memory_allocated()
    alloc_res_mem[name][0] = cur - alloc_res_mem[name][0]


def take_time_fwd_pre(layer, name, module, input):
    #print(f"FWD HOOK, layer is {name}")
    alloc_res_mem[name][0] = torch.cuda.memory_allocated()
    alloc_res_mem[name][1] = torch.cuda.memory_reserved()

    if name not in input_output_act_floats:
        input_output_act_floats[name] = [0,0]
        input_output_act_bytes[name] = [0,0]
        input_output_act_floats[name][0] = get_numel(input)
        input_output_act_bytes[name][0] = get_bytes(input)

    take_time_fwd_start_events[name] = torch.cuda.Event(enable_timing=True)
    take_time_fwd_start_events[name].record()


def take_time_and_mem_fwd(layer, name, module, input, output):

    end = torch.cuda.Event(enable_timing=True)
    take_time_fwd_end_events[name] = torch.cuda.Event(enable_timing=True)
    take_time_fwd_end_events[name].record()

    cur_mem_alloc = torch.cuda.memory_allocated()
    cur_mem_res = torch.cuda.memory_reserved()

    alloc_floats = (cur_mem_alloc - alloc_res_mem[name][0])//4
    reserved_floats = (cur_mem_res - alloc_res_mem[name][1])//4

    rank = torch.distributed.get_rank()
    #print(f"Rank {rank}, LAYER {name},  ALLOC FLOATS: {alloc_floats}, RESERVED FLOATS: {reserved_floats}")

    # set only at first iter (since we are looking at reserved memory):
    #if name not in mem_per_layer_floats:
    # due to diff in allocated and reserved
    mem_activation = cur_mem_alloc-alloc_res_mem[name][0] #max(cur_mem_alloc-alloc_res_mem[name][0], cur_mem_res-alloc_res_mem[name][1])
    mem_per_layer_floats[name] = mem_activation//4
    mem_per_layer_bytes[name] = mem_activation

    input_output_act_floats[name][1] = get_numel(output)
    input_output_act_bytes[name][1] = get_bytes(output)

    all_fwd = 0
    if name == hook_params.last_layer:
        for i, start in take_time_fwd_start_events.items():
            end = take_time_fwd_end_events[i]
            end.synchronize()
            time_sec_i = start.elapsed_time(end) / 1000
            if rank==0:
                print(f"Layer {i}, FWD took {time_sec_i} sec")
            time_fwd[i].append(time_sec_i)
            all_fwd += time_sec_i
        if rank == 0:
            print(f"FORWARD PASS TOOK {all_fwd} sec")


def take_time_bwd_pre(layer, name, module, input):
    take_time_bwd_start_events[name] = torch.cuda.Event(enable_timing=True)
    take_time_bwd_start_events[name].record()

def take_time_bwd(layer, name, module, input, output):
    take_time_bwd_end_events[name] = torch.cuda.Event(enable_timing=True)
    take_time_bwd_end_events[name].record()

    rank = torch.distributed.get_rank()

    all_bwd = 0
    if name == hook_params.first_layer:
        for i, start in take_time_bwd_start_events.items():
            end = take_time_bwd_end_events[i]
            end.synchronize()
            time_sec_i = start.elapsed_time(end) / 1000
            if rank==0:
                print(f"Layer {i} BWD took {time_sec_i} sec")
            time_bwd[i].append(time_sec_i)
            all_bwd += time_sec_i

        if rank==0:
            print(f"BACKWARD PASS TOOK {all_bwd} sec")


def take_time_bwd_llama(layer, name, module, input, output):
    # for llama, because the pre-hooks do not work for some reason
    # measure time between two consecutive layers
    take_time_bwd_end_events[name] = torch.cuda.Event(enable_timing=True)
    take_time_bwd_end_events[name].record()

    print(f"BWD HOOK, layer is {name}")

    if name == hook_params.first_layer:
        event_keys = sorted(take_time_bwd_end_events.keys())
        print(event_keys)
        for i in range(0, len(event_keys)-1):
            end = take_time_bwd_end_events[str(i)]
            end.synchronize()
            if i == len(event_keys)-2:
                time_sec_i = take_time_bwd_end_events['loss_fn'].elapsed_time(end) / 1000
            else:
                time_sec_i = take_time_bwd_end_events[str(i+1)].elapsed_time(end) / 1000
            print(f"Layer {i} BWD took {time_sec_i} sec")
            time_bwd[str(i)].append(time_sec_i)


def take_time_bwd_estimate(layer, name, module, input, output):
    # oobleck
    time_bwd[name] = time_fwd[name]*3


def add_hooks(model, pre_fwd_hook, fwd_hook, pre_bwd_hook=None, bwd_hook=None):

    for name, layer in model.named_children():
        alloc_res_mem[name] = [0, 0]
        params_per_layer_floats[name] = sum(p.numel() for p in layer.parameters())
        params_per_layer_bytes[name] = sum(p.numel() * p.element_size() for p in layer.parameters())

        print(f"---------- Name: {name}, params is {params_per_layer_floats[name]}")

        if name == "tied_modules": # Megatron-specific
            if "embed" in layer:
                layer = layer["embed"]
            else:
                continue

        layer.register_forward_pre_hook(
            partial(pre_fwd_hook, layer, name))
        layer.register_forward_hook(partial(fwd_hook, layer, name))
        if pre_bwd_hook:
            layer.register_full_backward_pre_hook(
                partial(pre_bwd_hook, layer, name))
        if bwd_hook:
            layer.register_full_backward_hook(partial(bwd_hook, layer, name))
        time_fwd[name] = []
        time_bwd[name] = []


def estimate_ar_time(tensor_size, num_workers, network_coeffs):
    if tensor_size==0 or num_workers <= 1:
        return 0
    part_time = get_time_on_config(tensor_size/num_workers, network_coeffs)
    mul_factor = 1 #2 if (num_workers % 2 == 0) else 3 # to simulate NCCL's lack of bidirectional communication
    total_time = 2 * (num_workers-1) * mul_factor * part_time
    return total_time


def estimate_send_time(tensor_size, network_coeffs):
    if tensor_size==0:
        return 0
    time = get_time_on_config(tensor_size, network_coeffs)
    return time

def find_bw(tensor_size, num_workers, network_coeffs):
    if tensor_size==0:
        return 0.0
    bw = get_bw_on_config(tensor_size/num_workers, network_coeffs)
    return bw
