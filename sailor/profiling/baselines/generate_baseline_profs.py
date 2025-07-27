import sys
import json
import numpy as np
import copy
import os
import argparse
from pathlib import Path

from os.path import expanduser
from sailor.providers.gcp.model_networking import get_time_on_config
from sailor.Planner.simulations.constants import GPU_MEMORY_GB
from sailor.profiling.profile_utils import estimate_ar_time

ONE_MB = 1024 * 1024
ONE_GB = 1024 * 1024 * 1024


# from Galvatron
def total_memcost(pp_deg, layernum, layertype, per_layer_cost, stage_idx):
    layer_costs = []
    for l in range(layertype):
        layer_costs += [per_layer_cost[l]] * layernum
    total_layer_num = layertype * layernum
    avg_layer_num = int(total_layer_num // pp_deg)
    last_layer_num = total_layer_num - avg_layer_num * (pp_deg-1)
    pp_divide = [avg_layer_num] * (pp_deg-1) + [last_layer_num]
    return np.sum(layer_costs[int(np.sum(pp_divide[:stage_idx])):int(np.sum(pp_divide[:stage_idx+1]))])


def layer_mem_cost(params, activation_params, fp_size, mbs, optimizer):

    if optimizer == 'sgd':
        # sgd saves only a copy of model parameters in fp32
        memory_multiplier_optim = 4*1  # bytes
    else:
        # this works for fp16
        memory_multiplier_optim = 4*2  # bytes - only 2 keys in state dict
    model_copy = 4  # keep model in fp32
    additional_ds_copies = 2 * 4  # Deepspeed creates 2 additional copies of the model (start of the training)
    gradients = 4
    comm = 4

    mf = memory_multiplier_optim + model_copy + gradients + comm
    model_bytes = params * mf
    activation_bytes = activation_params * mbs * fp_size
    return model_bytes + activation_bytes

def get_ar_bw(world_size, gpus_per_node, network_coeffs, intra_network_coeffs):
    # Bandwidth in GB/sec
    min_size = 16 * ONE_MB
    sizes = [min_size*pow(2, x) for x in range(0, 5)]

    non_cons = []
    cons = []
    for size in sizes:
        non_cons_time = 2 * (world_size-1) * get_time_on_config(size/world_size, network_coeffs)
        non_cons_bw = (size/non_cons_time) / ONE_GB
        if world_size <= gpus_per_node:
            cons_time = 2 * (world_size-1) * get_time_on_config(size/world_size, intra_network_coeffs)
            cons_bw = (size/cons_time) / ONE_GB
        else:
            cons_bw = non_cons_bw
        non_cons.append(non_cons_bw)
        cons.append(cons_bw)

    avg_non_cons = np.average(non_cons)
    avg_cons = np.average(cons)

    return avg_non_cons, avg_cons


def get_p2p_bw(world_size, network_coeffs):
    min_size = 16 * ONE_MB
    sizes = [min_size*pow(2, x) for x in range(0, 5)]

    p2p_bws = []

    for size in sizes:
        # TODO: what to do when world_size > 2?
        p2p_time = get_time_on_config(size, network_coeffs)
        p2p_bw = (size/p2p_time) / ONE_GB
        p2p_bws.append(p2p_bw)

    avg_p2p_bw = np.average(p2p_bws)
    return avg_p2p_bw


def get_hw_prof_galvatron(max_num_nodes, gpus_per_node, network_coeffs, intra_network_coeffs, gpu_type):
    comm_prof = {}

    with open(f"{expanduser('~')}/sailor/sailor/Planner/baselines/Galvatron/overlap_coe_dicts.json", 'r') as f:
        galvatron_overlap_coe_dict = json.load(f)

    for num_gpus in range(2, max_num_nodes+1):
        ar_bw_non_cons, ar_bw_cons = get_ar_bw(num_gpus, gpus_per_node, network_coeffs, intra_network_coeffs)
        p2p_bw = get_p2p_bw(num_gpus, network_coeffs)

        comm_prof[f"allreduce_size_{num_gpus}_consec_0"] = ar_bw_non_cons
        comm_prof[f"allreduce_size_{num_gpus}_consec_1"] = ar_bw_cons
        comm_prof[f"p2p_size_{num_gpus}"] = p2p_bw
        comm_prof[f"overlap_coe"] = galvatron_overlap_coe_dict[gpu_type][str(gpus_per_node)]

    return comm_prof


def generate_profile_Galvatron(timing_info, mem_info, max_num_nodes, gpus_per_node, network_coeffs, intra_network_coeffs, gpu_type, generate_hw_prof=True):

    print(f"Network coeffs is {network_coeffs}, gpu_type is {gpu_type}")

    # for some reason, considers only the transformer
    tmp_keys = list(timing_info["1"].keys())
    min_tmp = min(tmp_keys)
    prof = timing_info["1"][min_tmp]
    last_layer = len(prof.keys())-1

    print(f"last_layer is {last_layer}")

    time_profile = {
        "layertype_0": prof["1"][0]*1000, # fwd only, in ms
        "layertype_other_0": (prof["0"][0]+prof[str(last_layer)][0])*1000
    }

    if args.optimizer == 'sgd':
        # sgd saves only a copy of model parameters in fp32
        memory_multiplier_optim = 4*1  # bytes
    else:
        # this works for fp16
        memory_multiplier_optim = 4*2  # bytes - only 2 keys in state dict
    model_copy = 4  # keep model in fp32
    gradients = 4
    comm = 4

    mf = memory_multiplier_optim + model_copy + gradients + comm

    print(mem_info.keys())

    min_tmp_str = str(min_tmp)
    mem_info_tp1 = mem_info[min_tmp_str]
    num_layers = len(mem_info_tp1.keys())
    mem_dict = {}
    transformer_size_total = mf * mem_info_tp1["1"]["params_floats"]
    transformer_size_total /= ONE_MB
    mem_dict['parameter_size'] = transformer_size_total
    mem_dict['tp_activation_per_bsz_dict'] = {}
    for tp in [1, 2, 4]:
        mem_dict['tp_activation_per_bsz_dict'][tp] = mem_info[str(tp)]["1"]["act_mem_bytes"] / ONE_MB
    mem_profile = {'layertype_0': mem_dict}

    # Taken from Galvatron - this takes into account extra memory in first and last layers

    mem_info = mem_info_tp1
    other_ms_first = mf * mem_info["0"]["params_floats"] / ONE_MB # embedding params
    other_act_first = mem_info["0"]["act_mem_bytes"] / ONE_MB # embedding act, in bytes

    other_ms_last = mf * mem_info[str(num_layers-1)]["params_floats"] / ONE_MB # final layer params
    other_act_last = mem_info[str(num_layers-1)]["act_mem_bytes"] / ONE_MB # final layer act, in bytes

    off_mem_dict = {
        "model_states": other_ms_first,
        "activation": other_act_first
    }
    on_last_mem_dict = {
        "model_states": other_ms_last,
        "activation": other_act_last
    }
    on_first_mem_dict = {
        "model_states": other_ms_first,
        "activation": other_act_first
    }

    mem_profile["other_memory_pp_off"] = off_mem_dict
    mem_profile["other_memory_pp_on_last"] = on_last_mem_dict
    mem_profile["other_memory_pp_on_first"] = on_first_mem_dict

    if generate_hw_prof:
        hw_prof_galvatron = get_hw_prof_galvatron(max_num_nodes, gpus_per_node, network_coeffs, intra_network_coeffs, gpu_type)
    else:
        hw_prof_galvatron = {}

    return [time_profile, mem_profile, hw_prof_galvatron]


def generate_profile_Varuna(timing_info, mem_info, num_layers, gpus_per_node, act_mem_file):
    # No TMP here

    with open(act_mem_file, 'r') as f:
        act_mem_profile = json.load(f)

    tmp_keys = list(mem_info.keys())
    mem_info = mem_info[str(gpus_per_node)]

    profile_dict = {}
    for layer in range(0, num_layers):
        layer_str = str(layer)
        layer_dict = {}
        for mbs in timing_info:
            if (str(gpus_per_node) in timing_info[mbs]) and (str(mbs) in act_mem_profile):
                mbs_int = int(mbs)
                timing_info_tmp = timing_info[mbs][str(gpus_per_node)]
                mbs_dict = {
                    "forward": timing_info_tmp[layer_str][0],
                    "backward": timing_info_tmp[layer_str][1],
                    "params_bytes": mem_info[layer_str]["params_bytes"],
                    "mem_required": act_mem_profile[str(mbs)]["4"][layer_str],
                    "output_activation": (mem_info[layer_str]["act_output_bytes"] * mbs_int)/gpus_per_node
                }
                layer_dict[mbs] = mbs_dict
            else:
                layer_dict[mbs] = {}
        profile_dict[layer] = layer_dict

    return profile_dict


def generate_profile_AMP(timing_info, num_layers):
    profiles = {}

    timing_info_bs1 = timing_info["1"]
    for tmp in timing_info_bs1.keys():
        # only forward
        sim_list = [timing_info_bs1[tmp][str(layer)][0] for layer in range(num_layers)]
        profiles[tmp] = sim_list
        print(tmp, profiles[tmp])

    print(timing_info_bs1)

    return profiles


def generate_profile_Oobleck(timing_info, mem_info, num_layers, max_num_nodes, network_coeffs, gpus_per_node, act_mem_file):

    # act_mem_file contains activation memory for all layers using OObleck-style profiling

    profiles = {}
    tmp_keys = list(mem_info.keys())
    mem_info = mem_info[str(gpus_per_node)]

    with open(act_mem_file, 'r') as f:
        act_mem_profile = json.load(f)

    for mbs_str, timing_info_mbs in timing_info.items():
        mbs = int(mbs_str)
        profile = {}

        profile['min_num_nodes'] = 1
        profile['max_num_nodes'] = max_num_nodes
        profile['num_gpus_per_node'] = gpus_per_node
        profile['num_layers'] = num_layers

        profile['data'] = {}

        if str(gpus_per_node) in timing_info_mbs:
            timing_info_tmp = timing_info_mbs[str(gpus_per_node)] # it divides with tmp
            for layer in range(0, num_layers):
                layer_profile = {}
                layer_profile["ar_node"] = 0
                layer_str = str(layer)
                ar_dict = {}
                for n in range(1, max_num_nodes+1):
                    ar_dict[n] = 2 * (n-1) * get_time_on_config(mem_info[layer_str]["params_bytes"]/n, network_coeffs)
                layer_profile["forward"] = timing_info_tmp[layer_str][0]
                layer_profile["backward"] = timing_info_tmp[layer_str][1]
                layer_profile["ar_across"] = ar_dict
                act = int(act_mem_profile["1"][layer_str]/gpus_per_node)
                layer_profile["mem_required"] = [mem_info[layer_str]["params_bytes"], act*mbs]

                profile['data'][layer] = layer_profile
        profiles[mbs_str] = profile

    return profiles


def generate_profile_Piper(timing_info, mem_info, num_layers, gpu_type, model_name):
    profiles = {}

    # for memory, Piper uses the formula a*(s/d)+b, where s,d are data-parallel degrees, and a and b are coefficients
    # We used synchronous 1F1B, thus for us, a=activations, b=model weigths + optimizer state + temporary buffer
    # Piper does not explicitely say how to compute a and b, thus we compute them based on the Sailor logic

    for mbs_str, timing_info_mbs in timing_info.items():
        mbs = int(mbs_str)
        profile = {}
        valid_tmp = list(timing_info_mbs.keys())
        min_tmp = min(list(mem_info.keys()))

        if args.optimizer == 'sgd':
            # sgd saves only a copy of model parameters in fp32
            memory_multiplier_optim = 4*1  # bytes
        else:
            # this works for fp16
            memory_multiplier_optim = 4*2  # bytes - only 2 keys in state dict
        model_copy = 4  # keep model in fp32
        gradients = 4
        comm = 4

        mf = memory_multiplier_optim + model_copy + gradients + comm

        # set at runtime, no need to change
        profile["maxDevices"] = 1
        profile["maxMemoryPerDevice"] = GPU_MEMORY_GB[gpu_type] * 1024 * 1024 * 1024
        profile["bandwidth"] = 1
        profile["mbsInBatch"] = 1

        profile["nodes"] = []
        for layer in range(num_layers):
            layer_str = str(layer)
            if layer == 0:
                name = "Embedding"
            elif layer == num_layers-1:
                name = "Head"
            else:
                if ('LLAMA' in model_name) and (layer == num_layers-2):
                    name = "RMS"
                else:
                    name = "Transformer"
            new_node = {
                "id": layer,
                "name": name
            }

            TMPC = {}
            for tmp in valid_tmp:

                if (tmp >= '9'):
                    continue

                params = mem_info[tmp][layer_str]["params_floats"]
                params_bytes = mem_info[tmp][layer_str]["params_bytes"]
                act_bytes = mem_info[tmp][layer_str]["act_mem_bytes"] * mbs

                fwd = timing_info_mbs[tmp][layer_str][0]
                bwd = timing_info_mbs[tmp][layer_str][1]
                TMPC[tmp] = [
                    {
                        "id": "vanilla",
                        "timePerSample": fwd + bwd,
                        "parameterSize": params_bytes,
                        "memoryUsageA": act_bytes, # activations
                        "memoryUsageB": params * mf,  # optimizer + temp buffer
                        "syncTimeFw": {} if layer == 0 else {str(layer-1): 0},
                        "syncTimeBw": {} if layer == num_layers-1 else {str(layer+1): 0}
                    }
                ]

            new_node["TMPCs"] = TMPC
            profile["nodes"].append(new_node)

        profile["edges"] = []
        for layer in range(num_layers-1):
            output_act = mem_info[min_tmp][str(layer)]["act_output_bytes"]

            new_edge = {
                "sourceId": layer,
                "destId": layer+1,
                "communicationCost": output_act  # TODO
            }
            profile["edges"].append(new_edge)

        profiles[mbs_str] = profile

    return profiles

def generate_profile_Metis(timing_info, mem_info, num_layers, max_bw, gpu_type, model_name, optimizer, fp_size):
    profile = {}

    for mbs, timing_info_mbs in timing_info.items():
        prof_mbs = {}
        mbs_int = int(mbs)

        for tmp, timing_info_mbs_tmp in timing_info_mbs.items():

            if (tmp >= '5'):
                continue

            prof_mbs_tmp = {}

            total_parameters_bytes = 0
            parameters_per_layer_bytes = []
            activation_parameters_bytes = []
            print(f"tmp is {tmp}")
            mem_info_tmp = mem_info[str(tmp)]

            for _, layer_info in mem_info_tmp.items():
                total_parameters_bytes += layer_info["params_bytes"]
                parameters_per_layer_bytes.append(layer_info["params_bytes"])
                # "An array representing the bytes of activation parameters AFTER each layer(in bytes)"
                # TODO: do we need to split activation on TP?
                activation_parameters_bytes.append(layer_info["act_output_bytes"] * mbs_int)

            print(mbs, tmp, activation_parameters_bytes)

            model_info = {}
            model_info["model_name"] = model_name
            model_info["num_layers"] = num_layers
            model_info["parameters"] = {
                "total_parameters_bytes": total_parameters_bytes,
                "parameters_per_layer_bytes": parameters_per_layer_bytes,
                "activation_parameters_bytes": activation_parameters_bytes
            }

            ######################################################################################

            layer_compute_total_ms = []
            total_time_ms = 0.0
            optimizer_time_ms = 0.0
            forward_backward_time_ms = 0.0
            for layer in range(num_layers):
                layer_str = str(layer)
                fwd_ms = timing_info_mbs_tmp[layer_str][0] * 1000
                bwd_ms = timing_info_mbs_tmp[layer_str][1] * 1000
                update_ms = timing_info_mbs_tmp[layer_str][2]*1000
                layer_compute_ms = fwd_ms + bwd_ms

                optimizer_time_ms += update_ms
                forward_backward_time_ms += layer_compute_ms
                total_time_ms += (layer_compute_ms+update_ms)
                layer_compute_total_ms.append(layer_compute_ms)


            execution_time_info = {}
            execution_time_info["total_time_ms"] = total_time_ms # TODO: How is 'iteration' defined here? One global batch size?
            execution_time_info["forward_backward_time_ms"] = forward_backward_time_ms
            execution_time_info["batch_generator_time_ms"] = 0.0 # TODO
            # if tmp==1:
            #     execution_time_info["layernorm_grads_all_reduce_time_ms"] = 0 # not clear if intra-node or inter-node
            #     execution_time_info["embedding_grads_all_reduce_time_ms"] = 0
            # else:
            #     execution_time_info["layernorm_grads_all_reduce_time_ms"] = estimate_ar_time(mem_info["0"]["params_bytes"], tmp, network_coeffs["intra"][gpu_type][tmp]) # not clear if intra-node or inter-node
            #     execution_time_info["embedding_grads_all_reduce_time_ms"] = estimate_ar_time(mem_info[str(num_layers-1)]["params_bytes"], tmp, network_coeffs["intra"][gpu_type][tmp])
            execution_time_info["optimizer_time_ms"] = optimizer_time_ms
            execution_time_info["layer_compute_total_ms"] = layer_compute_total_ms

            ######################################################################################

            layer_memory_total_mb = []
            #total_memory_mb = 0.0

            for _, layer_info in mem_info_tmp.items():
                params = layer_info["params_floats"]
                activation_params = layer_info["act_mem_floats"]
                # since it is nor clear from the description: https://github.com/SamsungLabs/Metis?tab=readme-ov-file#3-measuring-memory-usage,
                # we use our own method
                total_layer_mem = layer_mem_cost(params, activation_params, fp_size, mbs_int, optimizer)
                total_layer_mem_mb = total_layer_mem / ONE_MB
                #total_memory_mb += total_layer_mem_mb
                layer_memory_total_mb.append(total_layer_mem_mb)


            execution_memory_info = {}
            #execution_memory_info["total_memory_mb"] = total_memory_mb
            execution_memory_info["layer_memory_total_mb"] = layer_memory_total_mb

            prof_mbs_tmp["model"] = model_info
            prof_mbs_tmp["execution_time"] = execution_time_info
            prof_mbs_tmp["execution_memory"] = execution_memory_info

            prof_mbs[tmp] = prof_mbs_tmp

        profile[mbs] = prof_mbs

        print(mbs, profile[mbs])

    return profile



def main(args):
    with open(args.path_sim, 'r') as f:
        path_dict = json.load(f)
    timing_info = path_dict[args.model][args.gpu_type]

    with open(args.path_mem, 'r') as f:
        mem_dict = json.load(f)
    mem_info = mem_dict[args.model]

    with open(args.network_coeff_path, 'r') as f:
        network_coeffs = json.load(f)

    network_coeffs_inter_spec = network_coeffs["inter"][args.gpu_type][str(args.gpus_per_node)][0]
    network_coeffs_intra_spec = network_coeffs["intra"][args.gpu_type]["2"][0]

    max_bw = network_coeffs["inter"][args.gpu_type]["1"][1]

    home_dir = expanduser("~")

    if args.planner == "Varuna":
        profile = generate_profile_Varuna(timing_info, mem_info, args.num_layers, args.gpus_per_node, args.extra_mem_file)
    elif args.planner == "AMP":
        profiles = generate_profile_AMP(timing_info, args.num_layers)
        par_dir = f'{home_dir}/sailor/sailor/Planner/baselines/AMP/profiles/{args.model}/{args.gpu_type}'
        Path(par_dir).mkdir(parents=True, exist_ok=True)
        for tmp, sim_list in profiles.items():
            with open(f"{par_dir}/profile_{tmp}.npy", 'wb') as f:
                np.save(f, np.asarray(sim_list))
    elif args.planner == "Oobleck":
        profile = generate_profile_Oobleck(timing_info, mem_info, args.num_layers, args.max_num_nodes, network_coeffs_inter_spec, args.gpus_per_node, args.extra_mem_file)
    elif args.planner == "Piper":
        profile = generate_profile_Piper(timing_info, mem_info, args.num_layers, args.gpu_type, args.model)
    elif args.planner == "Metis":
        profile = generate_profile_Metis(timing_info, mem_info, args.num_layers, max_bw, args.gpu_type, args.model, args.optimizer, args.fp_size)
        par_dir = f'{home_dir}/sailor/sailor/Planner/baselines/Metis/profiles/{args.model}/{args.gpu_type}'
        Path(par_dir).mkdir(parents=True, exist_ok=True)
        for mbs, mbs_prof in profile.items():
            for tmp, prof in mbs_prof.items():
                print(mbs, tmp)
                with open(f'{par_dir}/mbs{mbs}_tmp{tmp}.json', 'w') as f: # note that it is mbs-tmp!
                    json.dump(prof, f, indent=2)
    elif args.planner == "Galvatron":
        profile = generate_profile_Galvatron(timing_info, mem_info, args.max_num_nodes,
                                             args.gpus_per_node, network_coeffs_inter_spec, network_coeffs_intra_spec, args.gpu_type)
        par_dir = f'{home_dir}/sailor/sailor/Planner/baselines/Galvatron/profiles/{args.model}/{args.gpu_type}'
        Path(par_dir).mkdir(parents=True, exist_ok=True)
        with open(f'{par_dir}/compute_profile.json', 'w') as f:
            json.dump(profile[0], f, indent=2)
        # replaced by the exact profiling used from Galvatron to avoid issues
        # with open(f'{par_dir}/memory_profile.json', 'w') as f:
        #     json.dump(profile[1], f, indent=2)
        with open(f'{par_dir}/hw_prof.json', 'w') as f:
            json.dump(profile[2], f, indent=2)
    else:
        raise NotImplementedError

    if args.planner in ["Varuna", "Oobleck", "Piper"]:
        par_dir = f'{home_dir}/sailor/sailor/Planner/baselines/{args.planner}/profiles/{args.model}/{args.gpu_type}'
        Path(par_dir).mkdir(parents=True, exist_ok=True)
        if args.planner == "Piper":
            prof_file = f'{par_dir}/profile.json'
        else:
            prof_file = f'{par_dir}/profile_{args.gpus_per_node}.json'
        with open(prof_file, 'w') as f:
            json.dump(profile, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate profile files for baselines',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path-sim', type=str, default="", help='Path to simulation file')
    parser.add_argument('--path-mem', type=str, default="", help='Path to file with memory info')
    parser.add_argument('--network-coeff-path', type=str, default="", help='Path to file with network coefficients')
    parser.add_argument('--planner', type=str, default="", help='Planner baseline to generate profile for')

    parser.add_argument('--model', type=str, default="", help='Model name')
    parser.add_argument('--optimizer', type=str, default="", help='Optimizer name')
    parser.add_argument('--fp_size', type=int, default=4, help='Floating point size')
    parser.add_argument('--gpu-type', type=str, default="", help='GPU type')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of model layers (transformer_layers + 2)')

    parser.add_argument('--max-num-nodes', type=int, default=128,
                        help='Maximum number of nodes (needed for Oobleck and Galvatron)')
    parser.add_argument('--gpus-per-node', type=int, default=1, help='Maximum number of nodes (needed for Galvatron)')
    parser.add_argument('--extra-mem-file', type=str, default="", help='File containing planner-specific memory profiles')

    args = parser.parse_args()
    main(args)
