# pylint: disable-all
# flake8: noqa

# Script adapted from https://github.com/DachengLi1/AMP/blob/main/src/cost_homo.py
from collections import defaultdict

import sys
import os

import torch
import torch.nn as nn
import numpy as np

from sailor.Planner.baselines.AMP.amp_utils import axis2rank
from sailor.Planner.baselines.AMP.pipe import pipe_ds, pipe_cost

home_dir = os.environ['HOME']
curr_dir = os.getcwd()
workdir_path = os.path.join(
    home_dir, "AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")
example_path = os.path.join(workdir_path, "examples")
sys.path.append(workdir_path)
sys.path.append(example_path)

# TODO: pass as input
PROFILE_BY_AMP = {}


class AMP(nn.Module):
    '''
        AMP optimization class to obtain the global iteration time for each given model and cloud configuration
        Inputs:
        - model_config: dictionary with all the input configuration values.
        - model_type: type of model provided (gpt for now).
    '''

    def __init__(self, model_config, cluster_config, profile_dir, llm_info, placement=False):

        super().__init__()
        self.model_config = model_config

        self.model_type = model_config["type"]
        self.placement = placement
        self.profile_dir = profile_dir
        self.llm_info = llm_info[model_config['model']]
        assert self.model_type == "gpt2"

        PROFILE_BY_AMP["total_layers"] = int(self.model_config["num_layers"].item()) + 2
        PROFILE_BY_AMP["fix_input_layers"] = int(self.model_config["num_layers"].item())
        PROFILE_BY_AMP["extra_layers_in_profiling"] = 2
        PROFILE_BY_AMP["first_transformer_layer"] = 1
        PROFILE_BY_AMP["last_transformer_layer"] = int(self.model_config["num_layers"].item())

        self.cluster_config = cluster_config
        self.init_param()

    def init_param(self):
        h = float(self.model_config["hidden_size"].item())
        n = float(self.model_config["num_layers"].item())
        s = float(self.model_config["sequence_length"].item())
        v = float(self.model_config["vocab_size"].item())
        num_all_layers = int(self.model_config["num_all_layers"])

        self.profile_cost = {}
        self.profile_cost_per_gpu = {}
        for gpu_type, info in self.cluster_config.items():
            gpu_prof_cost = {}
            for mp_size in [1,2,4,8]:
                known_record = f"{self.profile_dir}/{gpu_type}/profile_{mp_size}.npy"
                if os.path.exists(known_record):
                    # We model backward time as forward time * 2
                    cur_profile_cost = 3 * np.load(f"{known_record}")
                    gpu_prof_cost[str(mp_size)] = cur_profile_cost
            if gpu_prof_cost:
                self.profile_cost_per_gpu[gpu_type] = gpu_prof_cost

        for mp_size in [1,2,4,8]:
            num_diff_gpus = 0
            total_cost = np.zeros(int(num_all_layers), dtype=float)
            for gpu_type in list(self.cluster_config.keys()):
                if ((gpu_type in self.profile_cost_per_gpu) and (str(mp_size) in self.profile_cost_per_gpu[gpu_type])):
                    num_diff_gpus += 1
                    total_cost += self.profile_cost_per_gpu[gpu_type][str(mp_size)]
            # AMP uses weighted average, but they dont explain how the weights are set, so we will just use average
            self.profile_cost[str(mp_size)] = total_cost/num_diff_gpus

        print(f"profile_cost_per_gpu is {self.profile_cost_per_gpu}")
        print(f"profile_cost is {self.profile_cost}")


    def forward(self, args, partition=None):
        '''
            Forward function to run the layer iteration cost for the input configuration.
        '''
        model_type = self.model_type
        config, bs, micro_bs, cluster_info, model_config, oth = args
        amp_config = {"profile_cost": self.profile_cost}
        if isinstance(bs, list):
            return predict_multi(config, bs, micro_bs, cluster_info, model_config, amp_config, oth, self.placement)
        else:
            assert isinstance(bs, int)
            return predict_single(config, bs, micro_bs, cluster_info, model_config, amp_config, oth, self.placement, self.llm_info, partition)

# pipeline communication cost, return shape: (L-1, pp-1)

def get_cost_c(cluster_info, model_config, parallel_config, amp_config, llm_info, dp_index=0):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]

    _layer = ["embed2h", "noop"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")

    _layer.extend(["noop"])
    _num_layer = len(_layer)

    # build layer activation lookup table. Noop exatly has the same activation as the previous op.
    # Leave bs factor outside.
    layer_volume = []
    last_volume = torch.zeros(1,)
    print(bs, s, h)
    for i in range(_num_layer):
        layer_type = _layer[i]

        if layer_type != "noop":
            mp_str = str(int(mp.item()))
            if mp_str not in llm_info:
                return None
            # Our change: get activation count directly, to work for all models
            last_volume = bs * llm_info[mp_str][str(i)]['act_output_floats']
            layer_volume.append(last_volume)
        else:
            layer_volume.append(last_volume)
        # if layer_type == "embed2h" or layer_type == "transformer_layer":
        #     last_volume = bs * s * h
        #     layer_volume.append(last_volume)
        # elif layer_type == "embed2v":
        #     last_volume = bs * s * v / mp
        #     layer_volume.append(last_volume)
        # elif layer_type == "noop":
        #     layer_volume.append(last_volume)
        # else:
        #     raise RuntimeError("Unknown layer type.")

    # Build communication cost between pipeline stages by looking up the cluster information
    cost_c = torch.zeros((int(dp.item()), _num_layer-1, int(pp.item()-1)))
    for i in range(int(dp.item())):
        for j in range(int(pp.item()-1)):
            # get the slowest mp gpu connection
            slowest_bandwidth = np.inf
            for k in range(int(mp.item())):
                rank_cur = axis2rank(
                    axis=(j, i, k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                rank_peer = axis2rank(
                    axis=(j+1, i, k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                node_peer = rank_node_map[int(rank_peer.item())]

                # It assumes network bandwidth is the same, regardless of size - leave as is
                if node_cur != node_peer:
                    cur_bandwidth = min(
                        cluster_info[node_cur][0], cluster_info[node_peer][0])
                else:
                    cur_bandwidth = cluster_info[node_cur][1]
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth
            for k in range(_num_layer-1):
                #print(f"Layer {k}, layer_volume is {layer_volume[k]}, slowest_bandwidth is {slowest_bandwidth}")
                cost_c[i][k][j] = layer_volume[k] / slowest_bandwidth
    cost_c = torch.mean(cost_c, dim=0)
    # print(f"using cost_c: {cost_c.size()}")
    return cost_c

# execution cost for one layer (gpt2 architecture), return shape (L,)


def get_cost_e(cluster_info, model_config, parallel_config, amp_config, placement):

    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]

    profile_cost = amp_config["profile_cost"]
    # print(f"PROFILE COST IS {profile_cost}")

    _layer = ["embed2h"]  # vocab and position embedding layers
    for i in range(int(n.item())):
        _layer.append("transformer_layer")

    _layer.extend(["noop"])
    if 'LLAMA' in model_config['model']:
        _layer.extend(["noop"])

    _num_layer = len(_layer)

    cost_e = np.zeros((int(dp.item()), _num_layer))
    for i in range(int(dp.item())):
        assert _num_layer == len(
            profile_cost[str(int(mp.item()))]), "predicted number of layers not equal to actual"

        # mp_avg is only used with placement ablation study. Ignore it in reproducing main results.
        # cost_e in the main result is equivalent to using profile_cost.
        mp_avg = 0
        if placement:
            for j in range(int(pp.item())):
                slowest = float("inf")
                for k in range(int(mp.item())):
                    rank_cur = axis2rank(
                        axis=(j, i, k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_cur = rank_node_map[int(rank_cur.item())]

                    rank_next = axis2rank(
                        axis=(j, i, (k+1) % (mp.item())), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_next = rank_node_map[int(rank_next.item())]

                    if node_cur == node_next:
                        connectivity = cluster_info[node_cur][1]
                    else:
                        connectivity = min(
                            cluster_info[node_cur][0], cluster_info[node_next][0])
                slowest = min(slowest, connectivity)

        for layer_id in range(_num_layer):
            # Only supports gpt architecture
            layer_type = _layer[layer_id]
            #print(f"layer id is {layer_id}, placement is {placement}, layer_type is {layer_type}")

            if placement:
                cur_layer = bs * profile_cost["1"][layer_id] / mp.item()
            else:
                cur_layer = bs * profile_cost[str(int(mp.item()))][layer_id]

            if layer_type == "embed2h":
                pass
            elif layer_type == "embed2v":
                cur_layer += (v * h / mp * mp_avg).item()
            elif layer_type == "transformer_layer":
                cur_layer += ((7*h**2/mp + 2*bs*s*h) * mp_avg).item()
            elif layer_type == "noop":
                pass
            else:
                raise RuntimeError("Unknown layer type.")

            cost_e[i][layer_id] = cur_layer

    cost_e = torch.from_numpy(np.stack(cost_e, axis=0))
    cost_e = torch.mean(cost_e, dim=0)
    return cost_e

# Retrieve the worst case ga cost and adds it to the total cost time


def dp_cost(config, cluster_info, model_config, parallel_config, amp_config, partition, llm_info):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]

    _layer = ["embed2h"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")

    _layer.extend(["noop"])
    if 'LLAMA' in model_config['model']:
        _layer.extend(["noop"])
    _num_layer = len(_layer)

    # First translate to deepspeed partition form
    ds_partition = [0]
    for i in range(len(partition)):
        ds_partition.append(ds_partition[-1]+partition[i])
    assert ds_partition[-1] == _num_layer
    assert len(ds_partition) == pp + 1

    # should be per-dp_group time
    max_dp = torch.zeros(1,)
    for i in range(int(pp.item())):
        for j in range(int(mp.item())):

            slowest = float("inf")
            for k in range(int(dp.item())):
                rank_cur = axis2rank(
                    axis=(i, k, j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]

                rank_next = axis2rank(
                    axis=(i, (k+1) % (dp.item()), j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_next = rank_node_map[int(rank_next.item())]

                if node_cur == node_next:
                    connectivity = cluster_info[node_cur][1]
                else:
                    connectivity = min(
                        cluster_info[node_cur][0], cluster_info[node_next][0])

            slowest = min(slowest, connectivity)
            dp_const = 2 * (dp-1) / (dp * slowest)
            dp_const = torch.tensor([dp_const])

            param_count = torch.zeros(1,)
            counted = False
            for layer_id in range(ds_partition[i], ds_partition[i+1]):
                layer_type = _layer[layer_id]
                mp_str = str(int(mp.item()))

                # Our change: get param count directly, to work for all models
                param_layer = llm_info[mp_str][str(layer_id)]['params_floats']

                # Old code starts
                # if layer_type == "embed2h" or layer_type == "embed2v":
                #     if not counted:
                #         counted = True
                #         param_layer += h * v / mp
                #     # print(f"embed size {h * v / mp}")
                # elif layer_type == "transformer_layer":
                #     param_layer += 12 * h ** 2 / mp
                # elif layer_type == "noop":
                #     pass
                # else:
                #     raise RuntimeError("Unknown layer type.")
                # Old code ends

                param_count += param_layer

            # print(f"dp: {dp_const} and param {param_count}")
            cur_dp = dp_const * param_count
            if cur_dp > max_dp:
                max_dp = cur_dp

    return ds_partition, max_dp


def predict_multi(config, bs, mbs, cluster_info, model_config, amp_config, oth, placement):
    costs = torch.zeros(len(config))
    rank_maps = []
    partitions = []
    for i in range(len(config)):
        rank_map, partition, cost = predict_single(
            config[i], bs[i], mbs[i], cluster_info, model_config, amp_config, oth[i], placement)
        costs[i] = cost
        rank_maps.append(rank_map)
        partitions.append(partition)

    return rank_maps, partitions, costs


def predict_single(config, bs, mbs, cluster_info, model_config, amp_config, oth, placement, llm_info, partition=None):
    '''
        Base function to obtain the total iteration time. It consists of the sum of:
        1. cost_e: execution cost forward pass per layer, adapted to type of layer. Obtained a vector of size (L,).
        2. cost_c: communication cost for each layer in each stage. It is obtained by computing the layer activation size divided by cluster bandwidth.
        3. pipe_ds: partition based on DeepSpeed (original AMP applies here the Dynamic Programming algorithm to split layers among stages).
                    In our case, this is the direct uniform partition applied by DeepSpeed.
        4. pipe_cost: total cost of gradient accumulation step, adding the previous `cost_e` and `cost_c` (forward and backward pass for each layer in each stage).
        5. dp_cost: time for all reduce step given the DP size, the number of parameters and the bandwidth in the worst case.
    '''

    L = model_config["num_layers"]
    cost = torch.zeros(1,)
    M, N = config.shape
    config = np.asarray(config)

    if np.all(config == -1):
        rank_map = defaultdict(list)
        rank_node_map = dict()

        m = oth["mp_deg"]
        n = oth["dp_deg"]
        pp = oth["pp_deg"]

        # infer a GPU rank map
        counter = 0
        for j in range(N):
            for k in range(M):
                # (inherit from AMP) TODO: bad code here, config counts from 1
                rank_map[j].append(counter)
                rank_node_map[counter] = j
                counter += 1

        # print(f"AMP estimate default to {rank_map}")

    else:
        config = torch.from_numpy(config)
        pp = torch.max(config).float()

        # infer rank_map: given node name, returns the global mapped rank(int) in (pp, dp, mp) order
        # rank_node_map: given rank, returns the node
        rank_map = defaultdict(list)
        rank_node_map = dict()

        if pp >= (L + 2):
            print(f"early return with pp={pp}, L={L}")
            return None, None, torch.tensor([float("inf")])

        m = oth["mp_deg"]
        n = oth["dp_deg"]
        assert pp == oth["pp_deg"]

        rank_counter = np.zeros(int(pp.item()))

        # infer a GPU rank map
        for j in range(N):
            for k in range(M):
                # (inherit from AMP) TODO: bad code here, config counts from 1
                cur_pp = int(config[k][j] - 1)
                rank_map[j].append(
                    int((rank_counter[cur_pp] + cur_pp * m * n).item()))
                rank_node_map[int(
                    (rank_counter[cur_pp] + cur_pp * m * n).item())] = j
                rank_counter[cur_pp] += 1

    # infer number of micro-batch size B
    B = bs / (n * mbs)

    parallel_config = {"mp": m, "dp": n, "pp": pp, "micro_bs": mbs,
                       "rank_map": rank_map, "rank_node_map": rank_node_map}

    cost_e = get_cost_e(cluster_info=cluster_info,
                        model_config=model_config, parallel_config=parallel_config, amp_config=amp_config, placement=placement)
    cost_c = get_cost_c(cluster_info=cluster_info,
                        model_config=model_config, parallel_config=parallel_config, amp_config=amp_config, llm_info=llm_info)

    if cost_c is None or cost_e is None:
        return None, None, None

    # The following line is the original AMP code (Dynamic Programming algo), which is replaced by the DeepSpeed uniform partition
    # partition, _ = pipe_ast(len(cost_e), np.asarray(cost_e), np.asarray(cost_c), int(pp.item()), int(B.item()))

    if partition is None:
        partition = pipe_ds(len(cost_e), int(pp.item()))

    cost = pipe_cost(L, cost_e, cost_c, pp, B, partition)


    # translate to ds form, add data parallelism cost
    ds_partition, dp_side_cost = dp_cost(config, cluster_info=cluster_info,
                                         model_config=model_config, parallel_config=parallel_config,
                                         amp_config=amp_config, partition=partition, llm_info=llm_info)

    cost += dp_side_cost
    return rank_map, ds_partition, cost
