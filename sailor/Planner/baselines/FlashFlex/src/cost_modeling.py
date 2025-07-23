import numpy as np
import math

class MemoryCost:
    def __init__(self, device_memory, layers, stage_strategy, configs, flags ) -> None:
        """
            flags: [Embedding, Prenorm, Output]
        """
        self.stage_strategy = stage_strategy
        self.layers = layers
        self.device_memory = device_memory
        self.configs = configs
        self.flags = flags

        #print(f"----------- AT MEMORY COST: stage_strategy {stage_strategy}, layers {layers}, device_memory {device_memory}, configs {configs}, flags {flags}")

        self.tp_deg = len(self.stage_strategy)

    def param_memory(self):
        return ((self.layers * 12 * math.pow(self.configs.H, 2) / self.tp_deg )  \
                    + self.configs.H * self.flags[1] + self.configs.V * self.configs.H * self.flags[2] \
                    + self.configs.H * (self.configs.V + self.configs.S) * self.flags[0]) * self.configs.B_type / (1024 ** 3) * 4

    def activation_memory(self, recompute=False):
        """
            No activation recompute is applied in our project, need to store the full micro-batch activation (and it's gradients)
        """
        return (self.layers * self.configs.MB * self.configs.S * self.configs.H / self.tp_deg + \
                self.configs.MB * self.configs.S * self.configs.H * 2) * self.configs.B_type / 1024 ** 3

    def activation_memory_profiled(self):
        """
            This function use the profiled results from 4090 server, work for Llama2-70b
        """
        assert self.configs.N_attn_heads == 64

        profiled_mem = {8: 0.73652935, 4: 0.848956108, 2: 1.080645561, 1: 1.542071342}

        return self.layers * self.configs.MB * profiled_mem[self.tp_deg]

    def if_oom(self):

        return self.overall_memory() > self.device_memory / len(self.stage_strategy)

    def overall_memory(self, recompute=False):
        return self.param_memory() + self.activation_memory(recompute=recompute)

def gen_layer_dp_rank_group(layer_related_tp_groups):
    tp_sizes = [len(tp_group) for tp_group in layer_related_tp_groups]
    max_tp_size = max(tp_sizes)

    padded_tp_groups = []
    for i in range(len(layer_related_tp_groups)):
        group = layer_related_tp_groups[i]
        max_replicate = max_tp_size // tp_sizes[i]
        padded_tp_group = []
        for r in group:
            for _ in range(max_replicate):
                padded_tp_group.append(r)
        padded_tp_groups.append(padded_tp_group)

    related_dp_rank_groups = []
    for i in range(max_tp_size):
        related_dp_rank_group = []
        for group in padded_tp_groups:
            related_dp_rank_group.append(group[i])
        related_dp_rank_groups.append(related_dp_rank_group)

    return related_dp_rank_groups

class TimeCost:
    def __init__(self, all_pipelines, configs) -> None:
        # each pipeline: [strategy, layer_partition, ]
        self.all_pipelines = all_pipelines
        self.npipelines = len(all_pipelines)
        self.devices = configs.devices

        self.configs = configs
        self.tensor_cores, self.comm_bws, self.comm_bws_dict, _= configs.specs


    def gen_dp_devices_matrix(self):

        self.dp_devices_matrix = [[] for _ in range(self.configs.L)]
        self.layer_tp_size_matrix = [[] for _ in range(self.configs.L)]

        for layer_id in range(self.configs.L):
            layer_devices = []
            for pp_id in range(len(self.all_pipelines)):
                strategy, layer_partition, _ = self.all_pipelines[pp_id]
                layer_stage = [i for i in range(len(layer_partition)) for _ in range(layer_partition[i])]

                layer_devices.append(strategy[layer_stage[layer_id]])

                self.layer_tp_size_matrix[layer_id].append(len(strategy[layer_stage[layer_id]]))


            relative_tp_size = np.array([len(layer_devices[i]) for i in range(len(layer_devices))])
            relative_tp_size = max(relative_tp_size // min(relative_tp_size))
            layer_dp_devices = gen_layer_dp_rank_group(layer_devices)

            self.dp_devices_matrix[layer_id].extend(layer_dp_devices[:relative_tp_size])

        self.layer_max_tp_size = np.array(self.layer_tp_size_matrix).max(axis=1)
        self.layer_min_tp_size = np.array(self.layer_tp_size_matrix).min(axis=1)

    def gen_gpipe_map(self, pp_id):
        """
            for a pipeline, generate each column as shown in gpipe, in forward pass
        """

        nstages = len(self.all_pipelines[pp_id][0])
        n_mb = self.configs.N_MB

        gpipe_map = np.ones((nstages, n_mb + nstages - 1))
        for i in range(nstages):
            for j in range(nstages - i - 1):
                gpipe_map[i, j] = 0
            for j in range(i):
                gpipe_map[i, n_mb + nstages - 1 - j - 1] = 0

        return gpipe_map

    def dp_layer_comm_cost(self, dp_devices, layer_id):

        return 2 * max([sum([12 * math.pow(self.configs.H, 2) * self.configs.B_type / (self.layer_min_tp_size[layer_id] * len(dp_devices) * self.comm_bws_dict[i, j])
                    for j in dp_devices if j != i]) for i in dp_devices ])

    def mb_tp_layer_comm_cost(self, tp_devices, recompute=False):
        alpha = 12 if recompute else 8

        return alpha * max([sum([self.configs.MB * self.configs.S * self.configs.H * self.configs.B_type /
                              (len(tp_devices) * self.comm_bws_dict[i, j])
                    for j in tp_devices if j != i]) for i in tp_devices ])

    def mb_pp_comm_cost(self, tp_devices_pre, tp_devices_post):
        if tp_devices_post is None:
            return 0

        def get_broadcast_cost_list(d_from):
            broadcast_cost_list = [self.configs.MB * self.configs.S * self.configs.H
                                * self.configs.B_type / (len(tp_devices_post) *
                                self.comm_bws_dict[d_from, d_to])
                                for d_to in tp_devices_post if d_to != d_from]
            return broadcast_cost_list

        cost_list = [[self.configs.MB * self.configs.S * self.configs.H *
                                  self.configs.B_type /
                                  self.comm_bws_dict[d_pre, d_post] +
                                  sum(get_broadcast_cost_list(d_from=d_post))
                                  for d_post in tp_devices_post] for d_pre in tp_devices_pre]

        return 2 * min(np.array(cost_list).flatten())


    def mb_stage_cost(self, tp_devices, nlayers):

        return nlayers * (self.mb_tp_layer_comp_cost(tp_devices) + self.mb_tp_layer_comm_cost(tp_devices))

    def mb_tp_layer_comp_cost(self, tp_devices, recompute=False):
        alpha = 96 if recompute else 72
        return alpha * self.configs.MB * self.configs.S * math.pow(self.configs.H, 2) * \
                (1 + self.configs.S / (6 * self.configs.H) + self.configs.V / (16 * self.configs.L * self.configs.H)) / (self.tensor_cores[tp_devices[0]] * len(tp_devices))

    def mb_flops(self):
        return 72 * self.configs.MB * self.configs.S * math.pow(self.configs.H, 2) * \
                (1 + self.configs.S / (6 * self.configs.H))

    def glb_flops(self,):
        alpha = 72

        return alpha * self.configs.L * self.configs.GLB_B * self.configs.S * math.pow(self.configs.H, 2) * \
                (1 + self.configs.S / (6 * self.configs.H) + self.configs.V / (16 * self.configs.L * self.configs.H))

    def pipeline_cost(self, pp_id):
        """
            bubble cost is added here
        """
        gpipe_map = self.gen_gpipe_map(pp_id)
        strategy = self.all_pipelines[pp_id][0]
        layer_partition = self.all_pipelines[pp_id][1]
        # # ------ assume known now
        # layer_partition = [10, 8, 8, 6]
        # strategy = [[0, 1], [2], [3], [4]]
        # # ------ assume known now

        stage_costs = [self.mb_stage_cost(strategy[stage], layer_partition[stage]) for stage in range(len(strategy))]

        pp_comm_cost = [self.mb_pp_comm_cost(tp_devices_pre=strategy[j], tp_devices_post=strategy[j + 1])
                         if j != len(strategy) - 1
                        else self.mb_pp_comm_cost(tp_devices_pre=strategy[j], tp_devices_post=None)
                        for j in range(len(strategy))]

        return sum(np.max((np.array(pp_comm_cost) + np.array(stage_costs))[::-1].reshape(-1, 1) * gpipe_map, axis=0))

    def dp_cost(self):
        self.gen_dp_devices_matrix()

        dp_devices_matrix = self.dp_devices_matrix

        return sum([sum([self.dp_layer_comm_cost(dp_set, i) for dp_set in dp_devices_matrix[i]]) for i in range(len(dp_devices_matrix))])

    def overall_cost(self):
        """
            for all pipelines
        """

        alpha = 1
        pp_cost = []

        for i in range(self.npipelines):
            pp_cost.append(self.pipeline_cost(i))

        return max(pp_cost) + self.dp_cost()
        #return max(pp_cost) * 1.8 + self.dp_cost() * alpha * 1.1


    def token_throughput(self):
        # some pipeline cannot find a strategy that is not OOM
        if None in self.all_pipelines:
            return 0

        overall_cost = self.overall_cost()
        print(f"************ OVERALL COST IS {overall_cost}")
        return self.configs.T / overall_cost

    def mfu(self, actual_running_time=None):

        running_time = actual_running_time if actual_running_time is not None else self.overall_cost()

        return  self.glb_flops() / running_time / sum(self.tensor_cores)
