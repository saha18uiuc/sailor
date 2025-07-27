
# pylint: disable-all
# File taken from: https://github.com/microsoft/varuna/blob/master/varuna/auto_config.py

import math
import os
import torch
import json
from os.path import expanduser

from sailor.profiling.profile_utils import estimate_send_time, estimate_ar_time


class AutoConfig:

    def __init__(self, num_gpus, gpus_per_vm, batch_size,
                 profile_file, optimizer, gpu_type, zone, gpu_memory_capacity=None, verbose=False,
                 autofill_missing_compute=True):

        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.gpus_per_vm = gpus_per_vm
        self.profile_file = profile_file
        self.optimizer = optimizer
        if gpu_memory_capacity is None:
            gpu_memory_capacity = torch.cuda.get_device_properties(0).total_memory
        self.gpu_memory_capacity = gpu_memory_capacity
        self.verbose = True

        print(f"NUM GPUS {self.num_gpus}, MEM CAPACITY {self.gpu_memory_capacity}")
        self.batch_times = dict()
        self.micro_batch = dict()

        self.read_profile(self.profile_file, autofill_missing_compute)

        # check that plan without TMP is possible
        # (if not, it means that the profile of one or more layers will be empty)
        # for _, layer_data in self.profile.items():
        #     for mbs, mbs_data in layer_data.items():
        #         print(mbs, mbs_data)
        #         if not mbs_data:
        #             return

        home_dir = os.environ.get('SAILOR_PATH')
        with open(f'{home_dir}/sailor/sailor/providers/gcp/multizone_bandwidths_het.json', 'r') as f:
            self.network_coeffs = json.load(f)

        self.network_coeffs = self.network_coeffs[zone][gpu_type][str(gpus_per_vm)][zone][gpu_type][str(gpus_per_vm)][0]

        self.num_stages_candidates = [i for i in range(1, self.num_pstages+1) if self.num_pstages % i == 0]
        print(f"NUM_STAGES_CANDIDATES IS {self.num_stages_candidates}, Pstages is {self.num_pstages}, num_gpus is {num_gpus}")

        self.layers_mem = []
        for _, data in self.profile.items():
            self.layers_mem.append(data[1]['params_bytes'])

    def generate_plan(self):
        for pp_size in self.num_stages_candidates:
            if self.num_gpus < pp_size:
                print(f"can't have {pp_size} stages!")
                continue
            # if verbose:
            #     print("Stages", pp_size)
            # get highest micro batch for each num_stage_cand
            mbs = self.get_microbatch_size(pp_size, 1, self.optimizer)
            if (mbs > 1) and (mbs % 2 != 0):
                mbs -= 1
            # print(f"Predicted microbatch size for {pp_size}: {mbs}")
            self.micro_batch[pp_size] = mbs
            dp_size = self.num_gpus // pp_size
            self.process_config(pp_size, dp_size, mbs)

    def process_config(self, pp_size, dp_size, mbs, partitions=None):

        cuts_per_stage = self.num_pstages // pp_size
        print(f"At process_config, cuts_per_stage is {cuts_per_stage}, partitions is {partitions}")
        num_microbatches = math.ceil((self.batch_size // dp_size) / mbs)

        self.calc_and_write_compute_times(pp_size, mbs, partitions)
        # TODO: comm profile for last cp in stage for each stage
        if pp_size > 1:
            comm_size = self.profile[cuts_per_stage][mbs]['output_activation']
        else:
            comm_size = 0

        send_time = 0
        long_send_time = estimate_send_time(comm_size, self.network_coeffs)
        long_send_time *= 1000000 # in usec

        # TODO
        max_tensor_size_alr = 0
        for i in range(pp_size):
            sum_tensors = sum(self.layers_mem[i*pp_size:(i+1)*pp_size])
            max_tensor_size_alr = max(max_tensor_size_alr, sum_tensors)

        alr = estimate_ar_time(max_tensor_size_alr, dp_size, self.network_coeffs)
        alr *= 1000000 # in usec
        batch_time = self.get_simulated_time(pp_size, num_microbatches, send_time,
                                             long_send_time, alr, self.verbose)
        self.batch_times[pp_size] = batch_time
        return batch_time


    def calc_and_write_compute_times(self, pp_size, mbs, partitions):
        #if partitions is None:
        pstages_per_stage = self.num_pstages // pp_size

        fwd_times = []
        bwd_times = []
        out = open("compute.txt", "w")
        for stage in range(pp_size):
            fwd_time = 0.0
            bwd_time = 0.0
            if partitions is None:
                pstages = range(pstages_per_stage*stage, pstages_per_stage*(stage+1))
            else:
                pstages = partitions[stage]
            for pstage in pstages:
                # print(pstage, self.profile[pstage])
                # print(mbs, self.profile[pstage][mbs])
                fwd_time += self.profile[pstage][mbs]["forward"]
                bwd_time += self.profile[pstage][mbs]["backward"]


            # TODO: COPY OR NO COPY !!!
            # # acts copy
            # if stage < (pp_size-1):
            #     copy = self.compute_profile[pstages_per_stage*(stage+1) - 1][mbs]["copy"]
            #     fwd_time += copy; bwd_time += copy

            # # grads copy
            # if stage > 0:
            #     copy = self.compute_profile[pstages_per_stage * stage][mbs]["copy"]
            #     fwd_time += copy; bwd_time += copy

            # Our change
            fwd_time *= 1000000 # in usec
            bwd_time *= 1000000 # in usec
            fwd_times.append(fwd_time)
            bwd_times.append(bwd_time)

            out.write(f"{fwd_time} {bwd_time}\n")

        out.close()

    def get_simulated_time(self, pp_size, num_microbatches, send_time,
                           long_send_time, alr, verbose=False):
        home_dir = os.environ.get('SAILOR_PATH')
        tools_dir = f"{home_dir}/sailor/sailor/Planner/baselines/Varuna"
        sim_binary = os.path.join(tools_dir, "simulator", "simulate-varuna.bin")
        # TODO: use C++ extension for python?
        command = f"GPUS_PER_VM={self.gpus_per_vm} {sim_binary} " + \
            f"{pp_size} {num_microbatches} {send_time} {alr} {long_send_time}"
        if verbose:
            print(command)
        #os.system(command)
        simulate = os.popen(command).read()
        batch_time = simulate.split("\n")[0]
        batch_time = float(batch_time.split(" ")[-1])
        batch_time = batch_time / 1000000
        return batch_time

    def get_min(self):
        min_time = 1000000000000
        best_pp = -1
        best_mbs = -1
        for pp_size in self.batch_times:
            if min_time > self.batch_times[pp_size]:
                best_pp = pp_size
                best_mbs = self.micro_batch[pp_size]
                min_time = self.batch_times[pp_size]
        return best_pp, best_mbs, min_time

    def get_sorted_list(self):
        plans = []
        for pp_size in self.batch_times:
            plan_time = self.batch_times[pp_size]
            if plan_time>0.0:
                best_mbs = self.micro_batch[pp_size]
                plans.append([pp_size, best_mbs, plan_time])

        plans = sorted(plans, key=lambda x: x[2])
        print(f"--------------------- BATCH TIMES: {plans}")
        return plans

    def read_profile(self, profile_file, autofill_missing_compute=False):

        self.profile = {}
        with open(profile_file, 'r') as f:
            self.profile_from_file = json.load(f)
        self.num_pstages = len(self.profile_from_file)

        for key, mbs_dict in self.profile_from_file.items():
            layer_dict = {}
            for mbs, data in mbs_dict.items():
                layer_dict[int(mbs)] = data
            self.profile[int(key)] = layer_dict

    def get_max_mem(self, mbs, tp_size, pp_size, pstages_per_stage, optimizer):

        print(f"At get_max_mem, mbs is {mbs}, pp_size is {pp_size}, pstages_per_stage is {pstages_per_stage}")

        # Note: we have replaced it with the SAILOR logic, as to profile the whole model
        # on a distributed setup is not always an option
        # We use only the torch allocated memory (i.e. no extra mem) since it is not
        # taken into account in the original implementation

        max_memory_used = 0
        for stage in range(pp_size):
            pstages = pstages_per_stage[stage]

            mem_usage = 0
            for pstage in pstages:
                mem_usage += (self.profile[pstage][mbs]["mem_required"] - \
                                self.profile[pstage][mbs]["output_activation"] )
                #print(pstage, self.profile[pstage][mbs]["mem_required"], self.profile[pstage][mbs]["output_activation"])
            last_cp = pstages[-1]
            mem_usage += self.profile[last_cp][mbs]["output_activation"]
            print(f"stage: {stage}, Mem_usage: {mem_usage}")
            max_memory_used = max(mem_usage, max_memory_used)

        # print(f"PP_SIZE {pp_size}, max_memory_used {max_memory_used}")
        return max_memory_used // tp_size


    def get_microbatch_size(self, tp, pp_size, optimizer, partitions=None):
        if partitions is None:
            pstages = self.num_pstages // pp_size
            pstages_per_stage = [range(pstages*i, pstages*(i+1)) for i in range(pp_size)]
        else:
            pstages_per_stage = partitions

        max_micro_bs = max([int(x) for x in self.profile[0].keys()])
        print(f"max_micro_bs is {max_micro_bs}")
        mbs = max_micro_bs
        limit = self.gpu_memory_capacity

        # small change over binary search - same complexity
        while mbs > 0:
            mem_usage = self.get_max_mem(mbs, tp, pp_size, pstages_per_stage, self.optimizer)
            if mem_usage <= limit:
                break
            mbs /= 2

        if mbs>0:
            return mbs
        return -1
