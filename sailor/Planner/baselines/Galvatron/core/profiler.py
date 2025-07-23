import os
import time
import torch
import numpy as np
from sailor.Planner.baselines.Galvatron.utils import save_profiled_memory, print_peak_memory, save_profiled_time, array2str, str2array, read_json_config, write_json_config
import re

class GalvatronProfiler():
    def __init__(self, args, pp_deg, tp_deg, gbs, mbs, world_size):
        self.args = args
        self.layernum_arg_names = None
        self.mem_path = None
        self.time_path = None
        self.model_name = None
        self.pp_deg = pp_deg
        self.global_tp_deg = tp_deg
        self.global_train_batch_size = gbs
        self.global_checkpoint = 0
        self.mbs = mbs
        self.world_size = world_size
        print(f"Start profiler with pp_deg {pp_deg}, tp_deg {tp_deg}, gbs {gbs}, mbs {mbs}, world_size {world_size}")

    # =============== For Setting Galvatron Profiler ===============
    def set_profiler_dist(self, path=None, model_layer_configs=None, model_name=None, profile_ranks=None, start_iter=10, end_iter=20, rank=None):
        rank = torch.distributed.get_rank() if rank is None else rank
        if profile_ranks is None:
            profile_ranks = [0, self.world_size-1]
        self.set_model_layer_configs(model_layer_configs)
        self.set_path(path)
        self.set_memory_profiler(rank, profile_ranks)
        exit_ = self.args.exit_after_profiling if 'exit_after_profiling' in self.args else True
        self.set_model_name(model_name)


    def set_model_layer_configs(self, model_layer_configs):
        if model_layer_configs is None:
            return
        self.hiddensize_list = [config['hidden_size'] for config in model_layer_configs]
        self.layernum_list = [config['layer_num'] for config in model_layer_configs]
        self.seqlen_list = [config['seq_len'] for config in model_layer_configs]

    def set_path(self, path):
        self.path = path

    def set_model_name(self, name):
        self.model_name = name

    def memory_profiling_path(self):
        if self.mem_path is not None:
            return self.mem_path
        assert self.model_name is not None, 'Should specify the model name!'
        args = self.args
        #memory_config_path = 'configs/memory_profiling_%s_%s.json'%(args.mixed_precision, self.model_name)
        memory_config_path = "memory_profile.json"
        self.mem_path = os.path.join(self.path, memory_config_path)
        return self.mem_path


    # =============== For Runtime Memory Profiling ===============
    def set_memory_profiler(self, rank, profile_ranks=[], max_profile_iter=5):
        self.rank = rank
        self.profile_ranks = profile_ranks if len(profile_ranks) > 0 else [rank]
        self.mem_dict = {}
        self.max_profile_iter = max_profile_iter

    def profile_memory(self, iter, stage=""):
        args, rank, profile_ranks, mem_dict, max_profile_iter = \
            self.args, self.rank, self.profile_ranks, self.mem_dict, self.max_profile_iter
        print(f"INSIDE profile_memory, rank: {rank}, profile_ranks: {profile_ranks}, mem_dict: {mem_dict}, max_profile_iter: {max_profile_iter}")

        if rank in profile_ranks and iter <= max_profile_iter:
            print(f"DO PROFILING, GALVATRON STYLE")
            local_rank = args.local_rank if 'local_rank' in args else 0
            profile_type = args.profile_type if 'profile_type' in args else 'allocated'
            if stage == "Before Forward":
                torch.cuda.reset_peak_memory_stats(local_rank)
                _, cur_mem = print_peak_memory("\n"+stage, local_rank, profile_type)
                mem_dict['iter_%d_before_forward'%iter] = cur_mem
            elif stage == "After Forward":
                _, cur_mem = print_peak_memory(stage, local_rank, profile_type)
                mem_dict['iter_%d_after_forward'%iter] = cur_mem
            elif stage == "After Backward":
                max_mem, cur_mem = print_peak_memory(stage, local_rank, profile_type)
                mem_dict['iter_%d_after_backward'%iter] = cur_mem
                mem_dict['iter_%d_after_backward_max'%iter] = max_mem
            else:
                print_peak_memory(stage, local_rank, profile_type)

    def post_profile_memory(self, iter):
        args, rank, profile_ranks, mem_dict, max_profile_iter = \
            self.args, self.rank, self.profile_ranks, self.mem_dict, self.max_profile_iter
        print(f"inside post_profile_memory, iter is {iter}, max_profile_iter is {max_profile_iter}")
        if iter == max_profile_iter:
            if rank in profile_ranks:
                mem_dict['model_states'] = mem_dict['iter_4_after_backward']
                if 'pipeline_type' not in args or args.pipeline_type == "gpipe":
                    mem_dict['model_states_and_activation'] = mem_dict['iter_4_after_forward']
                    mem_dict['activation'] = mem_dict['iter_4_after_forward'] - mem_dict['iter_4_before_forward']
                mem_dict['model_states_and_peak_activation'] = mem_dict['iter_4_after_backward_max']
                mem_dict['peak_activation'] = mem_dict['iter_4_after_backward_max'] - mem_dict['iter_4_after_backward']
                time.sleep(0.2*rank)
                print('[Profiled memory for rank %d]:'%rank)
                for key, val in mem_dict.items():
                    print("\t%s: %.2f MB"%(key, val))
                assert self.layernum_list is not None
                memory_config_path = self.memory_profiling_path()
                save_profiled_memory(memory_config_path, self.pp_deg, self.global_tp_deg, self.world_size, self.layernum_list, \
                                    self.global_train_batch_size, rank, mem_dict['model_states'], mem_dict['activation'], mem_dict['peak_activation'], self.global_checkpoint)
            if 'save_profiled_memory' in args and args.save_profiled_memory:
                exit(0)


    # =============== For Processing Profiled Memory and Time ===============
    def process_profiled_data(self, profile_type):
        args = self.args
        layernum_lists = [[1], [2]]
        if profile_type == 'computation':
            time_config_path = self.time_profiling_path()
            config = read_json_config(time_config_path)
            key_base = self.key_format(layernum_lists[0], args.profile_batch_size)
            val_base = config[key_base]
            for idx, layernum in enumerate(layernum_lists[1:]):
                key = self.key_format(layernum, args.profile_batch_size)
                val = config[key]
                avg_time = val - val_base
                avg_time = avg_time / args.profile_batch_size / (args.layernum_max-args.layernum_min)
                write_key = 'layertype_%d'%idx
                config[write_key] = avg_time
            write_json_config(config, time_config_path)
            print('Already written processed computation time into env config file %s!\n'%(time_config_path))
        elif profile_type == 'memory':
            memory_config_path = self.memory_profiling_path()
            config = read_json_config(memory_config_path)
            print(config)

            bsz = self.mbs
            layernum_list_base = layernum_lists[0]
            layertype = len(layernum_list_base)
            layernum_lists = layernum_lists[1:]
            layernum_diff = 1 #args.layernum_max - args.layernum_min
            param_result_list, act_result_list, param_list = [dict() for _ in range(layertype)], [dict() for _ in range(layertype)], [-1]*layertype

            pp_deg, tp_deg = 1, 1
            while True:
                if pp_deg * tp_deg > self.world_size:
                    break
                strategy = '%d_%d_%d'%(pp_deg,tp_deg,1)
                if strategy not in config:
                    print(f"STRATEGY: {strategy}")
                    tp_deg *= 2
                    continue
                re = config[strategy]
                for l in range(layertype):
                    layernum_key_0, layernum_key_1 = layernum_list_base, layernum_lists[l]
                    param_per_layer = (re[self.key_format(layernum_key_1, bsz, 0, 'ms')] - re[self.key_format(layernum_key_0, bsz, 0, 'ms')])/layernum_diff*pp_deg/4
                    act_per_layer_per_sample = (re[self.key_format(layernum_key_1, bsz, 0, 'act')] - re[self.key_format(layernum_key_0, bsz, 0, 'act')])/layernum_diff*pp_deg/(pp_deg*tp_deg)
                    act_per_layer_per_sample *= self.world_size / bsz
                    param_result, act_result, param = param_result_list[l], act_result_list[l], param_list[l]
                    param = max(param, param_per_layer*tp_deg)
                    print(param_per_layer, act_per_layer_per_sample, param)
                    param_result[tp_deg] = param_per_layer
                    act_result[tp_deg] = act_per_layer_per_sample
                    param_result_list[l], act_result_list[l], param_list[l] = param_result, act_result, param
                tp_deg *= 2

            for l in range(layertype):
                print('[layertype %d:]'%l)
                param_result, act_result, param = param_result_list[l], act_result_list[l], param_list[l]
                print('param:', param)
                # print('param_dict:', param_result)
                print('act_dict:', act_result)

            act_dict_c_list, act_cpt_list = [dict() for _ in range(layertype)], [-1]*layertype
            pp_deg, tp_deg = 1, 1
            while True:
                if pp_deg * tp_deg > self.world_size:
                    break
                print(pp_deg, tp_deg)
                strategy = '%d_%d_%d_c'%(pp_deg,tp_deg,1)
                if strategy not in config:
                    tp_deg *= 2
                    continue
                re = config[strategy]
                for l in range(layertype):
                    layernum_key_0, layernum_key_1 = layernum_list_base, layernum_lists[l]
                    act_per_layer_per_sample = (re[self.key_format(layernum_key_1, bsz, 0, 'act')] - re[self.key_format(layernum_key_0, bsz, 0, 'act')])/layernum_diff*pp_deg/(pp_deg*tp_deg)
                    act_per_layer_per_sample *= self.world_size / bsz
                    print(act_per_layer_per_sample)
                    act_dict_c, act_cpt = act_dict_c_list[l], act_cpt_list[l]
                    act_cpt = max(act_cpt, act_per_layer_per_sample)
                    act_dict_c[tp_deg] = act_per_layer_per_sample
                    act_dict_c_list[l], act_cpt_list[l] = act_dict_c, act_cpt
                tp_deg *= 2

            for l in range(layertype):
                print('[layertype %d:]'%l)
                act_dict_c, act_cpt = act_dict_c_list[l], act_cpt_list[l]
                print('act_dict_c:', act_dict_c)
                print('act_cpt:', act_cpt)

            inf=1e6
            other_memory_pp_off, other_memory_pp_on_first, other_memory_pp_on_last = \
                {'model_states': inf, 'activation': inf}, {'model_states': inf, 'activation': inf}, {'model_states': inf, 'activation': inf}
            pp_deg = 1
            while True:
                if pp_deg > self.world_size:
                    break
                tp_deg = 1
                while True:
                    if pp_deg * tp_deg > self.world_size:
                        break
                    print(pp_deg, tp_deg)
                    strategy = '%d_%d_%d'%(pp_deg,tp_deg,1)

                    world_size = pp_deg * tp_deg
                    if strategy not in config:
                        tp_deg *= 2
                        continue
                    re = config[strategy]
                    if pp_deg == 1:
                        layernum_list = layernum_list_base
                        layernum = layernum_list_base[0]
                    else:
                        layernum = pp_deg
                        layernum_list = [layernum] * layertype
                    ms_cost, act_cost = [], []
                    print(param_result_list, layertype)
                    for l in range(layertype):
                        ms_cost.append(param_result_list[l][tp_deg]*4)
                        act_cost.append(act_result_list[l][tp_deg])
                    layer_ms_costs_first = self.total_memcost(pp_deg, layernum, layertype, ms_cost, 0)
                    layer_ms_costs_last = self.total_memcost(pp_deg, layernum, layertype, ms_cost, pp_deg-1)
                    layer_act_costs_first = self.total_memcost(pp_deg, layernum, layertype, act_cost, 0)
                    layer_act_costs_last = self.total_memcost(pp_deg, layernum, layertype, act_cost, pp_deg-1)
                    other_ms_first = re[self.key_format(layernum_list, bsz, 0, 'ms')] - layer_ms_costs_first
                    other_ms_last = re[self.key_format(layernum_list, bsz, world_size-1, 'ms')] - layer_ms_costs_last
                    act_peak_first = max(re[self.key_format(layernum_list, bsz, 0, 'act_peak')], re[self.key_format(layernum_list, bsz, 0, 'act')])
                    act_peak_last = max(re[self.key_format(layernum_list, bsz, world_size-1, 'act_peak')], re[self.key_format(layernum_list, bsz, world_size-1, 'act')])
                    other_act_first = act_peak_first * world_size / bsz  - layer_act_costs_first * (pp_deg*tp_deg)
                    other_act_last = act_peak_last * world_size / bsz - layer_act_costs_last * (pp_deg*tp_deg)
                    print(other_ms_first, other_act_first, other_ms_last, other_act_last)
                    other_ms_first = other_ms_first if other_ms_first > 0 else 0
                    other_ms_last = other_ms_last if other_ms_last > 0 else 0
                    other_act_first = other_act_first if other_act_first > 0 else 0
                    other_act_last = other_act_last if other_act_last > 0 else 0
                    if pp_deg == 1:
                        other_memory_pp_off['model_states'] = min(other_memory_pp_off['model_states'], other_ms_first)
                        other_memory_pp_off['activation'] = min(other_memory_pp_off['activation'], other_act_first)
                    else:
                        other_memory_pp_on_first['model_states'] = min(other_memory_pp_on_first['model_states'], other_ms_first)
                        other_memory_pp_on_first['activation'] = min(other_memory_pp_on_first['activation'], other_act_first / pp_deg)
                        other_memory_pp_on_last['model_states'] = min(other_memory_pp_on_last['model_states'], other_ms_last)
                        other_memory_pp_on_last['activation'] = min(other_memory_pp_on_last['activation'], other_act_last / pp_deg)
                    tp_deg *= 2
                pp_deg *=2

            # other_memory_pp_on_first['activation'] = other_memory_pp_on_last['activation'] = max(other_memory_pp_on_first['activation'], other_memory_pp_on_last['activation'])
            print('other_memory_pp_off:', other_memory_pp_off)
            print('other_memory_pp_on_first:', other_memory_pp_on_first)
            print('other_memory_pp_on_last:', other_memory_pp_on_last)

            for l in range(layertype):
                if 'layertype_%d'%l not in config.keys():
                    config['layertype_%d'%l] = dict()
                config['layertype_%d'%l]['parameter_size'] = param_list[l]
                config['layertype_%d'%l]['tp_activation_per_bsz_dict'] = act_result_list[l]
            config['other_memory_pp_off'] = other_memory_pp_off
            config['other_memory_pp_on_first'] = other_memory_pp_on_first
            config['other_memory_pp_on_last'] = other_memory_pp_on_last
            write_json_config(config, memory_config_path)


    # =============== Util functions ===============
    def key_format(self, layernum, bsz, rank=None, type=None):
        if isinstance(layernum, list):
            s =  "layernum[%s]_bsz%d"%(array2str(layernum), bsz)
        else:
            s =  "layernum%d_bsz%d"%(layernum, bsz)
        if rank is not None and type is not None:
            s += '_rank%d_%s'%(rank, type)
        return s

    def match_key_str(self, s):
        if '[' in s and ']' in s:
            layernum = str2array(s[s.find('[')+1:s.find(']')])
            s = s[s.find(']')+2:]
            if 'rank' in s:
                pattern = r'bsz(\d+)_rank(\d+)_(\w+)'
                match = re.match(pattern, s)
                bsz = int(match.group(1))
                rank = int(match.group(2))
                type = match.group(3)
                results = [layernum, bsz, rank, type]
            else:
                pattern = r'bsz(\d+)'
                match = re.match(pattern, s)
                bsz = int(match.group(1))
                results = [layernum, bsz]
        else:
            if 'rank' in s:
                pattern = r'layernum(\d+)_bsz(\d+)_rank(\d+)_(\w+)'
                match = re.match(pattern, s)
                layernum = int(match.group(1))
                bsz = int(match.group(2))
                rank = int(match.group(3))
                type = match.group(4)
                results = [layernum, bsz, rank, type]
            else:
                pattern = r'layernum(\d+)_bsz(\d+)'
                match = re.match(pattern, s)
                layernum = int(match.group(1))
                bsz = int(match.group(2))
                results = [layernum, bsz]
        return results

    def total_memcost(self  , pp_deg, layernum, layertype, per_layer_cost, stage_idx):
        layer_costs = []
        for l in range(layertype):
            layer_costs += [per_layer_cost[l]] * layernum
        total_layer_num = layertype * layernum
        avg_layer_num = int(total_layer_num // pp_deg)
        last_layer_num = total_layer_num - avg_layer_num * (pp_deg-1)
        pp_divide = [avg_layer_num] * (pp_deg-1) + [last_layer_num]
        return np.sum(layer_costs[int(np.sum(pp_divide[:stage_idx])):int(np.sum(pp_divide[:stage_idx+1]))])
