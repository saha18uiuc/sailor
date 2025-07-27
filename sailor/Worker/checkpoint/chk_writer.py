"""Process for asynchronous checkpointing"""

import torch
import os
import copy
import time
from collections import OrderedDict
import signal
import sys
from ctypes import *
import numpy as np

from sailor.Worker.worker_utils import debug_print

class ChkWriter:

    def __init__(self, save_dir, pp_rank, tp_rank, pccheck_threads):

        self.cpu_buffer = None
        self.total_size = 0
        self.model = None
        self.opt = None
        self.save_dir = save_dir
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank

        self.ckp_path = f"{self.save_dir}/checkpoint_PP{self.pp_rank}_TP{self.tp_rank}.pt"
        self.pccheck_writer = None
        self.pccheck_threads = pccheck_threads

    def init_buffer(self):
        # TODO: fix for fp16
        lib_path = "/root/sailor/sailor/Worker/checkpoint/libtest_pccheck.so"
        self.lib = cdll.LoadLibrary(lib_path.encode())
        self.pccheck_writer = self.lib.writer(self.ckp_path.encode())

        self.lib.savenvm_new.argtypes = [
            c_void_p,
            c_void_p,
            c_size_t,
            c_int,
            c_int
        ]

        sz = 0
        for k, v in self.model.items():
            if torch.is_tensor(v):
                sz += torch.numel(v)
        print(f"model_size is {sz}")

        # TODO: fix params
        for it, val in self.optimizer['state'].items():
            for k, v in val.items():
                if torch.is_tensor(v):
                    sz += torch.numel(v)

        self.total_size = sz
        debug_print(f"Checkpoint size is {sz} floats")
        self.cpu_buffer=torch.empty(sz, dtype=torch.float32, pin_memory=True, device="cpu")

    def save_async(
        self,
        model,
        optimizer,
        res_state,
        lock,
        initial_set,
        cp_in_progress,
        start,
        barrier
    ):
        print(f"************ Async proc started")

        # wait to be initialized
        while True:
            with lock:
                if initial_set.value==1:
                    break

        print(f"************ About to initialize buffers")
        self.model = model
        self.optimizer = optimizer
        self.res_state = res_state

        self.init_buffer()
        barrier.wait()

        print(f"************ About to enter checkp-train loop")
        # checkp-train loop
        while True:
            with lock:
                if start.value == 0:
                    continue

            # 1. snapshot
            self.copy_to_cpu()

            with lock:
                cp_in_progress.value = 0

            # 2. persist
            save_path = f"{self.save_dir}/checkpoint_PP{self.pp_rank}_TP{self.tp_rank}_STEP{self.res_state['global_steps']}.pt"
            self.persist(save_path) # TODO: fix path

            with lock:
                start.value = 0

    def copy_to_cpu(self):
        start_idx = 0
        #print(f"FROM CHECKPOINT - optimizer is {self.optimizer['state'][193]['exp_avg']}")

        torch.cuda.synchronize()
        gpu_start = time.time()

        for k, v in self.model.items():
            if torch.is_tensor(v):
                v_numel = torch.numel(v)
                self.cpu_buffer[start_idx:start_idx+v_numel].copy_(torch.flatten(v))
                start_idx += v_numel

        for it, val in self.optimizer['state'].items():
            for k, v in val.items():
                if torch.is_tensor(v):
                    v_numel = torch.numel(v)
                    self.cpu_buffer[start_idx:start_idx+v_numel].copy_(torch.flatten(v))
                    start_idx += v_numel

        torch.cuda.synchronize()
        gpu_end = time.time()
        print(f"Copying took {gpu_end-gpu_start}")

    def persist(self, path):
        persist_start = time.time()

        # write model + optimizer
        piter = self.res_state['global_steps'] % 2
        cpu_arr = np.ctypeslib.as_array(self.cpu_buffer, shape=(self.total_size,))
        cpu_arr_ct = np.ctypeslib.as_ctypes(cpu_arr)
        self.lib.savenvm_new(self.pccheck_writer, cpu_arr_ct, self.total_size, self.pccheck_threads, piter)
        persist_end = time.time()

        # write metadata
        checkpoint = {
            'res_state': self.res_state
        }
        torch.save(checkpoint, path)
        print(f"***************** Checkpoint persisted to {path}, Persist took {persist_end-persist_start} sec")
