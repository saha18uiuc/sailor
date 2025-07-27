"""Process for asynchronous checkpointing"""

import torch
import os
import copy
import time
from collections import OrderedDict
import signal
import sys
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock, Barrier


from sailor.Worker.worker_utils import debug_print
from sailor.Worker.checkpoint.chk_writer import ChkWriter

class Chk_manager:

    def __init__(self, save_dir, pp_rank, tp_rank):

        manager = Manager()
        self.res_state = manager.dict()
        self.lock = Lock()
        self.cp_in_progress = Value("i", 0)
        self.start = Value("i", 0)
        self.set = Value("i", 0)
        self.initialized = False
        self.barrier = Barrier(2)
        self.save_dir = save_dir
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank

    def init_buffer_and_writer(self, model, optimizer):

        # self.model_state.update(model.state_dict())
        # self.opt_state.update(optimizer.state_dict())

        self.chk_writer = ChkWriter(self.save_dir, self.pp_rank, self.tp_rank)
        self.chk_process = Process(
            target=self.chk_writer.save_async,
            args=[model, optimizer, self.res_state, self.lock, self.set, self.cp_in_progress, self.start, self.barrier]
        )
        self.chk_process.start()

        with self.lock:
            self.set.value = 1

        self.barrier.wait()
        self.initialized = True

    def gpu_copy_in_progress(self):

        # return True if at least one of the background processes is copying
        with self.lock:
            if self.cp_in_progress.value == 1:
                return True

        return False

    def checkpoint_in_progress(self):
        with self.lock:
            if self.start.value == 1:
                return True

    def save_checkpoint(self, path, res_dict):
        while True:
            with self.lock:
                if self.start.value == 0:
                    break

        # additional state here
        self.res_state.update(res_dict)

        with self.lock:
            self.cp_in_progress.value = 1
            self.start.value = 1
