"""Process for asynchronous checkpointing"""

import torch
import os
import copy
import time
from collections import OrderedDict
import signal
import sys

from sailor.Worker.worker_utils import debug_print


def to_cpu(ref, snapshot=None):

    if snapshot is None:
        snapshot = {}

    if hasattr(ref, 'cpu'):
        snapshot = ref.cpu()

    elif isinstance(ref, dict):
        snapshot = {}
        for k, v in ref.items():
            snapshot[k] = None
            snapshot[k] = to_cpu(v, snapshot[k])

    elif isinstance(ref, list):
        snapshot = [None for _ in range(len(ref))]
        for idx, v in enumerate(ref):
            snapshot[idx] = to_cpu(v, snapshot[idx])

    else:
        return ref

    return snapshot


class Chk_manager:

    def __init__(self, model_dict, opt_dict):

        self.gpu_last = None
        self.cpu_last = None
        self.path_storage_last = None
        self.snapshot = None

        # handles to the parent's copies of the model and optimizer
        # these are cuda tensors
        self.pmodel = model_dict
        self.popt = opt_dict

        # we save on DRAM, to save space

        # keep two copies of the snapshot - write one after the other
        # ('mode' determines which copy to write)
        # when loading, load from the latest *completed* snapshot

        # each snapshot has:
        # - params + meta dictionary
        # - completed bool
        # - timestamp (latest timestamp it was modified)

        self.cpu_copy_one = {}
        self.cpu_copy_one['model'] = {}
        self.cpu_copy_one['opt'] = {}
        self.cpu_copy_one['opt']['param_groups'] = {}
        self.cpu_copy_one['opt']['state'] = {}
        self.cpu_copy_one['meta'] = {}
        self.cpu_copy_one['timestamp'] = None
        self.cpu_copy_one['completed'] = False

        self.cpu_copy_two = {}
        self.cpu_copy_two['model'] = {}
        self.cpu_copy_two['opt'] = {}
        self.cpu_copy_two['opt']['param_groups'] = {}
        self.cpu_copy_two['opt']['state'] = {}
        self.cpu_copy_two['meta'] = {}
        self.cpu_copy_two['timestamp'] = None
        self.cpu_copy_two['completed'] = False

        self.mode = 0
        self.snap_list = [self.cpu_copy_one, self.cpu_copy_two]

        self.init_snaps(self.cpu_copy_two)
        self.init_snaps(self.cpu_copy_one)

    def init_snaps(self, array):

        cpu_model_copy = array['model']
        cpu_opt_copy = array['opt']

        sz = 0

        for k, v in self.pmodel.items():
            cpu_model_copy[k] = torch.empty(
                v.shape, dtype=torch.float32, pin_memory=True)
            sz += torch.numel(v)*4

        for it, val in self.popt['state'].items():
            cpu_opt_copy['state'][it] = {}
            for k, v in val.items():
                cpu_opt_copy['state'][it][k] = torch.empty(
                    v.shape, dtype=torch.float32, pin_memory=True)
                sz += torch.numel(v)*4

        debug_print(f"Checkpoint size is {sz} bytes")

    def copy_to_cpu(self, meta):

        if self.mode == 0:
            array = self.snap_list[0]

        else:
            array = self.snap_list[1]

        array['completed'] = False

        for k, v in self.pmodel.items():
            array['model'][k].copy_(v)

        array['opt']['param_groups'] = copy.deepcopy(self.popt['param_groups'])

        for it, val in self.popt['state'].items():
            for k, v in val.items():
                array['opt']['state'][it][k].copy_(v)

        array['meta'] = copy.deepcopy(meta)

        # fix timestamp, and complete
        array['timestamp'] = time.time()
        array['completed'] = True

    def save(self, sdict, at_gpu=False, at_cpu=False):

        if at_gpu:
            self.save_gpu(sdict)
        elif at_cpu:
            self.save_cpu_fast(sdict)
        else:
            # TODO(#21)
            raise NotImplementedError
            # self.save_storage(sdict)

    def check_load(self):
        t0 = self.snap_list[0]['timestamp']
        c0 = self.snap_list[0]['completed']

        t1 = self.snap_list[1]['timestamp']
        c1 = self.snap_list[1]['completed']

        debug_print(f"T0: {t0}, C0: {c0}")
        debug_print(f"T1: {t1}, C1: {c1}")

        if t0 > t1:
            if c0:
                return self.snap_list[0]
            if c1:
                return self.snap_list[1]
        else:
            if c1:
                return self.snap_list[1]
            if c0:
                return self.snap_list[0]

        debug_print("None of the snapshots are completed! Ignore for now!")
        return None

    def load_from_cpu(self, source):

        if source is None:
            return {}

        debug_print(self)

        # ones = torch.ones(10, dtype = torch.float32)
        # self.cpu_model_copy['module.fc.bias'].copy_(ones)
        # print("---- load: ", self.cpu_model_copy['module.fc.bias'])

        ret_state = {'model': source['model'], 'optimizer': source['opt'],
                     'epoch': source['meta']['epoch'], 'iter': source['meta']['iter']}
        return ret_state

    def load(self, from_gpu=False, from_cpu=False):

        if from_gpu:
            return self.gpu_last
        if from_cpu:
            source = self.check_load()
            return self.load_from_cpu(source)  # pass source here
        checkpoint = torch.load(self.path_storage_last)
        return checkpoint

    def cpu_snap(self, sdict):

        snap = {}
        for name, ref in sdict.items():
            snap[name] = to_cpu(ref)

        return snap

    def save_gpu(self, sdict):

        new_snap = OrderedDict()
        for name, ref in sdict.items():
            if name not in new_snap:
                new_snap[name] = copy.deepcopy(ref)

        self.gpu_last = new_snap

    def save_cpu(self, sdict):

        self.cpu_last = self.cpu_snap(sdict)
        print(self.cpu_last)

    def save_storage(self, sdict, fpath):

        self.snapshot = self.cpu_snap(sdict)
        torch.save(self.snapshot, fpath)
        with open(fpath, 'a+') as f:
            os.fsync(f.fileno())
            f.close()

        self.path_storage_last = fpath

    def get_mode(self):
        return self.mode

    def set_mode(self, m):
        self.mode = m


def save_cpu_async(chk, do_snap, in_progress, sdict):

    # No need for locks here since Values are process-safe:
    # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Value
    debug_print("Asynchronous checkpoint function is created")
    debug_print(chk)

    def terminate(signum, _):
        # with plock:
        do_snap.value = 0
        in_progress.value = 0
        debug_print(f"[Checkpoint Manager:] Got signal {signum} to EXIT")
        sys.exit()

    signal.signal(signal.SIGTERM, terminate)

    while True:

        if do_snap.value == 0:
            continue

        # print(f"---------- Start Asynchronous snapshot ----------")

        in_progress.value = 1
        # check here where to save
        mode = chk.get_mode()
        chk.copy_to_cpu(sdict)
        debug_print(f"Checkpoint at {mode}")

        chk.set_mode(1-mode)

        in_progress.value = 0
        do_snap.value = 0
