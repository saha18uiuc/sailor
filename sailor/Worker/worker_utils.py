import os
import random
import time
import numpy as np
from datetime import timedelta

import torch
import torch.distributed as dist


def debug_print(*args):
    x = " ".join(str(t) for t in args)
    print(x, flush=True)

# pylint: disable=too-many-locals


def assign(tensor, model, optimizer):

    model_state = model.state_dict()
    opt_state = optimizer.state_dict()

    mstate = {}

    idx = 0  # pointer into the allocated space
    for name, ref in model_state.items():
        if torch.is_tensor(ref):
            tsize = ref.numel()
            # print(name, idx, tsize)
            ar = tensor[idx:idx+tsize]
            idx += tsize
            mstate[name] = ar.reshape(ref.shape)
        else:  # scalar
            if isinstance(ref, int):
                mstate[name] = int(tensor[idx])
            else:
                mstate[name] = tensor[idx]
            idx += 1

    optstate = {}
    # reconstruct dimensions - optimizer
    for name, ref in opt_state['state'].items():
        d = {}
        # print(name, ref.keys())
        for n, r in ref.items():
            if torch.is_tensor(r):
                # print(name, n, r.numel())
                tsize = r.numel()
                ar = tensor[idx:idx+tsize]
                idx += tsize
                d[n] = ar.reshape(r.shape)
            else:
                if isinstance(ref, int):
                    d[n] = int(tensor[idx])
                else:
                    d[n] = tensor[idx]
                idx += 1

        optstate[name] = d
    optstatetotal = {'state': optstate, 'param_groups': optimizer.state_dict()[
        'param_groups']}
    model.load_state_dict(mstate)
    optimizer.load_state_dict(optstatetotal)


def init_optimizer(optimizer):

    opt_state = optimizer.state_dict()
    if len(opt_state['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
    optimizer.step()
    return optimizer


def seed_everything(seed=42):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup(master_ip, master_port, rank, world_size, gname):

    # os.environ['MASTER_ADDR'] = master_ip
    # os.environ['MASTER_PORT'] = master_port

    # initialize the process group

    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    # os.environ["ENABLE_NCCL_HEALTH_CHECK"] = "1"
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

    # the master is a non-stopping process in the controller

    tcp_start = time.time()
    store = dist.TCPStore(host_name=master_ip, port=int(
        master_port), is_master=False)
    dist.init_process_group("nccl", world_size=world_size, rank=rank,
                            store=store, group_name=gname, timeout=timedelta(seconds=10))
    debug_print(f"Time to join the TCPStore is {time.time()-tcp_start} sec")
    return store
