import torch
import signal
from deepspeed.utils import logger
import os
import gc
import sys
import torch.distributed as dist
import time
import io
from sailor.examples.deepspeed.train_utils import log_metrics_memory, timing_stack, TIMEOUT_NCCL

MULTIPLIER_RESTORING_SNAPSHOT = 5
WORKER_AGENT_PORT = "50051"
NUM_DATA_LOADERS = 2


def handle_cluster_change(chk, model_engine, training_args):
    """
    Function from worker agent that waits for the checkpoint recovery and cleans up the state.
    Steps:
    1. Wait for adjusting snapshot after reconfiguration in checkpoint process
    2. Retrieve blob checkpointing from child process
    3. Cleanup state
    """
    logger.info("A change in cluster occured!")
    while chk.exception_main_process.value == 0 and chk.exception_child_process.value == 0:
        logger.info("Wait for checkpoiting process to get the exception")
        time.sleep(1)
        continue
    wait = 0
    while chk.snapshot_recovered.empty() and wait < TIMEOUT_NCCL*MULTIPLIER_RESTORING_SNAPSHOT:
        logger.info(
            "Wait for checkpointing process to restore snapshot and put on queue")
        time.sleep(1)
        wait += 1
        continue
    if chk.snapshot_recovered.empty():
        return None
    logger.info(f"Queue is not empty {chk.snapshot_recovered}")
    # timing_stack(training_args.rank, os.getpid(), "snapshot_main_received")
    snapshot_recovered = chk.snapshot_recovered.get()
    my_snapshot = recover_snapshot_from_ckpt(snapshot_recovered)
    # timing_stack(training_args.rank, os.getpid(), "snapshot_recovered")
    # cleanup state
    cleanup(chk, model_engine, training_args)
    return my_snapshot


def recover_snapshot_from_ckpt(snapshot_recovered):
    """
    Recover snapshot from checkpoint process after it has been splitted.
    """
    log_metrics_memory()
    # Recovering snapshot:
    with io.BytesIO(snapshot_recovered.numpy()) as f:
        my_snapshot = torch.load(f)

    logger.info(
        f"""Snapshot recovered with keys: {my_snapshot.keys()} and iter {my_snapshot['iter']}
        and num tensors {len([t for _,t in my_snapshot['model'].items()])} are:
        layers {list(my_snapshot['model'].keys())}""")
    log_metrics_memory()
    return my_snapshot


def cleanup(chk, model_engine, training_args):
    """
    Cleanup function to kill distributed NCCL process, checkpoint process and free GPU memory.
    """
    log_metrics_memory()
    if chk is not None:
        if chk.chk_process is not None:
            os.kill(chk.chk_process.pid, signal.SIGTERM)
            chk.chk_process.join()
            logger.info("Killed ckpt")
    if chk is not None:
        logger.info("chk is still not none and clean it")
        del chk
    # timing_stack(training_args.rank, os.getpid(), "ckpt_cleaned")
    if dist.is_initialized():  # dist equivalent to ds_comm
        dist.destroy_process_group()
        logger.info("Killed NCCL")
    else:
        logger.info("NCCL is not initialized")
    # timing_stack(training_args.rank, os.getpid(), "NCCL_cleaned")
    log_metrics_memory()
    free_gpu_memory(
        gpu_memory_snapshot=torch.cuda.memory._snapshot(), model_engine=model_engine)
    timing_stack(training_args.rank, os.getpid(), "gpu_memory_cleaned")
    if model_engine is not None:
        if model_engine.use_cocktail_sgd:
            raise NotImplementedError
    # for GPU memory
    if model_engine is not None:
        del model_engine.optimizer
        del model_engine.module
        del model_engine
    log_metrics_memory()

    logger.info("Cleaned!")


# pylint: disable=unused-argument
def free_gpu_memory(gpu_memory_snapshot, model_engine):
    '''
    Free GPU memory that frees only pipe buffers from model engine that DeepSpeed does not free upon failure during train batch.
    Always empty GPU memory cache.
    '''
    try:
        del model_engine.pipe_buffers
        logger.info("deleted pipe buffers")
        torch.cuda.empty_cache()
    except Exception:
        logger.info("NOT deleted pipe buffers")


# pylint: disable=unused-argument
def worker_exit(training_process_id, worker_agent_process_id, model_engine, chk):
    """
    Function to kill all processes in case of exception and terminate training.
    - Kill checkpoint process: gloo distributed process and child process.
    - Kill NCCL process for global training.
    - Kill cocktail sgd process if applies. #TODO: not correct.
    - Delete parallel data workers in data loader
    - Terminate worker agent process.
    """
    logger.info("In exception I am going to kill all processes")
    log_metrics_memory()
    if chk is not None:
        if chk.chk_process is not None:
            os.kill(chk.chk_process.pid, signal.SIGTERM)
            chk.chk_process.join()
            logger.info("Killed ckpt")

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Killed NCCL")

    if model_engine is not None:
        if model_engine.use_cocktail_sgd:
            logger.info("Killed cocktail if applies")
            raise NotImplementedError
            # TODO: send kill message and SIGTERM?
            # model_engine.cocktail_sgd_ddp.dp_group.join()
            # #ERROR: 'torch._C._distributed_c10d.ProcessGroup' object has no attribute 'join'
        logger.info("garbage collector")
        gc.collect()
        logger.info("Cleaned cuda cache")
        torch.cuda.empty_cache()
    log_metrics_memory()
    if worker_agent_process_id is not None:
        os.kill(worker_agent_process_id, signal.SIGTERM)
        logger.info("Killed worker agent and training process")
    sys.exit()
