import grpc
from concurrent import futures
import argparse
import signal
from deepspeed.utils import logger
import traceback
import sailor
import os
from torch import multiprocessing
import threading

from sailor.protos import orchestration_pb2, orchestration_pb2_grpc
from sailor.examples.deepspeed.train_llm_ckpt_cpu import run, add_model_opt_specific_args, add_model_gpt_specific_args, join_run
from sailor.examples.deepspeed.train_utils import log_time_to_csv, init_csv

WORKER_AGENT_PORT = "50051"


class ElasticWorkerAgent(orchestration_pb2_grpc.WorkerAgentServicer):
    def __init__(self, static_args):
        self.training_process = None
        self.static_args = static_args
        self.model_name = static_args.model_name
        self.lock = threading.Lock()
        self.set = False
        self.cluster_change = multiprocessing.Value('i', 0)
        self.counter_changes = 0
        self.worker_agent_process_id = os.getpid()
        self.training_args = multiprocessing.Manager().Namespace()

    def CheckReady(self, request, context):
        logger.info("----- got request checkready from the controller -----")
        return orchestration_pb2.CheckReadyResponse()

    def Kill(self, request, context):
        logger.info("----- got kill request from the controller -----")
        with self.lock:
            if self.training_process:
                try:
                    self.training_process.terminate()
                    logger.info("Killed agent process")
                except Exception:
                    logger.error(
                        "failed to terminate the training process; here is the traceback:")
                    logger.error(traceback.format_exc())
                else:
                    logger.info("the training process has been terminated")
                finally:
                    self.training_process = None
            else:
                logger.info("Received a KILL request but training process not started yet. Do nothing")
        return orchestration_pb2.KillResponse()

    def ConfigurationChange(self, request, context):
        init_csv()
        log_time_to_csv(["ConfigurationChange", None])
        logger.info(
            "----- got configuration change request from the controller -----")
        configuration = request.configuration
        hyper_params = request.hyper_params
        remote_ckpt = request.remote_ckpt
        self.counter_changes = int(request.configuration.gname.split("-")[1])
        logger.info(
            f"""Cluster change detected, new configuration is {configuration},
            Hyper params are {hyper_params}, Checkpoint group is {remote_ckpt}"""
        )

        training_process_parser = argparse.ArgumentParser()
        training_process_parser = sailor.add_core_arguments(training_process_parser)
        print(f"model_name: {self.model_name}")
        print(f"training process parser: {training_process_parser._get_args()}")
        if self.model_name == 'gpt':
            print("adding gpt specific args")
            training_process_parser, model_specific_args = add_model_gpt_specific_args(training_process_parser)
        elif self.model_name == 'opt':
            print("adding opt specific args")
            try:
                training_process_parser, model_specific_args = add_model_opt_specific_args(training_process_parser)
            except Exception as e:
                logger.error(f"Error in adding opt specific args: {e}")
        else:
            raise ValueError("model_name must be either 'gpt' or 'opt'")
        logger.info(f"model_specific_args: {model_specific_args}")
        # there are two kinds of arguments to feed into the training process:
        # 1. arguments from the controller via ClusterChange request. Those are typically resource-dependent dynamic arguments.
        # 2. arguments from program entry (i.e. in self.static_args). Those never change across restarting the training process.
        var_args = [
            # should correspond to sailor.add_core_arguments
            '--bucket_name', self.static_args.bucket_name,
            '--remote_root_dir', self.static_args.remote_root_dir,
            '--local_root_dir', self.static_args.local_root_dir,
            '--save_every', str(self.static_args.save_every),
            '--log_interval', str(self.static_args.log_interval),
            '--global_batch_size', str(hyper_params.global_batch_size),
            '--micro_batch_size', str(hyper_params.micro_batch_size),
            '--num_stages', str(hyper_params.num_stages),
            '--rank', str(configuration.rank),
            '--world_size', str(configuration.world_size),
            '--master_ip', configuration.master_ip,
            '--master_port', str(configuration.master_port),
            '--num_iters', str(self.static_args.num_iters),
            '--deepspeed',
            '--mixed_precision_training', str(
                self.static_args.mixed_precision_training),
            '--ckpt_group_rank', remote_ckpt.rank,
            '--ckpt_group_master_ip', remote_ckpt.master_ip,
            '--ckpt_group_master_port', remote_ckpt.master_port,
            '--ckpt_group_world_size', remote_ckpt.world_size,
            '--ckpt_group_id', remote_ckpt.group_id,
            '--num_ckpt_groups', remote_ckpt.num_ckpt_groups,
            '--ondem_group_master_ip', remote_ckpt.ondem_master_ip,
            '--ondem_group_master_port', remote_ckpt.ondem_master_port,
        ]
        # next element only appears when the flag is set
        print(f"hyper_params: {hyper_params}")
        if hyper_params.cocktail_sgd:
            var_args.extend(['--cocktail_sgd'])
        if hyper_params.ckpt_cpu:
            var_args.extend(['--ckpt_cpu'])
        # we assume all model_specific args are set from static_args
        for arg_name in model_specific_args:
            value = getattr(self.static_args, arg_name)
            if isinstance(value, list):
                value = ' '.join(value)
            elif isinstance(value, bool):
                if value:
                    var_args.append(f"--{arg_name}")
                continue
            logger.info(f"arg_name: {arg_name}; value: {value}")
            var_args.append(f'--{arg_name}')
            var_args.append(str(value))

        # logger.info(f"var_args: {var_args}")
        training_args = training_process_parser.parse_args(args=var_args)
        # Set attributes of the training_args manager namespace to share to subprocesses
        # CONVERT_TO_INT = ['ckpt_group_rank', 'ckpt_group_world_size', 'ckpt_group_id', 'num_ckpt_groups']
        with self.lock:
            for key, value in vars(training_args).items():
                # val = int(value) if key in CONVERT_TO_INT else value
                setattr(self.training_args, key, value)
        logger.info(f"training_args: {training_args}")
        with self.lock:
            # timing_stack(training_args.rank, os.getpid(), "start")
            # start the training process
            if self.training_process is None:
                if self.counter_changes == 1:
                    self.training_process = multiprocessing.Process(target=run, args=(
                        self.training_args, self.cluster_change, self.worker_agent_process_id, None, ))
                    logger.info(
                        "starting a new training process for the first time...")
                    logger.info(
                        f"sent configuration for training process: {self.training_process}")
                    self.training_process.start()
                    logger.info("a new training process has started")
                else:
                    logger.info(
                        """Starting new training process after a cluster change, VM has just joined.
                        First the snapshot should be retrieved"""
                    )
                    self.training_process = multiprocessing.Process(target=join_run, args=(
                        self.training_args, self.cluster_change, self.worker_agent_process_id, ))
                    logger.info(
                        "starting a new training process for the first time...")
                    logger.info(
                        f"sent configuration for training process: {self.training_process}")
                    self.training_process.start()
            else:
                self.cluster_change.value = 1
                logger.info(
                    "the training process is already running, args have been sent to the live processes")

        return orchestration_pb2.WorkerConfigurationResponse()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    from datasets import load_dataset
    load_dataset('wikitext', 'wikitext-103-v1', split='train')

    pre_parser = argparse.ArgumentParser(description='parser for static args')
    pre_parser.add_argument('--model_name', type=str)
    choice_parse = pre_parser.parse_known_args()[0]
    parser = argparse.ArgumentParser(description='parser for static args')
    if choice_parse.model_name == 'gpt':
        parser, _ = add_model_gpt_specific_args(parser)
    elif choice_parse.model_name == 'opt':
        parser, _ = add_model_opt_specific_args(parser)
    else:
        raise ValueError("model_name must be either 'gpt' or 'opt'")

    parser.add_argument('--bucket_name', type=str,
                        help='the name of the google cloud storage bucket', required=True)
    parser.add_argument('--remote_root_dir', type=str,
                        help='the root directory of the remote storage for checkpoints')
    parser.add_argument('--local_root_dir', type=str,
                        help='the root directory for the local checkpoints')
    parser.add_argument('--save_every', type=int, default=None,
                        help='save checkpoint every n steps')
    parser.add_argument('--num_iters', type=int, default=0,
                        help='number of iterations to train in total; 0 to run infinitely')
    parser.add_argument('--log_interval', type=int,
                        default=20, help='log every n steps')
    parser.add_argument('--mixed_precision_training', type=bool, default=True,
                        help='whether to use mixed precision training')
    user_static_args = parser.parse_args()
    print(user_static_args)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent = ElasticWorkerAgent(user_static_args)
    orchestration_pb2_grpc.add_WorkerAgentServicer_to_server(
        agent, server)
    server.add_insecure_port(f'[::]:{WORKER_AGENT_PORT}')

    def terminate(signum, _):
        if agent.training_process is not None:
            try:
                agent.training_process.terminate()
            except Exception:
                logger.error(
                    "failed to terminate the training process; here is the traceback:")
                logger.error(traceback.format_exc())
            else:
                logger.info("the training process has been terminated")
            finally:
                agent.training_process = None
        done = server.stop(5)
        done.wait()
        logger.info(f"Received {signum}, stop complete!")

    logger.info("starting server")
    server.start()
    signal.signal(signal.SIGTERM, terminate)
    server.wait_for_termination()
