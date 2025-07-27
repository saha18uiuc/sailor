import sys
import grpc
from concurrent import futures
import argparse
import signal
from deepspeed.utils import logger
import traceback
import sailor
from sailor.protos import orchestration_pb2_grpc, orchestration_pb2
from sailor.examples.deepspeed.train_llm import run, add_model_opt_specific_args, add_model_gpt_specific_args

from torch import multiprocessing
import threading
import torch
import time
import json

from sailor.examples.deepspeed.train_utils import init_csv, log_time_to_csv
WORKER_AGENT_PORT = "50051"
TRAINING_START_PORT = "51000"

class ElasticWorkerAgent(orchestration_pb2_grpc.WorkerAgentServicer):
    def __init__(self, model_name, static_args, use_megatron):
        self.training_processes = []
        self.static_args = static_args
        self.model_name = model_name
        self.lock = threading.Lock()
        self.use_megatron = use_megatron
        self.ds_dict = {}
        if self.use_megatron:
            with open(self.static_args.ds_config_file, 'r') as f:
                ds_dict = json.load(f)
            self.ds_dict = ds_dict

    def CheckReady(self, request, context):
        logger.info("----- got request from the controller -----")
        return orchestration_pb2.CheckReadyResponse()

    def Kill(self, request, context):
        logger.info("----- got kill request from the controller -----")
        with self.lock:
            try:
                for x in self.training_processes:
                    x.terminate()
            except Exception:
                logger.error(
                    "failed to terminate the training process; here is the traceback:")
                logger.error(traceback.format_exc())
            else:
                logger.info("the training process has been terminated")
            finally:
                self.training_processes = []
        return orchestration_pb2.KillResponse()

    def ConfigurationChange(self, request, context):
        init_csv()
        log_time_to_csv(["ConfigurationChange", None])
        configuration = request.configuration
        hyper_params = request.hyper_params
        logger.info(
            f"Cluster change detected, new configuration is {configuration}")
        logger.info(f"Hyper params are {hyper_params}")

        if self.use_megatron:
            self.ds_dict['train_batch_size'] = hyper_params.global_batch_size
            self.ds_dict['train_micro_batch_size_per_gpu'] = hyper_params.micro_batch_size
            self.ds_dict['gradient_accumulation_steps'] = hyper_params.ga_steps

            with open(self.static_args.ds_config_file, 'w') as f:
                json.dump(self.ds_dict, f, indent=2)

            all_ranks_configs = []
            for stage in configuration.all_stages:
                stage_config = []
                for replica in stage.stage_replicas:
                    stage_config.append([int(x) for x in replica.replica_ranks])
                all_ranks_configs.append(stage_config)

            print(f"all_ranks_configs is {all_ranks_configs}")
            with open("dist_config.json", 'w') as f:
                json.dump(all_ranks_configs, f)

        layers_per_stage = json.dumps(list(configuration.layers_per_stage))

        for i,rank in enumerate(configuration.ranks):
            print(f"Start worker with rank {rank}")
            self.spawn_worker(
                hyper_params.global_batch_size,
                hyper_params.micro_batch_size,
                hyper_params.num_stages,
                rank,
                configuration.world_size,
                configuration.master_ip,
                configuration.master_port,
                self.static_args.ds_config_file,
                (i==0),
                configuration.tensor_parallelism,
                configuration.pipeline_parallelism,
                configuration.data_parallelism,
                configuration.max_tensor_parallelism,
                layers_per_stage
            )

        return orchestration_pb2.WorkerConfigurationResponse()


    def spawn_worker(
        self,
        global_batch_size,
        micro_batch_size,
        num_stages,
        rank,
        world_size,
        master_ip,
        master_port,
        ds_config_file,
        kill_all=False,
        tensor_model_parallel_size=None,
        pipeline_model_parallel_size=None,
        data_parallelism=None,
        max_tensor_parallelism=None,
        layers_per_stage=None
    ):

        if self.use_megatron:
            sys.path.append("/root/sailor/third_party/Megatron-DeepSpeed")
            from train_llm import run_megatron as run
            from megatron.arguments import parse_args

            training_args = parse_args(extra_args_provider=None, ignore_unknown_args=True)
            training_args.rank = rank
            training_args.world_size = world_size
            training_args.master_ip = master_ip
            training_args.master_port = master_port

            training_args.global_batch_size = global_batch_size
            training_args.micro_batch_size = micro_batch_size
            print(training_args)
            if tensor_model_parallel_size is not None:
                training_args.tensor_model_parallel_size = tensor_model_parallel_size
                training_args.pipeline_model_parallel_size = pipeline_model_parallel_size
                training_args.data_parallelism = data_parallelism
                training_args.max_tensor_parallelism = max_tensor_parallelism
                training_args.distributed_config_file = "dist_config.json"
                training_args.layers_per_stage = layers_per_stage
            training_args.deepspeed_config = ds_config_file
        else:
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
            # 1. arguments from the controller via ConfigurationChange request.
            # Those are typically resource-dependent dynamic arguments.
            # 2. arguments from program entry (i.e. in self.static_args).
            # Those never change across restarting the training process.

            var_args = [
                # should correspond to sailor.add_core_arguments
                '--bucket_name', self.static_args.bucket_name,
                '--remote_root_dir', self.static_args.remote_root_dir,
                '--local_root_dir', self.static_args.local_root_dir,
                '--save_every', str(self.static_args.save_every),
                '--log_interval', str(self.static_args.log_interval),
                '--global_batch_size', str(global_batch_size),
                '--micro_batch_size', str(micro_batch_size),
                '--num_stages', str(num_stages),
                '--rank', str(rank),
                '--local_rank', str(rank % torch.cuda.device_count()),
                '--world_size', str(world_size),
                '--master_ip', master_ip,
                '--master_port', str(master_port),
                '--num_iters', str(self.static_args.num_iters),
                '--deepspeed',
            ]

            if self.static_args.use_master_vm:
                var_args += ['--use_master_vm']

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

            if self.static_args.mixed_precision_training:
                var_args += ['--mixed_precision_training']

            logger.info(f"var_args: {var_args}")
            training_args = training_process_parser.parse_args(args=var_args)

        with self.lock:
            if kill_all:
                # kill the training process
                try:
                    for x in self.training_processes:
                        x.terminate()
                except Exception:
                    logger.error(
                        "failed to terminate the training process; here is the traceback:")
                    logger.error(traceback.format_exc())
                else:
                    logger.info("the training process has been terminated")
                finally:
                    self.training_processes = []

            # start the training process
            logger.info("starting a new training process...")
            new_proc = multiprocessing.Process(
                target=run, args=(training_args,))
            new_proc.start()
            self.training_processes.append(new_proc)
            logger.info("a new training process has started")



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    # pre-download datasets
    from datasets import load_dataset
    load_dataset('wikitext', 'wikitext-103-v1', split='train')

    # We need to parse the model-specific arguments first to determine the model name
    pre_parser = argparse.ArgumentParser(description='parser for static args')
    pre_parser.add_argument('--use_megatron', default=False, action='store_true')
    pre_parser.add_argument('--model_name', type=str)
    choice_parse = pre_parser.parse_known_args()[0]
    use_megatron = choice_parse.use_megatron
    model_name = choice_parse.model_name
    parser = argparse.ArgumentParser(description='parser for static args')
    if not use_megatron:
        if model_name == 'gpt':
            parser, _ = add_model_gpt_specific_args(parser)
        elif model_name == 'opt':
            parser, _ = add_model_opt_specific_args(parser)
        else:
            raise ValueError("model_name must be either 'gpt' or 'opt'")
    parser.add_argument('--bucket_name', type=str,
                        help='the name of the google cloud storage bucket')
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
    parser.add_argument('--mixed_precision_training', default=False, action='store_true',
                        help='whether to use mixed precision training')
    parser.add_argument('--use_master_vm', default=False,
                        action='store_true', help="use master VM to initialize PyTorch distributed group")
    parser.add_argument('--with_controller', default=False, action='store_true',
                        help='run with controller')
    parser.add_argument('--ds_config_file', type=str,
                        help='DeepSpeed config file', required=True)

    ## for testing ##
    parser.add_argument('--global_batch_size', type=int, default=0, help='Global batch size')
    parser.add_argument('--micro_batch_size', type=int, default=0, help='Micro batch size')
    parser.add_argument('--num_stages', type=int, default=0, help='Number of stages')
    parser.add_argument('--rank', type=int, default=0, help='Worker rank')
    parser.add_argument('--world_size', type=int, default=1, help='World size')
    parser.add_argument('--master_ip', type=str, default="", help='Master IP')
    parser.add_argument('--master_port', type=str, default="", help='Master Port')

    static_user_args = parser.parse_known_args()[0]

    if static_user_args.with_controller:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        agent = ElasticWorkerAgent(model_name, static_user_args, use_megatron)
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

        logger.info("server closed")
    else:
        # for testing purposes
        agent = ElasticWorkerAgent(model_name, static_user_args, use_megatron)
        agent.spawn_worker(
            static_user_args.global_batch_size,
            static_user_args.micro_batch_size,
            static_user_args.num_stages,
            static_user_args.rank,
            static_user_args.world_size,
            static_user_args.master_ip,
            static_user_args.master_port,
            static_user_args.ds_config_file
        )