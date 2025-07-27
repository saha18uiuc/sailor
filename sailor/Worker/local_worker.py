# Local worker that spawns a training process - used for ease of development
import argparse
from deepspeed.utils import logger
import sailor
from sailor.protos import orchestration_pb2
from sailor.examples.deepspeed.train_llm import run, add_model_opt_specific_args, add_model_gpt_specific_args
from torch import multiprocessing
import torch


def start_training(static_args, training_args, model_name):
    training_process_parser = argparse.ArgumentParser()
    training_process_parser = sailor.add_core_arguments(training_process_parser)

    print(f"training process parser: {training_process_parser._get_args()}")
    if model_name == 'gpt':
        print("adding gpt specific args")
        training_process_parser, model_specific_args = add_model_gpt_specific_args(training_process_parser)
    elif model_name == 'opt':
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

    print(training_args)

    opt_params = orchestration_pb2.OptParams(
            lr=training_args["lr"],
            momentum=training_args["momentum"],
            wd=training_args["wd"]
    )

    hyper_params = orchestration_pb2.HyperParams(
            num_stages=training_args["num_stages"],
            cocktail_sgd=False,
            micro_batch_size=training_args["micro_batch_size"],
            global_batch_size=training_args["global_batch_size"],
            opt_params=opt_params,
        )

    var_args = [
        # should correspond to sailor.add_core_arguments
        '--bucket_name', static_args.bucket_name,
        '--remote_root_dir', static_args.remote_root_dir,
        '--local_root_dir', static_args.local_root_dir,
        '--save_every', str(static_args.save_every),
        '--log_interval', str(static_args.log_interval),
        '--global_batch_size', str(hyper_params.global_batch_size),
        '--micro_batch_size', str(hyper_params.micro_batch_size),
        '--num_stages', str(hyper_params.num_stages),
        '--rank', str(static_args.rank),
        '--local_rank', str(static_args.rank % torch.cuda.device_count()),
        '--world_size', str(static_args.world_size),
        '--master_ip', static_args.master_ip,
        '--master_port', str(static_args.master_port),
        '--num_iters', str(static_args.num_iters),
        '--deepspeed'
    ]

    if hyper_params.cocktail_sgd:
        var_args += ['--cocktail_sgd']

    if static_args.use_master_vm:
        var_args += ['--use_master_vm']

    # we assume all model_specific args are set from static_args
    for arg_name in model_specific_args:
        value = getattr(static_args, arg_name)
        if isinstance(value, list):
            value = ' '.join(value)
        elif isinstance(value, bool):
            if value:
                var_args.append(f"--{arg_name}")
            continue
        logger.info(f"arg_name: {arg_name}; value: {value}")
        var_args.append(f'--{arg_name}')
        var_args.append(str(value))

    if static_args.mixed_precision_training:
        var_args += ['--mixed_precision_training']

    logger.info(f"var_args: {var_args}")
    training_args = training_process_parser.parse_args(args=var_args)

    # # start the training process
    logger.info("starting a new training process...")
    training_process = multiprocessing.Process(
                target=run, args=(training_args,))
    training_process.start()
    logger.info("a new training process has started")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    # pre-download datasets
    from datasets import load_dataset
    load_dataset('wikitext', 'wikitext-103-v1', split='train')

    # We need to parse the model-specific arguments first to determine the model name
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
    parser.add_argument('--mixed_precision_training', default=False, action='store_true',
                        help='whether to use mixed precision training')
    parser.add_argument('--use_master_vm', default=False,
                        action='store_true', help="use master VM to initialize PyTorch distributed group")

    # these are additional, for local testing
    parser.add_argument('--rank', type=int, help='Rank of the worker', required=True)
    parser.add_argument('--world_size', type=int, help='World size', required=True)
    parser.add_argument('--master_port', type=int, help='Master port', required=True)
    parser.add_argument('--master_ip', type=str, help='Master IP', required=True)

    parser.add_argument('--momentum', type=float, help='Momentum', default=0.1)
    parser.add_argument('--wd', type=float, help='Weight decay', default=0.1)
    parser.add_argument('--num_stages', type=int, help='Number of stages', required=True)
    parser.add_argument('--micro_batch_size', type=int, help='Microbatch size', required=True)
    parser.add_argument('--global_batch_size', type=int, help='Global batch size', required=True)

    static_user_args = parser.parse_args()

    traning_dict = {
        "lr": static_user_args.lr,
        "momentum": static_user_args.momentum,
        "wd": static_user_args.wd,
        "num_stages": static_user_args.num_stages,
        "micro_batch_size": static_user_args.micro_batch_size,
        "global_batch_size": static_user_args.global_batch_size
    }

    start_training(static_user_args, traning_dict, choice_parse.model_name)
