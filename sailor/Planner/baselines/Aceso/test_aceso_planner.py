from aceso_planner import AcesoPlanner
# from sailor.Planner.baselines.Aceso.aceso_planner import AcesoPlanner

if __name__ == "__main__":
    profile_dir = "profiler/RTX-3090/"
    planner = AcesoPlanner(profiling_file_dir=profile_dir, fp16=True)
    cluster_config = {
        'gpus_per_node': 8,
        'num_nodes': 2,
        'mem_per_gpu': 25769803776, 
        "gpu_type": "RTX-3090"
    }
    training_config = {
        "global_batch_size": 1024,
        "type": "gpt2",
        "hidden_size": 1024,
        "sequence_length": 2048,
        "num_layers": 24,
        "vocab_size": 50272,
        "model_name": "OPT-350",
        "model": "OPT-350",
        "optimizer": "Adam",
        "heads": 16,
        "head_dim": 64
    }

    plan = planner.get_plan(cluster_config, training_config)
    print(plan)
