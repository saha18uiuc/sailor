Sailor is based on profiling information for the model and cluster.
* We profile a model's memory footprint and forward/backward pass under different scenarios. For each GPU type, we need 1 machine with >=1 GPUs to profile multiple tensor parallelism degrees. For LLMs, we dont profile all transformer layers since they are identical. This enables us to profile large models on a single machine.
* We profile a cluster's network bandwidth across machines using Torch NCCL-based send/recv for varying message sizes. We then fit a function to get a relation between network bandwidth and message size.

## 1. Model Profiling

### 1. Profile the model

The script *profile.sh* profiles a given model under different combinations of tensor parallelism and microbatch size. We use a big enough *largest* microbatch size, and OOM failures are expected and do not affect profiling. Of course, you can adapt this accordingly. Make sure to provide the *model_name* and *prof_dir*.


```bash
cd /root/sailor/third_party/Megatron-DeepSpeed
bash profile.sh <model_name> <prof_dir>
```
A file will be generated in the form of *profile_model_gpu_tmp_mbs.json*, where *tmp* is the Tensor Model parallel degree, and *mbs* is the microbatch size.

### 2. Generate results for Sailor

After constructing the files for the different combinations of *tmp* and *mbs*, combine them by running:

```bash

cd /root/sailor/sailor/profiling
python gather_profs.py --model-name <model> --gpu-type <gpu_type> --profile-dir <prof_dir> --sailor-parent-dir <path_to_sailor_dir>

```

This will:
1. Generate a file of the form *model/gpu/profile.json* under *sailor/Planner/sailor_planner/profiles/*, containing timing information.
2. Append memory-specific info for the model in the file *sailor/Planner/llm_info.json*.
3. Append timing-specific info for this combination of model and gpu in the file *sailor/Planner/simulations/profiles_tmp.json*, which is the file used for simulations

### 3. Generate results for baselines

All baselines require some profiling information that we can extract from the data we already collected. Some baselines require some extra information which we collect:

### For Varuna

1. Collect the extra info:

```bash
cd /root/sailor/third_party/Megatron-DeepSpeed
bash profile_varuna.sh <model_name> <prof_dir>
```

2. Generate profiles:

```bash

cd /root/sailor/sailor/profiling/baselines
bash generate_varuna.sh
```

Adapt the GPU type, and number of GPUs per VM accordingly in the file

### For Galvatron

1. Collect the extra info:

```bash
cd /root/sailor/third_party/Megatron-DeepSpeed
bash profile_galvatron.sh <model_name> <prof_dir>
```

2. Generate profiles:

```bash

cd /root/sailor/sailor/profiling/baselines
bash generate_galvatron.sh
```

Adapt the GPU type, and number of GPUs per VM accordingly in the file

### For the rest baselines

Generate profiles:

```bash

cd /root/sailor/sailor/profiling/baselines
bash generate.sh
```

Adapt the name of the baseline, GPU type, and number of GPUs per VM accordingly in the file.

## 2. Network profiling