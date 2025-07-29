def partition_sailor(num_items, num_parts):
    chunk_size = num_items // num_parts
    residual = num_items % num_parts
    residual = num_parts - residual

    stages = [[] for _ in range(num_parts)]
    start = 0
    for i in range(num_parts):
        stage_size = chunk_size if i < residual else chunk_size+1
        stage = list(range(start, stage_size+start))
        start += stage_size
        stages[i] = stage
    return stages

def partition_uniform(num_items, num_parts, verbose=False):
    '''
        Copied from Deepspeed
        https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/utils.py#L581
    '''
    import numpy
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
    else:
        chunksize = num_items // num_parts
        residual = num_items - (chunksize * num_parts)
        parts = numpy.arange(0, (num_parts + 1) * chunksize, chunksize)
        for i in range(residual):
            parts[i + 1:] += 1
        parts = parts.tolist()

    if verbose:
        verbose_parts = []
        for i in range(len(parts)-1):
            stage = list(range(parts[i], parts[i+1]))
            verbose_parts.append(stage)
        return verbose_parts

    return parts

def calculate_exec_per_stage(exec_times, cutpoints):
    part_times = []
    start_idx = 0
    for i, cutpoint in enumerate(cutpoints):
        if i == 0:
            start_idx = 0
        else:
            start_idx = cutpoints[i - 1]
        if i < len(cutpoints) - 1:
            end_idx = cutpoint - 1
        else:
            end_idx = len(exec_times) - 1

        part_times.append(sum(exec_times[start_idx:end_idx + 1]))

    return part_times
