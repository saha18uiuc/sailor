from kubernetes import client, config
import time
import sys

stride = int(sys.argv[1])
path = sys.argv[2]


def get_avail_pods():

    config.load_config()

    v1 = client.CoreV1Api()
    ret = v1.list_pod_for_all_namespaces(
        label_selector="app=elastic-ml-worker",
        field_selector="status.phase=Running",
        watch=False)

    active_pods = [x for x in ret.items if not (
        x.status.container_statuses[0].state.running is None)]
    pod_list = [x.metadata.name for x in active_pods]
    return pod_list


def write_avail_pods(vlist, avail, filename):

    f = open(filename, 'w')
    n = len(vlist)
    for j in range(n):
        if avail[j]:
            f.write(vlist[j] + "\n")
    f.flush()


def get_gpu_trace(path, stride):
    # get num of GPUs
    f = open(path, 'r')
    lines = f.readlines()
    total = len(lines)
    gpus = []

    j = 0
    while j < total:
        # read <stride> lines
        i = 0
        running = 0
        while i < stride and j < total:
            line = lines[j]
            tokens = line.split()
            if ('RUNNING' in tokens):
                num_gpus = int(tokens[2].split("-")[4])
                running += num_gpus
            i += 1
            j += 1

        gpus.append(running)

    return gpus


def sim_avail_pods(pod_list, N, trace, avail_file):
    # config availability

    alive = [True]*(N//4)  # start with all machines alive
    trace_len = len(trace)
    sleep_time = 1  # how often to change availability
    i = 0
    while i < trace_len:
        cur_avail = trace[i]
        # cause fails
        print(cur_avail)
        to_rem = N-cur_avail
        j = 0
        while j*4 < to_rem:  # 4 gpus per VM
            alive[j] = False
            j += 1

        # rest are alive
        while j*4 < N:
            alive[j] = True
            j += 1

        print(alive)
        # write_avail_pods(pod_list, alive, avail_file)
        i += 1
        time.sleep(sleep_time)


# get current pod list
# pod_list = get_avail_pods()
# print(pod_list)
# trace
gpu_trace = get_gpu_trace(path, stride)
print(gpu_trace)
sim_avail_pods([], 64, gpu_trace, 'trace')
