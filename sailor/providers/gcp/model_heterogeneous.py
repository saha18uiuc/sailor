# models heterogenous bandwidth for gcp A100, V100.
import json
from os.path import expanduser

GPUS = ["A100-40", "V100-16"]
GPU_CNTS = ["1", "2", "4"]
ZONES = []

def get_config(
    data,
    send_zone,
    send_gpu_type,
    send_gpu_count,
    recv_zone,
    recv_gpu_type,
    recv_gpu_count
):
    if (send_gpu_type==recv_gpu_type and send_gpu_count==recv_gpu_count):
        return data[send_zone][send_gpu_type][send_gpu_count][recv_zone][recv_gpu_type][recv_gpu_count]

    if (send_gpu_type==recv_gpu_type):
        min_gpu_count = min(send_gpu_count, recv_gpu_count)
        return data[send_zone][send_gpu_type][min_gpu_count][recv_zone][recv_gpu_type][min_gpu_count]

    # for now, just get the one with smallest max bandwidth
    send_max_bw = data[send_zone][send_gpu_type][send_gpu_count][recv_zone][send_gpu_type][send_gpu_count][1]
    recv_max_bw = data[recv_zone][recv_gpu_type][recv_gpu_count][send_zone][recv_gpu_type][recv_gpu_count][1]
    if (send_max_bw < recv_max_bw):
        return data[send_zone][send_gpu_type][send_gpu_count][recv_zone][send_gpu_type][send_gpu_count]
    else:
        return data[recv_zone][recv_gpu_type][recv_gpu_count][send_zone][recv_gpu_type][recv_gpu_count]


def model(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
        for zone in data:
            ZONES.append(zone)

    new_data = {}
    for send_zone in ZONES:
        new_data[send_zone] = {}
        for send_gpu_type in GPUS:
            new_data[send_zone][send_gpu_type] = {}
            for send_gpu_count in GPU_CNTS:
                new_data[send_zone][send_gpu_type][send_gpu_count] = {}
                for recv_zone in ZONES:
                    new_data[send_zone][send_gpu_type][send_gpu_count][recv_zone] = {}
                    for recv_gpu_type in GPUS:
                        new_data[send_zone][send_gpu_type][send_gpu_count][recv_zone][recv_gpu_type] = {}
                        for recv_gpu_count in GPU_CNTS:
                            new_data[send_zone][send_gpu_type][send_gpu_count][recv_zone][recv_gpu_type][recv_gpu_count] = get_config(
                                data,
                                send_zone,
                                send_gpu_type,
                                send_gpu_count,
                                recv_zone,
                                recv_gpu_type,
                                recv_gpu_count
                            )

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2)

if __name__ == "__main__":
    input_file = f'{expanduser("~")}/elastic-spot-ml/sailor/providers/gcp/multizone_bandwidths.json'
    output_file = f'{expanduser("~")}/elastic-spot-ml/sailor/providers/gcp/multizone_bandwidths_het.json'
    model(input_file, output_file)