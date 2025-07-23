from dataclasses import dataclass

# Review GCP to see valid combinations (reviewed: Feb 2024)
GPU_CPU_COMBINATIONS = {
    'A100_40': ['a2-highgpu'],
    'A100_80': ['a2-ultragpu'],
    'L4': ['g2'],
}

# name_GPU -> name
CARTESIAN_PRODUCT = {
    'V100-1': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8',
               'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8',
               'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8'],
    'V100-2': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16',
               'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16',
               'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16'],
    'V100-4': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16', 'n1-standard-32',
               'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16', 'n1-highmem-32',
               'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16', 'n1-highcpu-32'],
    'V100-8': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16', 'n1-standard-32', 'n1-standard-64', 'n1-standard-96',
               'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16', 'n1-highmem-32', 'n1-highmem-64', 'n1-highmem-96',
               'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16', 'n1-highcpu-32', 'n1-highcpu-64', 'n1-highcpu-96'],

    'P100-1': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16',
               'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16',
               'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16'],
    'P100-2': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16', 'n1-standard-32',
               'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16', 'n1-highmem-32',
               'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16', 'n1-highcpu-32'],
    'P100-4': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16', 'n1-standard-32', 'n1-standard-64', 'n1-standard-96',
               'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16', 'n1-highmem-32', 'n1-highmem-64', 'n1-highmem-96',
               'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16', 'n1-highcpu-32', 'n1-highcpu-64', 'n1-highcpu-96'],

    'T4-1': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16', 'n1-standard-32',
             'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16', 'n1-highmem-32',
             'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16', 'n1-highcpu-32'],
    'T4-2': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16', 'n1-standard-32',
             'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16', 'n1-highmem-32',
             'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16', 'n1-highcpu-32'],
    'T4-4': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16', 'n1-standard-32', 'n1-standard-64', 'n1-standard-96',
             'n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16', 'n1-highmem-32', 'n1-highmem-64', 'n1-highmem-96',
             'n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16', 'n1-highcpu-32', 'n1-highcpu-64', 'n1-highcpu-96'],
}

A100_40 = {
    'type': 'GPU-CPU',
    # highgpu-16g = megagpu-16g
    'name': ['a2-highgpu-1g', 'a2-highgpu-2g', 'a2-highgpu-4g', 'a2-highgpu-8g', 'a2-highgpu-16g'],
    'name_GPU': 'A100-40',
    'num_GPUs': [1, 2, 4, 8, 16],
    'gpu_memory': [40, 80, 160, 320, 640],
    'gpu_memory_unit': 'GB',
    'gpu_memory_bandwidth': 1.6*1024,
    'gpu_memory_bandwidth_unit': 'GB/s',
    'throughput_FP32': 19.5,
    'throughput_FP16_32': 312,
    'throughput_unit': 'TFLOPS',
    'NVLink': True,
    'NVLink_bandwidth': 600,
    'NVLink_unit': 'GB/s',
    'compute_requirement': 'a2-highgpu',
    'max_network_egress': 50,
    'num_CPUs': [12, 24, 48, 96, 96],
    'cpu_memory': [85, 170, 340, 680, 1360],
    'cpu_memory_unit': 'GB',
    'network_bandwidth': [24, 32, 50, 100, 100],
    'network_bandwidth_unit': 'Gbps',
    'cost_per_hour': [3.673477, 7.346954, 14.693908, 29.387816, 58.775632],
    'cost_per_hour_preemptible': [1.469391, 2.938782, 5.877564, 11.755128, 23.510256],
    'cost_unit': 'USD',
    'zone': 'us-west1',
    'time_input_data': '12-2023'
}

A100_80 = {
    'type': 'GPU-CPU',
    'name': ['a2-ultragpu-1g', 'a2-ultragpu-2g', 'a2-ultragpu-4g', 'a2-ultragpu-8g'],
    'name_GPU': 'A100-80',
    'num_GPUs': [1, 2, 4, 8],
    'gpu_memory': [80, 160, 320, 640],
    'gpu_memory_unit': 'GB',
    'gpu_memory_bandwidth': 1.9*1024,
    'gpu_memory_bandwidth_unit': 'GB/s',
    'throughput_FP32': 19.5,
    'throughput_FP16_32': 312,
    'throughput_unit': 'TFLOPS',
    'NVLink': True,
    'NVLink_bandwidth': 600,
    'NVLink_unit': 'GB/s',
    'compute_requirement': 'a2-ultragpu',
    'max_network_egress': 50,
    'num_CPUs': [12, 24, 48, 96],
    'cpu_memory': [170, 340, 680, 1360],
    'cpu_memory_unit': 'GB',
    'network_bandwidth': [24, 32, 50, 100],
    'network_bandwidth_unit': 'Gbps',
    'cost_per_hour': [6.34, 12.68, 25.36, 50.72],
    'cost_per_hour_preemptible': [1.97, 3.95, 7.88, 15.79],
    'cost_unit': 'USD',
    'zone': 'us-west1',
    'time_input_data': '12-2023'
}
L4_G2 = {
    'type': 'GPU-CPU',
    'name': ['g2-standard-4', 'g2-standard-8', 'g2-standard-12', 'g2-standard-16', 'g2-standard-24', 'g2-standard-32', 'g2-standard-48', 'g2-standard-96'],
    'name_GPU': 'G2',
    'num_GPUs': [1, 1, 1, 1, 2, 1, 4, 8],
    'gpu_memory': [24, 24, 24, 24, 48, 24, 96, 192],
    'gpu_memory_unit': 'GB',
    'gpu_memory_bandwidth': 300,
    'gpu_memory_bandwidth_unit': 'GB/s',
    'throughput_FP32': 30.3,
    'throughput_FP16_32': 121,
    'throughput_unit': 'TFLOPS',
    'NVLink': False,
    'NVLink_bandwidth': None,
    'NVLink_unit': None,
    'compute_requirement': 'g2',
    'max_network_egress': 50,
    'num_CPUs': [4, 8, 12, 16, 24, 32, 48, 96],
    'cpu_memory': [16, 32, 48, 64, 96, 128, 192, 384],
    'cpu_memory_unit': 'GB',
    'network_bandwidth': [10, 16, 16, 32, 32, 32, 50, 100],
    'network_bandwidth_unit': 'Gbps',
    'cost_per_hour': [0.7068, 0.8536, 1.0004, 1.1472, 2.0008, 1.7344, 4.0017, 8.0033],
    'cost_per_hour_preemptible': [0.2267, 0.2854, 0.3441, 0.4029, 0.6883, 0.6378, 1.3766, 2.7534],
    'cost_unit': 'USD',
    'zone': 'us-west1',
    'time_input_data': '12-2023'
}

V100 = {
    'type': 'GPU',
    'name': ['V100-1', 'V100-2', 'V100-4', 'V100-8'],
    'name_GPU': 'V100',
    'num_GPUs': [1, 2, 4, 8],
    'gpu_memory': [16, 32, 64, 128],
    'gpu_memory_unit': 'GB',
    'gpu_memory_bandwidth': 900,
    'gpu_memory_bandwidth_unit': 'GB/s',
    'throughput_FP32': 15.7,
    'throughput_FP16_32': 125,
    'throughput_unit': 'TFLOPS',
    'NVLink': True,
    'NVLink_bandwidth': 300,
    'NVLink_unit': 'GB/s',
    'compute_requirement': 'n1',
    'max_network_egress': 100,
    'cost_per_hour': [2.48, 2.48*2, 2.48*4, 2.48*8],
    'cost_per_hour_preemptible': [0.992, 0.992*2, 0.992*4, 0.992*8],
    'cost_unit': 'USD',
    'zone': 'us-west1',
    'time_input_data': '12-2023'
}

P100 = {
    'type': 'GPU',
    'name': ['P100-1', 'P100-2', 'P100-4'],
    'name_GPU': 'P100',
    'num_GPUs': [1, 2, 4],
    'gpu_memory': [16, 32, 64],
    'gpu_memory_unit': 'GB',
    'gpu_memory_bandwidth': 192,
    'gpu_memory_bandwidth_unit': 'GB/s',
    'throughput_FP32': 9.3,
    'throughput_FP16_32': None,
    'throughput_unit': 'TFLOPS',
    'NVLink': True,
    'NVLink_bandwidth': None,
    'NVLink_unit': None,
    'compute_requirement': 'n1',
    'max_network_egress': 50,
    'cost_per_hour': [1.46, 1.46*2, 1.46*4],
    'cost_per_hour_preemptible': [0.584, 0.584*2, 0.584*4],
    'cost_unit': 'USD',
    'zone': 'us-west1',
    'time_input_data': '12-2023'
}

T4 = {
    'type': 'GPU',
    'name': ['T4-1', 'T4-2', 'T4-4'],
    'name_GPU': 'T4',
    'num_GPUs': [1, 2, 4],
    'gpu_memory': [16, 32, 64],
    'gpu_memory_unit': 'GB',
    'gpu_memory_bandwidth': 320,
    'gpu_memory_bandwidth_unit': 'GB/s',
    'throughput_FP32': 8.1,
    'throughput_FP16_32': 65,
    'throughput_unit': 'TFLOPS',
    'NVLink': False,
    'NVLink_bandwidth': None,
    'NVLink_unit': None,
    'compute_requirement': 'n1',
    'max_network_egress': 50,
    'cost_per_hour': [0.35, 0.35*2, 0.35*4],
    'cost_per_hour_preemptible': [0.14, 0.14*2, 0.14*4],
    'cost_unit': 'USD',
    'zone': 'us-west1',
    'time_input_data': '12-2023'
}

N1_STD = {
    'type': 'CPU',
    'name': ['n1-standard-1', 'n1-standard-2', 'n1-standard-4', 'n1-standard-8', 'n1-standard-16', 'n1-standard-32', 'n1-standard-64', 'n1-standard-96'],
    'num_CPUs': [1, 2, 4, 8, 16, 32, 64, 96],
    'cpu_memory': [3.75, 7.5, 15, 30, 60, 120, 240, 360],
    'cpu_memory_unit': 'GB',
    'network_bandwidth': [2, 10, 10, 16, 32, 32, 32, 32],
    'network_bandwidth_unit': 'Gbps',
    'cost_per_hour': [0.0475, 0.0950, 0.1899, 0.3798, 0.7596, 1.5192, 3.039984, 4.559976],
    'cost_per_hour_preemptible': [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.60776, 0.91164],
    'cost_unit': 'USD',
    'zone': 'us-west1',
    'time_input_data': '12-2023'
}

N1_HM = {
    'type': 'CPU',
    'name': ['n1-highmem-2', 'n1-highmem-4', 'n1-highmem-8', 'n1-highmem-16', 'n1-highmem-32', 'n1-highmem-64', 'n1-highmem-96'],
    'num_CPUs': [2, 4, 8, 16, 32, 64, 96],
    'cpu_memory': [13, 26, 52, 104, 208, 416, 624],
    'cpu_memory_unit': 'GB',
    'network_bandwidth': [10, 10, 16, 32, 32, 32, 32],
    'network_bandwidth_unit': 'Gbps',
    'cost_per_hour': [0.1183, 0.2366, 0.4732, 0.9464, 1.8928, 3.785696, 5.678544],
    'cost_per_hour_preemptible': [0.0249, 0.0498, 0.0996, 0.1992, 0.3985, 0.756832, 1.135248],
    'cost_unit': 'USD',
    'zone': 'us-west1',
    'time_input_data': '12-2023'
}

N1_HC = {
    'type': 'CPU',
    'name': ['n1-highcpu-2', 'n1-highcpu-4', 'n1-highcpu-8', 'n1-highcpu-16', 'n1-highcpu-32', 'n1-highcpu-64', 'n1-highcpu-96'],
    'num_CPUs': [2, 4, 8, 16, 32, 64, 96],
    'cpu_memory': [1.80, 3.60, 7.20, 14.40, 28.80, 57.60, 86.40],
    'cpu_memory_unit': 'GB',
    'network_bandwidth': [10, 10, 16, 32, 32, 32, 32],
    'network_bandwidth_unit': 'Gbps',
    'cost_per_hour': [0.0708486, 0.1416972, 0.2833944, 0.5667888, 1.1335776, 2.2671552, 3.4007328],
    'cost_per_hour_preemptible': [0.0141646, 0.0283292, 0.0566584, 0.1133168, 0.2266336, 0.4532672, 0.6799008],
    'cost_unit': 'USD',
    'zone': 'us-west1',
    'time_input_data': '12-2023'
}


@dataclass(frozen=True)
class Disk:
    type: str
    cost_per_hour_per_GB: float
    time_input_data: str = '12-2023'


PD_DISK = Disk(type='persistent-disk', cost_per_hour_per_GB=0.04,
               time_input_data='12-2023')
