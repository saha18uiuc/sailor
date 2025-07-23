import numpy as np


def get_time_on_config(x, constants):
    x_mb = np.log2(x / 1e6)
    bw_gb = 0
    for i in range(len(constants)):
        bw_gb += constants[i] * np.power(x_mb, len(constants) - i - 1)
    #print(f"x is {x}, bw_gb is {bw_gb}")
    time = (x / 1e9)/bw_gb
    return time


def get_bw_on_config(x, constants, poly=True):
    x_mb = np.log2(x / 1e6)
    bw_gb = 0
    for i in range(len(constants)):
        bw_gb += constants[i] * np.power(x_mb, len(constants) - i - 1)
    return bw_gb