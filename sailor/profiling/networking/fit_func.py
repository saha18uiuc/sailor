import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def func(x, coeffs):
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i] * np.power(np.log2(x), len(coeffs) - i - 1)
    return y

def get_data(input_file):
    df = pd.read_csv(input_file)
    x = [x/1e6 for x in df['bytes']]
    bw = df['bw_GB_s']
    return list(x), list(bw)


xdata, ydata = get_data(sys.argv[1])
print(xdata, ydata)
print(f"Max bw is {max(ydata) * 1e9} bytes/sec")

plt.plot(xdata, ydata, 'o', label='data')
coeffs = np.polyfit(np.log2(xdata), ydata, 2)

print(f"PARAMS FOUND: {coeffs}")
plt.plot(xdata, func(xdata, coeffs), 'r-', label='polyfit')
plt.legend()
plt.xlabel("Message Size (MB)")
plt.ylabel("Bandwidth (GB/s)")
plt.xscale("log", base=2)
plt.ylim([0, max(ydata) * 1.1])
plt.savefig("data.png")