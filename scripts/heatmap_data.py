import sys
import numpy as np
import csv


def line(a_bits, w_bits, baseline_avg, synthesis_times):
    data = np.loadtxt('data/raw_bitserial_conv2d_nhwc_a%dw%d_rasp3b.csv' % (a_bits, w_bits))[0]
    data_avg = np.average(data)
    speedup = baseline_avg / data_avg
    config = "A" + str(a_bits) + "W" + str(w_bits)
    print(config, speedup, synthesis_times[config])

def gen():
    baseline_avg = np.average(np.loadtxt('data/raw_tvm_conv2d_fp_rasp3b.csv')[0])

    # Synthesis times
    synthesis_times = {}
    with open('data/synthesis-time.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            synthesis_times[row[0]] = float(row[-1]) / 1000.0

    line(1, 1, baseline_avg, synthesis_times)
    line(2, 1, baseline_avg, synthesis_times)
    line(3, 1, baseline_avg, synthesis_times)
    line(1, 2, baseline_avg, synthesis_times)
    line(1, 3, baseline_avg, synthesis_times)
    line(2, 2, baseline_avg, synthesis_times)


if __name__ == '__main__':
    gen()