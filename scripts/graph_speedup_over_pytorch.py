import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import csv
import colorsys
import seaborn as sns
import math

def confidence_interval(stddev):
    samples = 10
    z95 = 2.228
    return np.array(stddev) / math.sqrt(samples) * z95

def calculate_speedup_mean_ci(baseline, comparison):
    # Means E(X) * E(1/Y)
    baseline_mean = np.mean(baseline, axis=1)
    inverse_comparison_mean = np.mean(1.0 / comparison, axis=1)
    speedup_mean = baseline_mean * inverse_comparison_mean

    # Variance E(X^2)E(1/Y^2) - E^2[X]E^2[1/Y]
    baseline2_mean = np.mean(baseline*baseline, axis=1)
    inverse2_comparison_mean = np.mean(1.0 / (comparison * comparison), axis=1)
    baseline_mean2 = baseline_mean * baseline_mean
    inverse_comparison_mean2 = inverse_comparison_mean * inverse_comparison_mean
    speedup_var = baseline2_mean*inverse2_comparison_mean - baseline_mean2*inverse_comparison_mean2
    speedup_stddev = np.sqrt(speedup_var)

    ci = confidence_interval(speedup_stddev)

    return speedup_mean, ci

def plot(target):
      
    # Value of each run in seconds
    baseline_raw = np.genfromtxt('data/raw_pytorch_bitserial_conv2d_a2w1_single.csv', delimiter=',')
    a2w1_single_raw = np.loadtxt('data/raw_bitserial_conv2d_nhwc_a2w1_%s_single.csv' % target)
    a2w1_multi_raw = np.loadtxt('data/raw_bitserial_conv2d_nhwc_a2w1_%s.csv' % target)
    a2w1_single_mean, a2w1_single_ci = calculate_speedup_mean_ci(baseline_raw, a2w1_single_raw)
    a2w1_multi_mean, a2w1_multi_ci= calculate_speedup_mean_ci(baseline_raw, a2w1_multi_raw)

    # Geometric mean across all convs
    a2w1_single_geo_mean = np.sum(np.mean(baseline_raw, axis=1)) / np.sum(np.mean(a2w1_single_raw, axis=1))
    a2w1_multi_geo_mean = np.sum(np.mean(baseline_raw, axis=1)) / np.sum(np.mean(a2w1_multi_raw, axis=1))

    # Colors, Labels
    sns.set_context("paper")
    sns.set_style("whitegrid", {"xtick.top": False})
    sns.set_palette("deep")
    labels = [r'$A^2W^1$ single threaded', r'$A^2W^1$ multi threaded']

    index = np.arange(len(a2w1_single_mean))
    bar_width = 0.30
    gap = (1 - bar_width * len(labels)) / 2
    fontsize = 16

    fig = plt.figure(figsize=(6.5, 4))
    ax = plt.subplot(111)
    legends = []

    err_bar_settings = dict(lw=1, capsize=5, capthick=1)

    # b0 = plt.bar(index, baseline, width=bar_width, align='center')
    b1 = plt.bar(index, a2w1_single_mean, width=bar_width, align='center',
        yerr=a2w1_single_ci, error_kw=err_bar_settings)
    b2 = plt.bar(index + bar_width, a2w1_multi_mean, width=bar_width, align='center',
        yerr=a2w1_multi_ci,error_kw=err_bar_settings)
    # legends.append(b0)
    legends.append(b1)
    legends.append(b2)


    ax.set_ylabel('Speedup over Pytorch', fontsize=fontsize)
    xlabels = ('C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12')
    xticks = (index + bar_width/2)

    plt.xticks(xticks, xlabels, fontsize=fontsize)
    plt.tick_params(axis='x', which='both', bottom='off', top='off')

    # grid line
    ax.set_axisbelow(True)  # grid lines are behind the rest

    # plt.legend(legends, labels, fontsize=fontsize-1)
    ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=2, fancybox=True, fontsize=fontsize-3)

    # threshold line
    plt.axhline(y=1.0,linewidth=1, ls='--', color='tab:gray')

    output = 'rasp3b_speedup_pytorch.pdf'
    plt.savefig(output, bbox_inches='tight')

if __name__ == '__main__':
    plot('rasp3b')
