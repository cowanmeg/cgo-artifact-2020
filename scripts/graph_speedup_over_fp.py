import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import csv
import colorsys
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
from graph_speedup_over_pytorch import calculate_speedup_mean_ci


def plot():
      
    # Data in ms
    target = 'rasp3b'
    layout = 'nhwc'

    a1w1_raw = np.loadtxt('data/raw_bitserial_conv2d_%s_a1w1_%s.csv' % (layout, target))
    a2w1_raw = np.loadtxt('data/raw_bitserial_conv2d_%s_a2w1_%s.csv' % (layout, target),)
    a2w2_raw = np.loadtxt('data/raw_bitserial_conv2d_%s_a2w2_%s.csv' % (layout, target))
    baseline_raw = np.loadtxt('data/raw_tvm_conv2d_fp_%s.csv' % target)

    a1w1, a1w1_ci = calculate_speedup_mean_ci(baseline_raw, a1w1_raw)
    a2w1, a2w1_ci = calculate_speedup_mean_ci(baseline_raw, a2w1_raw)
    a2w2, a2w2_ci = calculate_speedup_mean_ci(baseline_raw, a2w2_raw)

    xlabels = ('C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12')
    labels = [r'$A^1W^1$', r'$A^2W^1$', r'$A^2W^2$']

    index = np.arange(len(a1w1))
    bar_width = 0.2
    gap = (1 - bar_width * len(labels)) / 2
    fontsize = 16

    # Colors, Labels
    sns.set_context("paper")
    sns.set_style("whitegrid", {"xtick.top": False})
    sns.set_palette("deep")

    fig = plt.figure(figsize=(6.5, 4))
    ax = plt.subplot(111)
    legends = []

    err_bar_settings = dict(lw=1, capsize=3, capthick=1)

    b1 = plt.bar(index, a1w1, width=bar_width, align='center',
        yerr=a1w1_ci, error_kw=err_bar_settings)
    b2 = plt.bar(index + bar_width, a2w1, width=bar_width, align='center',
        yerr=a2w1_ci, error_kw=err_bar_settings)
    b3 = plt.bar(index + 2*bar_width, a2w2, width=bar_width, align='center',
        yerr=a2w2_ci, error_kw=err_bar_settings)
    legends.append(b1)
    legends.append(b2)
    legends.append(b3)



    ax.set_ylabel('Speedup over FP32', fontsize=fontsize)
    xticks = (index * 1.01 + 0.21)
    plt.xticks(xticks, xlabels, fontsize=fontsize)
    plt.tick_params(axis='x', which='both', bottom='off', top='off')
    ax.set_axisbelow(True)  # grid lines are behind the rest

    # plt.legend(legends, labels, fontsize=fontsize-1, loc='upper left')
    ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, fontsize=fontsize-3)
    
    # Threshold line 
    plt.axhline(y=1.0,linewidth=1, ls='--', color='tab:gray')

    plt.tight_layout()
    output = 'convolution_speedup_fp.pdf'

    plt.savefig(output, bbox_inches='tight')

if __name__ == '__main__':
    plot()
