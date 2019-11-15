import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
import math
import warnings
warnings.simplefilter("ignore")
from graph_speedup_over_pytorch import confidence_interval

SAMPLES = 10


def plot_end2end_facets():
    df = pd.read_csv("data/end2end.csv")

    xlabels = ['FP32', 'A2W2', 'A2W1', 'A1W1']
    times = [float(df[df['arch'] == "ARM"][df['type'] == x][df['kernel'] == "yes"]['avg-ms']) for x in xlabels]
    times_nou = [float(df[df['arch'] == "ARM"][df['type'] == x][df['kernel'] == "no"]['avg-ms']) for x in xlabels[1:]]

    stddev = [float(df[df['arch'] == "ARM"][df['type'] == x][df['kernel'] == "yes"]['std-dev-ms']) for x in xlabels]
    stddev_nou = [float(df[df['arch'] == "ARM"][df['type'] == x][df['kernel'] == "no"]['std-dev-ms']) for x in xlabels[1:]]
    ci = confidence_interval(stddev)
    ci_nou = confidence_interval(stddev_nou)

    index = list(np.arange(len(xlabels)))
    bar_width = 0.4

    baseline_cmap = sns.color_palette("Blues")
    our_cmap = sns.color_palette("Reds")
    color_fp32 = baseline_cmap[3]
    color_q_u = our_cmap[5]
    color_q_n = our_cmap[2]

    index.reverse()
    index = np.array(index)

    xlabels = [re.sub(r"A([0-9])W([0-9])", r"$A^\1W^\2$", x) for x in xlabels]

    sns.set_context("paper")
    sns.set_style("whitegrid", {"xtick.top": False})
    sns.set_palette("deep")

    fig = plt.figure(figsize=(3.5, 2.16))
    ax = plt.subplot(111)
    legends = []

    # fp32
    b0 = plt.barh(index[0], times[0], color=color_fp32, height=bar_width, align='center',
        xerr=ci[0], error_kw=dict(lw=1, capsize=3, capthick=0.5))
    # no ukernel
    b1 = plt.barh(index[1:] + bar_width/2, times_nou, color=color_q_n, height=bar_width, align='center',
        xerr=ci_nou, error_kw=dict(lw=1, capsize=3, capthick=0.5))
    # with ukernel
    b2 = plt.barh(index[1:] - bar_width/2, times[1:], color=color_q_u, height=bar_width, align='center',
        xerr=ci[1:], error_kw=dict(lw=1, capsize=3, capthick=0.5))

    xticks = index
    plt.yticks(xticks, xlabels)
    plt.tick_params(axis='y', which='both', left=True, right=False)
    plt.xlabel('Inference time (ms)')
    plt.legend([b1, b2], ["Without $\\mu$kernel", "With $\\mu$kernel"], loc=(0, 1.02), ncol=2, edgecolor='white')
    # rotate labels since they're kinda long
    # fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')

    # plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.20)

    output = 'end2end_ARM.pdf'
    plt.savefig(output)


if __name__ == "__main__":
    plot_end2end_facets()
