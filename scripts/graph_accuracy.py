import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
import math
import warnings
warnings.simplefilter("ignore")

def plot_accuracy():
    labels = ['FP32', 'A2W2', 'A2W1', 'A1W1']



    df1 = pd.read_csv("data/end2end.csv")
    times =     [float(df1[df1['arch'] == "ARM"][df1['type'] == x][df1['kernel'] == "yes"]['avg-ms']) for x in labels]
    times_nou = [float(df1[df1['arch'] == "ARM"][df1['type'] == x][df1['kernel'] == "no"]['avg-ms']) for x in labels[1:]]
    times_nou = [times[0]] + times_nou
    ips = [1000/x for x in times]
    ips_nou = [1000/x for x in times_nou]

    df2 = pd.read_csv("data/accuracy.csv")
    top1 = [float(df2[df2['type'] == x]['top1']) for x in labels]
    top5 = [float(df2[df2['type'] == x]['top5']) for x in labels]

    xlabels = [re.sub(r"A([0-9])W([0-9])", r"$A^\1W^\2$", x) for x in labels]

    baseline_cmap = sns.color_palette("Blues")
    our_cmap = sns.color_palette("Reds")
    color_fp32 = baseline_cmap[3]
    color_q_u = our_cmap[5]
    color_q_n = our_cmap[2]

    sns.set_context("paper")
    sns.set_style("whitegrid", {"xtick.top": False})
    sns.set_palette("deep")

    fig = plt.figure(figsize=(3.5, 2.16))
    ax = plt.subplot(111)
    legends = []

    b0 = plt.plot(top1, ips_nou, color=color_q_n, marker='s', label="Without $\\mu$kernel")
    b1 = plt.plot(top1, ips, color=color_q_u, marker='o', label="With $\\mu$kernel")
    b2 = plt.plot(top1[0], ips[0], color=color_fp32, marker='s', markersize=6)

    for i in range(1, 4):
        plt.annotate("", xy=(top1[i], ips[i]), xytext=(top1[i], ips_nou[i]),
                     arrowprops=dict(arrowstyle="-|>", color="#444444", ls='--', lw=0.5))
    for i in range(len(labels)):
        plt.text(top1[i] + 0.3, ips[i] + 0.3, xlabels[i], fontsize=8)


    plt.xlabel('Top-1 Accuracy (%)')
    plt.ylabel('Inferences/second')
    plt.legend(loc=(0, 1.02), ncol=2, edgecolor='white')
    # rotate labels since they're kinda long
    # fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')

    plt.xlim(45, 75)
    plt.ylim(2.0, 12.5)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.13, right=0.96, top=0.87, bottom=0.20)

    output = 'accuracy.pdf'
    plt.savefig(output)


if __name__ == "__main__":
    plot_accuracy()
