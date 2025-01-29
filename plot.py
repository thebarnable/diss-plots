import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean, correlation #cosine, cityblock
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes
import numpy as np
import os
import sys
from pathlib import Path
import math
import yaml
import argparse
import random

RED = "#D17171"
YELLOW = "#F3A451"
GREEN = "#7B9965"
DARKGREEN = "#46663C"
BLUE = "#5E7DAF"
DARKBLUE = "#3C5E8A"
DARKRED = "#A84646"
VIOLET = "#886A9B"
GREY = "#636363"
LIGHTGREY = "#c9c5c5"
BLACK = "#000000"
PLOT=True
SCORE=""
OUTPUT="plots"
colors  = [BLUE,YELLOW,RED,GREEN,VIOLET, DARKRED, DARKBLUE, GREY, BLACK]
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']


def lif():
    h     = 1 # ms
    K     = 100 # ms
    v     = np.zeros([K, 1])
    r     = np.zeros([K, 1])
    o_out = np.zeros([K, 1])
    o_in  = np.zeros([K, 1])

    tau   = 10 # ms
    alpha = math.exp(-h/tau)

    o_in[20]  = 1
    o_in[35]  = 1
    o_in[39]  = 1
    o_in[60]  = 1
    o_in[63]  = 1
    o_in[67]  = 1
    v_thresh = 1.0 
    w        = 8.0

    time_axis = list(range(0,K))

    for k in range(K-1):
        v[k+1] = alpha * v[k] + (1-alpha) * w * o_in[k] - alpha * o_out[k] * v[k]
        if v[k+1] > v_thresh:
            o_out[k+1] = 1

    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    fig.subplots_adjust(hspace=0.5)

    ax.plot(time_axis, v, color=BLUE, label="BSNN", linewidth=2.5, linestyle="solid", clip_on=False) # , drawstyle='steps-post'

    ## x axis
    ax.set_xticks([0,K/2,K])
    ax.set_xlim(0, K)
    ax.tick_params(axis='x', length=15, width=2.0, labelsize=15)
    ax.tick_params(axis='x', which='minor', length=5, width=0.5)
    ax.spines['bottom'].set_position(('outward', 15))
    ax.spines['bottom'].set_linewidth(2.0)
    ax.xaxis.set_label_coords(0.0, -0.15)
    ax.set_xlabel("Time [ms]", fontsize=15, fontweight='bold')

    ## y axis
    ax.set_yticks([0, v_thresh, v_thresh*2.])
    ax.set_ylim(0, v_thresh*2.)
    ax.tick_params(axis='y', length=15, width=2.0, labelsize=15)
    ax.tick_params(axis='y', which='minor', length=5, width=0.5)
    ax.spines['left'].set_position(('outward', 15))
    ax.spines['left'].set_linewidth(2.0)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylabel("Potential U(t)", fontsize=15, fontweight='bold')

    ## other axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/lif.pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/lif.svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/lif.png", format='png', dpi=300, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot script')
    parser.add_argument('--function', '-f', default='lif', help='Plot function to call')
    args = parser.parse_args()

    locals()[args.function]()