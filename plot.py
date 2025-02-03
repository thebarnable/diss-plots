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
import inspect

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
OUTPUT="plots"

colors  = [BLUE,YELLOW,RED,GREEN,VIOLET, DARKRED, DARKBLUE, GREY, BLACK]
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

FIGSIZE = (10,8)
HSPACE = 0.5
LINEWIDTH = 2.5  # of plot
AXISWIDTH = 2.0  # of axes
FONTSIZE = 15

X_MAJORTICKS_LENGTH = 15
X_MINORTICKS_LENGTH = 5
Y_MAJORTICKS_LENGTH = 15
Y_MINORTICKS_LENGTH = 5
X_MAJORTICKS_WIDTH = 2.0
X_MINORTICKS_WIDTH = 0.5
Y_MAJORTICKS_WIDTH = 2.0
Y_MINORTICKS_WIDTH = 0.5
X_MAJORTICKS_LABELSIZE = FONTSIZE
Y_MAJORTICKS_LABELSIZE = FONTSIZE

BOTTOM_POS = ('outward', 15)
BOTTOM_WIDTH = AXISWIDTH

LEFT_POS = ('outward', 15)
LEFT_WIDTH = AXISWIDTH

PNG_DPI = 300


def template():
    # generate random data
    x = np.linspace(0, 2, 100)
    y = np.sin(2 * np.pi * x)
    y[y<0] *= -1

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE) # , sharex=False, gridspec_kw={'height_ratios': [1, 2, 3, 2]}, figsize=(10,7))
    fig.subplots_adjust(hspace=HSPACE)
    if type(ax) is not list:
        ax = [ax]

    ax[0].plot(x, y, color=BLUE, label="Template", linewidth=LINEWIDTH, linestyle="solid", clip_on=False) # , drawstyle='steps-post'

    ## x axis
    ax[0].set_xticks([0, 1, 2])
    ax[0].set_xlim(0, 2)
    ax[0].xaxis.set_label_coords(0.0, -0.11)
    ax[0].set_xlabel("x [unit]", fontsize=FONTSIZE, fontweight='bold')
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    ax[0].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH)
    ax[0].spines['bottom'].set_position(BOTTOM_POS)
    ax[0].spines['bottom'].set_linewidth(BOTTOM_WIDTH)

    ## y axis
    ax[0].set_yticks([0, 0.5, 1])
    ax[0].set_ylim(0, 1)
    ax[0].yaxis.set_label_coords(-0.11, 0.5)
    ax[0].set_ylabel("y [unit]", fontsize=FONTSIZE, fontweight='bold')
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE)
    ax[0].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH)
    ax[0].spines['left'].set_position(LEFT_POS)
    ax[0].spines['left'].set_linewidth(LEFT_WIDTH)

    ## other axes
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


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

    x = list(range(0,K))

    for k in range(K-1):
        v[k+1] = alpha * v[k] + (1-alpha) * w * o_in[k] - alpha * o_out[k] * v[k]
        if v[k+1] > v_thresh:
            o_out[k+1] = 1

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    fig.subplots_adjust(hspace=HSPACE)
    if type(ax) is not list:
        ax = [ax]

    ax[0].plot(x, v, color=BLUE, label="LIF Neuron", linewidth=LINEWIDTH, linestyle="solid", clip_on=False) # , drawstyle='steps-post'

    ## x axis
    ax[0].set_xticks([0, max(x)/2, max(x)])
    ax[0].set_xlim(0, max(x))
    ax[0].xaxis.set_label_coords(0.0, -0.11)
    ax[0].set_xlabel("Time [ms]", fontsize=FONTSIZE, fontweight='bold')
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    ax[0].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH)
    ax[0].spines['bottom'].set_position(BOTTOM_POS)
    ax[0].spines['bottom'].set_linewidth(BOTTOM_WIDTH)

    ## y axis
    ax[0].set_yticks([0, v_thresh, v_thresh*2.])
    ax[0].set_ylim(0, v_thresh*2.)
    ax[0].yaxis.set_label_coords(-0.11, 0.5)
    ax[0].set_ylabel("Membrane Voltage u(t)", fontsize=FONTSIZE, fontweight='bold')
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE)
    ax[0].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH)
    ax[0].spines['left'].set_position(LEFT_POS)
    ax[0].spines['left'].set_linewidth(LEFT_WIDTH)

    ## other axes
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
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