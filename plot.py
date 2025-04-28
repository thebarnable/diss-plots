import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
from matplotlib.patches import FancyArrow
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean, correlation #cosine, cityblock
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import os
import sys
from pathlib import Path
import math
import yaml
import argparse
import random
import inspect
from tqdm import tqdm

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

GREENS = ["#90998A", "#8A9982", "#849979", "#7C996D"]
YELLOWS = ["#f2b879", "#f2cfaa", "#f2c391", "#f2b879"]
BLUES = ["#9ea5b0", "#8d9ab0", "#7b8fb0", "#6a85b0"]
REDS = ["#d1bcbc", "#d1a7a7", "#d19292", "#d17d7d"]
VIOLETS = ["#9a7c9f", "#a99fb1"]

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
    if type(ax) is not list and type(ax) is not np.ndarray:
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

    plt.tight_layout()
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def bnns_scaling():
    # params
    FONTSIZE = 12
    Y_MAJORTICKS_LABELSIZE = 12
    X_MAJORTICKS_LABELSIZE = 12
    MARKERSIZE = 8
    AXISWIDTH = 1.0
    FIGWIDTH = 5.6 # 6.3
    X_MAJORTICKS_LENGTH = 10
    Y_MAJORTICKS_LENGTH = 10
    X_MAJORTICKS_WIDTH = 1.0
    Y_MAJORTICKS_WIDTH = 1.0
    FIGSIZE = (FIGWIDTH, FIGWIDTH * (9/16))

    # experiment data (from https://www.overleaf.com/project/5ef59a9907971a00016a551d (BNN paper project))
    x = np.array([2, 4, 8, 16, 32, 64])

    y_8b = np.array([0.0222, 0.0456, 0.1152, 0.2408, 0.5070, 1])
    y_1b = np.array([0.0100, 0.0400, 0.1184, 0.2378, 0.4662, 1])
    y_linear = np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1])

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE) # , sharex=False, gridspec_kw={'height_ratios': [1, 2, 3, 2]}, figsize=(10,7))
    #fig.subplots_adjust(hspace=HSPACE)
    if type(ax) is not list and type(ax) is not np.ndarray:
        ax = [ax]

    ax[0].plot(x, y_8b, 'x-', color=BLUE, label="8b x 8b", linewidth=LINEWIDTH, clip_on=False, markersize=MARKERSIZE) # , drawstyle='steps-post'
    ax[0].plot(x, y_1b, 'o-', color=RED, label="1b x 1b", linewidth=LINEWIDTH, clip_on=False, markersize=MARKERSIZE) # , drawstyle='steps-post'
    ax[0].plot(x, y_linear, ':', color=BLACK, label="Linear", linewidth=LINEWIDTH, clip_on=False, markersize=MARKERSIZE) # , drawstyle='steps-post'

    ## x axis
    ax[0].set_xscale('log', base=2)
    ax[0].set_xticks([4, 16, 64], ["4", "16", "64"])
    ax[0].set_xlim(1.5, 80)
    #ax[0].xaxis.set_label_coords(0.0, -0.11)
    ax[0].set_xlabel("Vector Length (N)", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    #ax[0].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['bottom'].set_linewidth(AXISWIDTH)

    ## y axis
    ax[0].set_yticks([0.01, 0.1, 1])
    ax[0].set_ylim(0.005, 2)
    #ax[0].yaxis.set_label_coords(-0.11, 0.0)
    ax[0].set_ylabel("Normalized Energy Cost", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    ax[0].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].set_yscale('log')
    ax[0].spines['left'].set_linewidth(AXISWIDTH)

    ## other axes
    ax[0].spines['top'].set_linewidth(AXISWIDTH)
    ax[0].spines['right'].set_linewidth(AXISWIDTH)

    ## grid
    ax[0].grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)

    ## legend
    ax[0].legend(frameon=True, loc='lower right',bbox_to_anchor=(1.0, 0.0),fontsize=FONTSIZE)

    plt.tight_layout()
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def bnns_results():
    # params
    FONTSIZE = 12
    Y_MAJORTICKS_LABELSIZE = 12
    X_MAJORTICKS_LABELSIZE = 12
    MARKERSIZE = 8
    FIGWIDTH = 6.3
    AXISWIDTH = 1.0
    X_MAJORTICKS_LENGTH = 10
    Y_MAJORTICKS_LENGTH = 10
    X_MAJORTICKS_WIDTH = 1.0
    Y_MAJORTICKS_WIDTH = 1.0
    FIGSIZE = (FIGWIDTH, FIGWIDTH * (9/16)) # (9/16)

    # experiment data (from https://www.overleaf.com/project/5ef59a9907971a00016a551d (BNN paper project))
    cifar_1 = np.array([[0.917, 89.98], [0.690, 88.15], [0.917, 90.09]])
    cifar_2 = np.array([[2.633, 90.52], [10.281, 91.91]])
    cifar_3 = np.array([[0.883, 89.09], [0.896, 89.44], [1.433, 90.19], [1.551, 90.52],
                       [0.981, 89.44], [0.703, 89.38], [0.690, 88.15], [0.703, 90.09]])
    cifar_4 = np.array([[0.690, 88.60]])
    cifar_5 = np.array([[1.661, 90.85]])
    cifar_6 = np.array([[1.366, 84.0]])
    cifar_7 = np.array([[9.448, 90.1]])
    cifar_8 = np.array([[0.917, 89.83]])
    cifar_9 = np.array([[46.716, 92.97]])

    mnist_1 = np.array([[0.064, 99.07], [0.019, 99.22], [0.064, 99.27]])
    mnist_2 = np.array([[0.050, 99.28], [0.151, 99.31], [0.505, 99.32], [0.019, 99.26]])
    mnist_3 = np.array([[0.026, 98.59], [0.021, 99.27], [0.027, 99.14], [0.031, 99.20], [0.020, 98.90], [0.021, 99.08], [0.019, 99.18], [0.021, 99.16]])
    mnist_4 = np.array([[0.048, 98.6], [0.117, 99.04]])
    mnist_5 = np.array([[0.073, 99.38]])
    mnist_6 = np.array([[0.234, 98.75]])
    mnist_7 = np.array([[0.133, 98.82]])
    mnist_8 = np.array([[0.597, 99.4]])

    # plot results
    fig, ax = plt.subplots(1, 2, figsize=FIGSIZE, sharex=False, gridspec_kw={'width_ratios': [1, 1]}) #, figsize=(10,7))
    fig.subplots_adjust(top=0.8, bottom=0.15)
    if type(ax) is not list and type(ax) is not np.ndarray:
        ax = [ax]
    
    # mnist
    ax[0].scatter(mnist_1[:, 0], mnist_1[:, 1], color=BLUE, marker='x', s=50, label="First/Last (ours)")
    ax[0].scatter(mnist_2[:, 0], mnist_2[:, 1], color=BLUE, marker='*', s=50, label="Arch. (ours)")
    ax[0].scatter(mnist_3[:, 0], mnist_3[:, 1], color=BLUE, marker='^', s=50, label="Tern. (ours)")
    ax[0].scatter(mnist_4[:, 0], mnist_4[:, 1], color=GREY, marker='s', s=50, label="BNN")
    ax[0].scatter(mnist_5[:, 0], mnist_5[:, 1], color=GREY, marker='+', s=50, label="TBN")
    #ax[0].scatter(mnist_6[:, 0], mnist_6[:, 1], color=GREY, marker='|', s=50, label="HORQ")
    ax[0].scatter(mnist_7[:, 0], mnist_7[:, 1], color=GREY, marker='h', s=50, label="BC")
    ax[0].scatter(mnist_8[:, 0], mnist_8[:, 1], color=GREY, marker='o', s=50, label="Full-Prec.", facecolors='none')

    ax[0].set_xscale('log', base=10)
    ax[0].set_xticks([0.01, 0.1, 1.0], ["0.01", "0.1", "1.0"])
    ax[0].set_xlim(0.01, 1)
    #ax[0].xaxis.set_label_coords(0.0, -0.11)
    ax[0].set_xlabel("Energy Cost [μJ]", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    ax[0].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['bottom'].set_linewidth(AXISWIDTH)

    ## y axis
    ax[0].set_yticks([98.7, 98.9, 99.1, 99.3, 99.5])
    ax[0].set_ylim(98.7, 99.5)
    #ax[0].yaxis.set_label_coords(-0.11, 0.0)
    ax[0].set_ylabel("Accuracy [%]", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    #ax[0].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['left'].set_linewidth(AXISWIDTH)

    ## other axes
    ax[0].spines['top'].set_linewidth(AXISWIDTH)
    ax[0].spines['right'].set_linewidth(AXISWIDTH)

    ## grid
    ax[0].grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)
    ax[0].vlines(x=[0.021], ymin=98.7, ymax=99.5, colors=GREY, ls='--', lw=2, clip_on=False)
    ax[0].annotate(
        '',                       # Empty string for no text
        xy=(0.025, mnist_5[0][1]),        # Coordinates of the end point
        xytext=(mnist_5[0][0]-0.01, mnist_5[0][1]), # Coordinates of the start point
        arrowprops=dict(
            facecolor=GREY,     # Arrow color
            edgecolor=GREY,     # Border color of the arrow
            arrowstyle='->',       # Style of the arrow (e.g., '->', '-|>', etc.)
            lw=1.5,                # Line width
        ),
        fontsize=FONTSIZE
    )
    ax[0].text(
        (mnist_5[0][0]-0.01 + 0.025) / 2,   # X coordinate for the text (middle of the arrow)
        mnist_5[0][1]+0.01,    # Y coordinate for the text (middle of the arrow)
        'x3.5',                   # Text
        ha='center',              # Horizontal alignment
        va='bottom',              # Vertical alignment
        fontsize=FONTSIZE,        # Font size
        color='black'             # Text color
    )

    # cifar
    ax[1].scatter(cifar_1[:, 0], cifar_1[:, 1], color=BLUE, marker='x', s=50, label="First/Last (ours)")
    ax[1].scatter(cifar_2[:, 0], cifar_2[:, 1], color=BLUE, marker='*', s=50, label="Arch. (ours)")
    ax[1].scatter(cifar_3[:, 0], cifar_3[:, 1], color=BLUE, marker='^', s=50, label="Tern. (ours)")
    ax[1].scatter(cifar_4[:, 0], cifar_4[:, 1], color=GREY, marker='s', s=50, label="BNN")
    ax[1].scatter(cifar_5[:, 0], cifar_5[:, 1], color=GREY, marker='+', s=50, label="TBN")
    #ax[1].scatter(cifar_6[:, 0], cifar_6[:, 1], color=BLACK, marker='|', s=50, label="HORQ")
    ax[1].scatter(cifar_7[:, 0], cifar_7[:, 1], color=GREY, marker='h', s=50, label="BC")
    ax[1].scatter(cifar_8[:, 0], cifar_8[:, 1], color=GREY, marker='v', s=50, label="XNOR-NET")
    ax[1].scatter(cifar_9[:, 0], cifar_9[:, 1], color=GREY, marker='o', s=50, label="Full-Prec.", facecolors='none')

    ## x axis
    ax[1].set_xscale('log', base=10)
    ax[1].set_xticks([1, 10], ["1", "10"])
    ax[1].set_xlim(0.5, 70)
    # ax[1].xaxis.set_label_coords(0.0, -0.11)
    ax[1].set_xlabel("Energy Cost [μJ]", fontsize=FONTSIZE) # , fontweight='bold'
    # ax[1].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[1].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    ax[1].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[1].spines['bottom'].set_linewidth(AXISWIDTH)

    ## y axis
    ax[1].set_yticks([87, 89, 91, 93, 95])
    ax[1].set_ylim(87, 95)
    #ax[1].yaxis.set_label_coords(-0.11, 0.0)
    #ax[1].set_ylabel("Normalized Energy Cost", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[1].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    #ax[1].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[1].spines['left'].set_linewidth(AXISWIDTH)

    ## other axes
    ax[1].spines['top'].set_linewidth(AXISWIDTH)
    ax[1].spines['right'].set_linewidth(AXISWIDTH)

    ## grid
    ax[1].grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)

    ## legend
    plt.legend(ncol=4, frameon=True, loc='upper center', bbox_to_anchor=(-0.25, 1.32), fontsize=FONTSIZE, borderpad=0.2, columnspacing=0.1)

    #plt.tight_layout()
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def bnns_sota():
    # params
    FONTSIZE = 12
    Y_MAJORTICKS_LABELSIZE = 12
    X_MAJORTICKS_LABELSIZE = 12
    MARKERSIZE = 8
    FIGWIDTH = 6.3
    AXISWIDTH = 1.0
    X_MAJORTICKS_LENGTH = 10
    Y_MAJORTICKS_LENGTH = 10
    X_MAJORTICKS_WIDTH = 1.0
    Y_MAJORTICKS_WIDTH = 1.0
    FIGSIZE = (FIGWIDTH, FIGWIDTH / 1.6) # (9/16)
    ARROWPROPS = dict(
        facecolor=GREY,     # Arrow color
        edgecolor=GREY,     # Border color of the arrow
        arrowstyle='-|>',       # Style of the arrow (e.g., '->', '-|>', etc.)
        lw=1,                # Line width
    )

    # experiment data (from https://www.overleaf.com/project/5ef59a9907971a00016a551d (BNN paper project))
    data_1 = np.array([[2.594, 41.8], [40.279, 47.1]])  # BNN: (2.594, 41.8)  (40.279, 47.1)
    data_1b = np.array([[10.12, 41.8], [131.91, 47.1]])  # -> data_1[0], data_1[1]
    data_2 = np.array([[10.373, 44.2], [15.877, 51.2]])  # XNOR-Net:  (10.373, 44.2) (15.877, 51.2)
    data_3 = np.array([[4.25, 42.7], [63.74, 65.0], [6.547, 52.4], [121.15,68.4]])  # ABC-Net: (4.25, 42.7) (63.74, 65.0) (6.547, 52.4) (121.15,68.4)
    data_3b = np.array([[13.2, 42.7], [114.61, 65.0], [15.83, 52.4], [180.28,68.4]])  # -> data_3[*]
    data_4 = np.array([[10.003, 39.3], [10.771, 46.6]])  # Tang: (10.003, 39.3) (10.771, 46.6)
    data_5 = np.array([[18.78, 56.4], [21.09, 62.2]])  # Bireal-Net: (18.78, 56.4) (21.09, 62.2)
    data_6 = np.array([[10.373, 46.1], [15.877, 53.0]])   # BNN+: (10.373, 46.1) (15.877, 53.0)
    data_7 = np.array([[21.46, 67.5], [32.941, 71.8]])   # GroupNet: (21.46, 67.5) (32.941, 71.8)
    data_8 = np.array([[15.877, 58.1]])  # ResNetE: (15.877, 58.1)
    data_9 = np.array([[11.271, 49.7], [18.32, 55.6], [23.308, 58.2]]) # TBN: (11.271, 49.7) (18.32, 55.6) (23.308, 58.2)
    data_9b = np.array([[16.18, 55.6], [18.80, 58.2]]) # -> data_9[1], data_9[2]
    data_10 = np.array([[31.12, 50.2], [47.63, 53.6]]) # BENN: (31.12, 50.2) (47.63, 53.6)
    data_11 = np.array([[11.4, 66.6]]) # TTQ: (11.4, 66.6)
    data_12 = np.array([[12.968, 35.4], [20.679, 56.8]]) # BC: (12.968, 35.4) (20.679, 56.8)
    data_12b = np.array([[65.70, 35.4], [65.70, 56.8]]) # -> data_12[*]
    data_13 = np.array([[65.697, 56.6], [166.30, 69.3], [334.52,73.3], [52.25, 70.6], [27.26, 72], [35.44, 77.3], [38.17, 67.4]])  # FP: (65.697, 56.6) (166.30, 69.3) (334.52,73.3) (52.25, 70.6) (27.26, 72) (35.44, 77.3) (38.17, 67.4)

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE) #, figsize=(10,7))
    fig.subplots_adjust(top=0.8, bottom=0.15)
    if type(ax) is not list and type(ax) is not np.ndarray:
        ax = [ax]

    # pareto curve and background refs
    ax[0].hlines(y=[data_1[0][1]], xmin=1, xmax=data_3[0][0], colors=GREY, ls='--', lw=1, clip_on=False)  # data_1[0] -> data_3[0]
    ax[0].vlines(x=[data_3[0][0]], ymin=data_1[0][1], ymax=data_3[0][1], colors=GREY, ls='--', lw=1, clip_on=False)
    ax[0].hlines(y=[data_3[0][1]], xmin=data_3[0][0], xmax=data_3[2][0], colors=GREY, ls='--', lw=1, clip_on=False)  # data_3[0] -> data_3[2]
    ax[0].vlines(x=[data_3[2][0]], ymin=data_3[0][1], ymax=data_3[2][1], colors=GREY, ls='--', lw=1, clip_on=False)
    ax[0].hlines(y=[data_3[2][1]], xmin=data_3[2][0], xmax=data_11[0][0], colors=GREY, ls='--', lw=1, clip_on=False)  # data_3[2] -> data_11[0]
    ax[0].vlines(x=[data_11[0][0]], ymin=data_3[2][1], ymax=data_11[0][1], colors=GREY, ls='--', lw=1, clip_on=False)
    ax[0].hlines(y=[data_11[0][1]], xmin=data_11[0][0], xmax=data_7[0][0], colors=GREY, ls='--', lw=1, clip_on=False)  # data_11[0] -> data_7[0]
    ax[0].vlines(x=[data_7[0][0]], ymin=data_11[0][1], ymax=data_7[0][1], colors=GREY, ls='--', lw=1, clip_on=False)
    ax[0].hlines(y=[data_7[0][1]], xmin=data_7[0][0], xmax=data_13[4][0], colors=GREY, ls='--', lw=1, clip_on=False)  # data_7[0] -> data_13[4]
    ax[0].vlines(x=[data_13[4][0]], ymin=data_7[0][1], ymax=data_13[4][1], colors=GREY, ls='--', lw=1, clip_on=False)
    ax[0].hlines(y=[data_13[4][1]], xmin=data_13[4][0], xmax=data_13[5][0], colors=GREY, ls='--', lw=1, clip_on=False)  # data_13[4] -> data_13[5]
    ax[0].vlines(x=[data_13[5][0]], ymin=data_13[4][1], ymax=data_13[5][1], colors=GREY, ls='--', lw=1, clip_on=False)
    ax[0].hlines(y=[data_13[5][1]], xmin=data_13[5][0], xmax=500, colors=GREY, ls='--', lw=1, clip_on=False)  # data_13[5] -> end
    
    ax[0].annotate('', xytext=(data_1b[0][0]*0.9, data_1b[0][1]), xy=(data_1[0][0]*1.1, data_1[0][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    ax[0].annotate('', xytext=(data_1b[1][0]*0.9, data_1b[1][1]), xy=(data_1[1][0]*1.1, data_1[1][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    ax[0].annotate('', xytext=(data_3b[0][0]*0.9, data_3b[0][1]), xy=(data_3[0][0]*1.1, data_3[0][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    ax[0].annotate('', xytext=(data_3b[1][0]*0.9, data_3b[1][1]), xy=(data_3[1][0]*1.1, data_3[1][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    ax[0].annotate('', xytext=(data_3b[2][0]*0.9, data_3b[2][1]), xy=(data_3[2][0]*1.1, data_3[2][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    ax[0].annotate('', xytext=(data_3b[3][0]*0.9, data_3b[3][1]), xy=(data_3[3][0]*1.1, data_3[3][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    #ax[0].annotate('', xytext=(data_9b[0][0]*0.9, data_9b[0][1]), xy=(data_9[1][0]*1.1, data_9[1][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    #ax[0].annotate('', xytext=(data_9b[1][0]*0.9, data_9b[1][1]), xy=(data_9[2][0]*1.1, data_9[2][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    ax[0].annotate('', xytext=(data_12b[0][0]*0.9, data_12b[0][1]), xy=(data_12[0][0]*1.1, data_12[0][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    ax[0].annotate('', xytext=(data_12b[1][0]*0.9, data_12b[1][1]), xy=(data_12[1][0]*1.1, data_12[1][1]), arrowprops=ARROWPROPS, fontsize=FONTSIZE)
    
    for i,fp_data in enumerate(data_13):
        x_offset = 0
        y_offset = 0.8
        if i==5:
            x_offset = -5
            y_offset = 0

        ax[0].text(
            fp_data[0]+x_offset,         # x
            fp_data[1]+y_offset,    # y
            f"({i+1})",         # Text
            ha='center',        # Horizontal alignment
            va='bottom',        # Vertical alignment
            fontsize=10,  # Font size
            color=GREY       # Text color
        )

    # mnist
    ax[0].scatter(data_1[:, 0], data_1[:, 1], color=BLUE, marker='x', s=50, label="BNN")
    ax[0].scatter(data_2[:, 0], data_2[:, 1], color=BLUE, marker='*', s=50, label="XNOR")
    ax[0].scatter(data_3[:, 0], data_3[:, 1], color=BLUE, marker='^', s=50, label="ABC-Net")
    ax[0].scatter(data_4[:, 0], data_4[:, 1], color=BLUE, marker='s', s=50, label="Tang")
    ax[0].scatter(data_5[:, 0], data_5[:, 1], color=BLUE, marker='+', s=50, label="Bireal")
    ax[0].scatter(data_6[:, 0], data_6[:, 1], color=BLUE, marker='|', s=50, label="BNN+")
    ax[0].scatter(data_7[:, 0], data_7[:, 1], color=BLUE, marker='h', s=50, label="GroupNet")
    ax[0].scatter(data_8[:, 0], data_8[:, 1], color=BLUE, marker='.', s=50, label="ResNetE")
    ax[0].scatter(data_9[:, 0], data_9[:, 1], color=BLUE, marker='p', s=50, label="TBN")
    ax[0].scatter(data_10[:, 0], data_10[:, 1], color=BLUE, marker='d', s=50, label="BENN")
    ax[0].scatter(data_11[:, 0], data_11[:, 1], color=BLUE, marker='_', s=50, label="TTQ")
    ax[0].scatter(data_12[:, 0], data_12[:, 1], color=BLUE, marker='D', s=50, label="BC")
    ax[0].scatter(data_13[:, 0], data_13[:, 1], color=GREY, marker='o', s=50, facecolors='none')

    ax[0].scatter(data_1b[:, 0], data_1b[:, 1], color=GREY, marker='o', s=15)
    ax[0].scatter(data_3b[:, 0], data_3b[:, 1], color=GREY, marker='o', s=15)
    #ax[0].scatter(data_9b[:, 0], data_9b[:, 1], color=GREY, marker='o', s=15)
    ax[0].scatter(data_12b[:, 0], data_12b[:, 1], color=GREY, marker='o', s=15)


    ax[0].set_xscale('log', base=10)
    ax[0].set_xticks([10, 100], ["10", "100"])
    ax[0].set_xlim(1, 500)
    #ax[0].xaxis.set_label_coords(0.0, -0.11)
    ax[0].set_xlabel("Energy Cost [μJ]", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    ax[0].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['bottom'].set_linewidth(AXISWIDTH)

    ## y axis
    ax[0].set_yticks([40, 50, 60, 70, 80])
    ax[0].set_ylim(30, 80)
    #ax[0].yaxis.set_label_coords(-0.11, 0.0)
    ax[0].set_ylabel("Top-1 Accuracy [%]", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    #ax[0].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['left'].set_linewidth(AXISWIDTH)

    ## other axes
    ax[0].spines['top'].set_linewidth(AXISWIDTH)
    ax[0].spines['right'].set_linewidth(AXISWIDTH)

    ## grid
    ax[0].grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)
    # ax[0].text(
    #     (mnist_5[0][0]-0.01 + 0.025) / 2,   # X coordinate for the text (middle of the arrow)
    #     mnist_5[0][1]+0.01,    # Y coordinate for the text (middle of the arrow)
    #     'x3.5',                   # Text
    #     ha='center',              # Horizontal alignment
    #     va='bottom',              # Vertical alignment
    #     fontsize=FONTSIZE,        # Font size
    #     color='black'             # Text color
    # )

    ## legend
    plt.legend(ncol=4, frameon=True, loc='upper center', bbox_to_anchor=(0.5, 1.327), fontsize=FONTSIZE, borderpad=0.1, columnspacing=0.01) #, borderpad=0.1, columnspacing=0.01, labelspacing=0.01)

    #plt.tight_layout()
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def sg():
    # override variables
    LINEWIDTH = 8.5
    X_MAJORTICKS_WIDTH = 4.5
    X_MAJORTICKS_LENGTH = 20
    BOTTOM_WIDTH = 4.5
    FIGSIZE=(8,5)

    # generate random data
    x = np.array([np.linspace(-2, 0, 50), np.linspace(0, 2, 50)]).flatten()
    y = np.array([np.zeros((50,)), 0.5*np.ones((50,))]).flatten()

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE) # , sharex=False, gridspec_kw={'height_ratios': [1, 2, 3, 2]}, figsize=(10,7))
    #fig.subplots_adjust(hspace=HSPACE)
    if type(ax) is not list and type(ax) is not np.ndarray:
        ax = [ax]

    ax[0].plot(x, y, color=BLACK, label="Template", linewidth=LINEWIDTH, linestyle="solid", clip_on=False) # , drawstyle='steps-post'

    ## y axis
    ax[0].set_yticks([0, 0.6], ["", ""])
    ax[0].set_ylim(0,0.6)
    ax[0].yaxis.set_label_coords(*(0.0, 0.0))
    #ax[0].set_ylabel("u", fontsize=FONTSIZE, fontweight='bold')
    #ax[4].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='y', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    #ax[4].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH)
    ax[0].spines['left'].set_position(BOTTOM_POS)
    ax[0].spines['left'].set_linewidth(BOTTOM_WIDTH)
    ## x axis
    ax[0].set_xticks([-2, 0, 2], ["", "", ""])
    ax[0].set_xlim(-2, 2)
    ax[0].xaxis.set_label_coords(*(0.0, -0.2))
    ax[0].set_xlabel("u", fontsize=FONTSIZE, fontweight='bold')
    #ax[4].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    #ax[4].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH)
    ax[0].spines['bottom'].set_position(BOTTOM_POS)
    ax[0].spines['bottom'].set_linewidth(BOTTOM_WIDTH)
    ## other axes
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+"_H.pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+"_H.svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+"_H.png", format='png', dpi=PNG_DPI, transparent=True)
    # if PLOT:
    #     plt.show()
    #     plt.clf()
    plt.clf()
    plt.close()

    # override variables
    LINEWIDTH = 4.5
    X_MAJORTICKS_WIDTH = 2.0
    X_MAJORTICKS_LENGTH = 15
    Y_MAJORTICKS_WIDTH = 2.0
    Y_MAJORTICKS_LENGTH = 15
    BOTTOM_WIDTH = 2.0

    # generate SGs
    x_rect = np.array([np.linspace(-1, -0.5, 50), 
                  np.linspace(-0.5, 0, 50),
                  np.linspace(0, 0.5, 50), 
                  np.linspace(0.5, 1, 50)]).flatten()
    y_rect = np.concatenate([np.zeros(50), 
                        0.5*np.ones(100), 
                        np.zeros(50)])
    
    tanh_sg = lambda x: 1/np.pi * 1/(1+np.pow(np.pi*x,2))
    x_tanh = np.linspace(-1, 1, 200)
    y_tanh = np.array([tanh_sg(x) for x in x_tanh])

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=(8,7)) # , sharex=False, gridspec_kw={'height_ratios': [1, 2, 3, 2]}, figsize=(10,7))
    #fig.subplots_adjust(hspace=HSPACE)
    if type(ax) is not list and type(ax) is not np.ndarray:
        ax = [ax]

    l1, = ax[0].plot(x_tanh, y_tanh, color=BLUE, label="arctan", linewidth=LINEWIDTH, linestyle="solid", clip_on=False) # , drawstyle='steps-post'
    l2, = ax[0].plot(x_rect, y_rect, color=BLUE, label="rect", linewidth=LINEWIDTH, linestyle="dotted", clip_on=False) # , drawstyle='steps-post'
    ax[0].arrow(0, 0, 0, 0.7, head_width=0.05, head_length=0.1, fc='black', ec='black', label="rect")

    l3 = Line2D([0], [0], color=BLACK, marker=">", linestyle="None", markersize=10)

    ## y axis
    ax[0].set_yticks([])
    ax[0].set_ylim(0, 2)
    ## x axis
    ax[0].set_xticks([-1, 0, 1], ["", "", ""])
    ax[0].set_xlim(-1, 1)
    #ax[0].xaxis.set_label_coords(0.0, -0.11)
    #ax[0].set_xlabel("x [unit]", fontsize=FONTSIZE, fontweight='bold')
    #ax[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    #ax[0].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH)
    ax[0].spines['bottom'].set_position(BOTTOM_POS)
    ax[0].spines['bottom'].set_linewidth(BOTTOM_WIDTH)   
    ## other axes
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[0].legend(handles=[l1, l2, l3], loc='upper right', bbox_to_anchor=(1.1, 0.4),fontsize=20, labels=["arctan", "rect", "H'"]) # bbox_to_anchor=(1.0, 1.25),

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def stdp():
    # params
    FONTSIZE = 12
    Y_MAJORTICKS_LABELSIZE = 12
    X_MAJORTICKS_LABELSIZE = 12
    MARKERSIZE = 50
    AXISWIDTH = 1.0
    FIGWIDTH = 5.6 # 6.3
    X_MAJORTICKS_LENGTH = 10
    Y_MAJORTICKS_LENGTH = 10
    X_MAJORTICKS_WIDTH = 1.0
    Y_MAJORTICKS_WIDTH = 1.0
    BOTTOM_WIDTH = AXISWIDTH
    BOTTOM_POS = ('outward', 7)
    FIGSIZE = (FIGWIDTH, FIGWIDTH * (2/3))

    # generate random data
    # Parameters
    h=0.001 # timestep width
    t_sim=100 # timesteps
    lambda_j = 100 # presyn decay rate
    lambda_i = 100 # postsyn decay rate
    spike_train_j = [1,5,10,21]
    spike_train_i = [11,18,40]
    w_init = 0.0 # synaptic weight
    w_max = 5.0
    mu_ltd = 1.0
    mu_ltp = 1.0
    lr = 0.01 # learning rate -> scales weight update (nest: lambda_)
    lr_ltd_scale = 1e5 # learning rate scale for LTD update (nest: alpha_)
    d = 0.0 # synaptic delay

    # Initialize spike trains
    spikes_j = np.zeros(t_sim)
    spikes_j[spike_train_j] = 1
    spikes_i = np.zeros(t_sim)
    spikes_i[spike_train_i] = 1

    # Exact integration
    kj, ki = [0], [0]
    tj_axis, ti_axis = [0], [0]
    ki_event, kj_event = np.zeros(t_sim),np.zeros(t_sim)
    w = np.zeros(t_sim)
    last_spiketime_j, last_spiketime_i = 0, 0
    for t in tqdm(range(1,t_sim-1), desc="# Time-driven exact integration"):
        # integration of spike traces for reference
        tj_axis.append(t)
        ti_axis.append(t)
        kj.append(np.exp(-h*lambda_j)*kj[-1])
        ki.append(np.exp(-h*lambda_i)*ki[-1])
        if spikes_j[t]:
            tj_axis.append(t)
            kj.append(kj[-1] + ((1-np.exp(-h*lambda_j))/lambda_j))
        if spikes_i[t]:
            ti_axis.append(t)
            ki.append(ki[-1] + ((1-np.exp(-h*lambda_i))/lambda_i))

        # event-based update of spike traces
        if spikes_j[t]==1:
            if last_spiketime_j==0:
                kj_event[t] = (1-np.exp(-h*lambda_j))/lambda_j
            else:
                kj_event[t] = np.exp(-h*(t - last_spiketime_j)*lambda_j)*kj_event[last_spiketime_j] + (1-np.exp(-h*lambda_j))/lambda_j
            last_spiketime_j = t

        if spikes_i[t]==1:
            if last_spiketime_i==0:
                ki_event[t] = (1-np.exp(-h*lambda_i))/lambda_i
            else:
                ki_event[t] = np.exp(-h*(t - last_spiketime_i)*lambda_i)*ki_event[last_spiketime_i] + (1-np.exp(-h*lambda_i))/lambda_i
            last_spiketime_i = t

        # stdp
        w[t] = w[t-1]
        ## ltd
        if spikes_j[t]==1:
            # get current ki
            ki_t = np.exp(-h*(t-last_spiketime_i)*lambda_i)*ki_event[last_spiketime_i]
            w_old = w[t]
            w[t] = (w[t]/w_max) - lr_ltd_scale*lr*np.pow(w[t]/w_max,mu_ltd)*ki_t
            w[t] = w[t]*w_max if w[t]>0.0 else 0.0
            print(f"LTD @ t={t}: dw = {w[t]-w_old}")
        ## ltp
        if spikes_i[t]==1:
            # get current kj
            kj_t = np.exp(-h*(t-last_spiketime_j)*lambda_j)*kj_event[last_spiketime_j]
            w_old = w[t]
            w[t] = (w[t]/w_max) + lr*np.pow(1-w[t]/w_max,mu_ltp)*kj_t
            w[t] = w[t]*w_max if w[t]<1.0 else w_max
            print(f"LTP @ t={t}: dw = {w[t]-w_old}")

    # plot results
    X_AXIS_COORDS = (0.0, -0.35)
    Y_AXIS_COORDS = (-0.035, 0.5)

    fig, ax = plt.subplots(5, 1, sharex=True, gridspec_kw={'height_ratios': [0.5, 1, 0.5, 1, 1]}, figsize=FIGSIZE)
    #plt.rcParams['text.usetex'] = True
    fig.subplots_adjust(hspace=HSPACE)
    if type(ax) is not list and type(ax) is not np.ndarray:
        ax = [ax]

    # j spikes
    ax[0].scatter(np.where(spikes_j > 0)[0], spikes_j[spikes_j > 0], label="j spikes", s=MARKERSIZE, marker='.', color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False)
    ## y axis
    ax[0].set_yticks([])
    ax[0].set_ylim(0, 2)
    ax[0].yaxis.set_label_coords(*Y_AXIS_COORDS)
    ax[0].set_ylabel("j", fontsize=FONTSIZE)
    ## x axis
    ax[0].set_xticks([])
    ax[0].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0].tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
    ## other axes
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    # j trace
    ax[1].plot(tj_axis, kj, label="kj exact, time-based", color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False)
    ## y axis
    ax[1].set_yticks([0,max(kj)], ["", ""])
    ax[1].set_ylim(0,max(kj))
    ax[1].yaxis.set_label_coords(*Y_AXIS_COORDS)
    ax[1].set_ylabel("kj", fontsize=FONTSIZE)
    ax[1].tick_params(axis='y', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    ax[1].spines['left'].set_position(BOTTOM_POS)
    ax[1].spines['left'].set_linewidth(BOTTOM_WIDTH)
    ## x axis
    ax[1].set_xticks([])
    ax[1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[1].tick_params(axis='x', which='minor', bottom=False, labelbottom=False)    
    ## other axes
    #ax[1].spines['left'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    # i spikes
    ax[2].scatter(np.where(spikes_i > 0)[0], spikes_i[spikes_i > 0], label="i spikes", s=MARKERSIZE, marker='.', color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False)
    ## y axis
    ax[2].set_yticks([])
    ax[2].set_ylim(0, 2)
    ax[2].yaxis.set_label_coords(*Y_AXIS_COORDS)
    ax[2].set_ylabel("i", fontsize=FONTSIZE)
    ## x axis
    ax[2].set_xticks([])
    ax[2].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[2].tick_params(axis='x', which='minor', bottom=False, labelbottom=False)    
    ## other axes
    ax[2].spines['left'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)

    # i trace
    ax[3].plot(ti_axis, ki, label="ki exact, time-based", color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False)
    ## y axis
    ax[3].set_yticks([0,max(kj)], ["", ""])
    ax[3].set_ylim(0,max(kj))
    ax[3].yaxis.set_label_coords(*Y_AXIS_COORDS)
    ax[3].set_ylabel("ki", fontsize=FONTSIZE)
    ax[3].tick_params(axis='y', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    ax[3].spines['left'].set_position(BOTTOM_POS)
    ax[3].spines['left'].set_linewidth(BOTTOM_WIDTH)
    ## x axis
    ax[3].set_xticks([])
    ax[3].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[3].tick_params(axis='x', which='minor', bottom=False, labelbottom=False)    
    ## other axes
    #ax[1].spines['left'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)

    # dw
    ax[4].plot(w, label="ki exact, time-based", color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False, drawstyle='steps-post')
    ## y axis
    ax[4].set_yticks([0,max(w)], ["", ""])
    ax[4].set_ylim(0,max(w))
    ax[4].yaxis.set_label_coords(*Y_AXIS_COORDS)
    ax[4].set_ylabel("w", fontsize=FONTSIZE)
    ax[4].tick_params(axis='y', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    ax[4].spines['left'].set_position(BOTTOM_POS)
    ax[4].spines['left'].set_linewidth(BOTTOM_WIDTH)
    ## x axis
    ax[4].set_xticks([0, len(w)])
    ax[4].set_xlim(0, len(w))
    ax[4].xaxis.set_label_coords(*X_AXIS_COORDS)
    ax[4].set_xlabel("Time [ms]", fontsize=FONTSIZE)
    #ax[4].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[4].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    ax[4].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH)
    ax[4].spines['bottom'].set_position(BOTTOM_POS)
    ax[4].spines['bottom'].set_linewidth(BOTTOM_WIDTH)
    ## other axes
    ax[4].spines['top'].set_visible(False)
    ax[4].spines['right'].set_visible(False)

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
    # params
    FONTSIZE = 15
    Y_MAJORTICKS_LABELSIZE = 15
    X_MAJORTICKS_LABELSIZE = 15
    MARKERSIZE = 50
    AXISWIDTH = 1.0
    FIGWIDTH = 5.6 # 6.3
    X_MAJORTICKS_LENGTH = 10
    Y_MAJORTICKS_LENGTH = 10
    X_MAJORTICKS_WIDTH = 1.0
    Y_MAJORTICKS_WIDTH = 1.0
    BOTTOM_WIDTH = AXISWIDTH
    BOTTOM_POS = ('outward', 4)
    LEFT_WIDTH = AXISWIDTH
    LEFT_POS = ('outward', 6)
    FIGSIZE = (FIGWIDTH, FIGWIDTH * (2/3))

    h     = 1 # ms
    K     = 101 # ms
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
    w        = 0.7/(1-alpha)

    x = list(range(0,K))

    for k in range(K-1):
        v[k+1] = alpha * v[k] + (1-alpha) * w * o_in[k] - alpha * o_out[k] * v[k]
        if v[k+1] > v_thresh:
            v[k+1] = v_thresh
            o_out[k+1] = 1

    # filter v to make smoother
    #b, a = butter(4, 0.4, btype='low', analog=False)
    #print(v.shape)
    #v = filtfilt(b, a, v[:,0])

    # plot results
    fig, ax = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True,gridspec_kw={'height_ratios': [0.2, 1, 0.2]})
    fig.subplots_adjust(hspace=0.2)
    if type(ax) is not list and type(ax) is not np.ndarray:
        ax = [ax]

    # FONTSIZE = 20
    # Y_MAJORTICKS_LABELSIZE = 25
    # X_MAJORTICKS_LABELSIZE = 25
    ## input spikes
    ax[0].scatter(np.where(o_in > 0)[0], o_in[o_in>0], s=100, marker=".", color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False)
    ## x axis
    ax[0].set_xticks([])
    ax[0].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[0].tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
    ax[0].spines['bottom'].set_visible(False)
    ## y axis
    ax[0].set_yticks([])
    ax[0].tick_params(axis='y', bottom=False, labelbottom=False)
    ax[0].spines['left'].set_visible(False)
    ax[0].yaxis.set_label_coords(-0.04, 0.55)
    ax[0].set_ylabel(r"$\sum_j o_j(t)$", fontsize=FONTSIZE)
    ## other axes
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ## voltage
    ax[1].plot(x, v, color=BLUE, label="LIF Neuron", linewidth=LINEWIDTH, linestyle="solid", clip_on=False) # , drawstyle='steps-post'
    ax[1].axhline(y = 1.0, color=GREY, ls='--', lw=2, clip_on=False)
    ## x axis
    ax[1].set_xticks([])
    ax[1].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[1].tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
    ax[1].spines['bottom'].set_visible(False)
    ## y axis
    ax[1].set_yticks([0, v_thresh*1.0])
    ax[1].set_ylim(0, v_thresh*1.0)
    ax[1].yaxis.set_label_coords(-0.08, 0.5)
    ax[1].set_ylabel(r"$u(t)/u_t$", fontsize=FONTSIZE) # , fontweight='bold') # uₜ
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[1].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE)
    ax[1].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH)
    ax[1].spines['left'].set_position(LEFT_POS)
    ax[1].spines['left'].set_linewidth(LEFT_WIDTH)
    ## other axes
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    ## output spikes
    ax[2].scatter(np.where(o_out > 0)[0], o_out[o_out>0], s=100, marker=".", color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False)
    ## x axis
    ax[2].set_xticks([0, max(x)/2, max(x)])
    ax[2].set_xlim(0, max(x))
    ax[2].xaxis.set_label_coords(0.0, -0.11)
    #ax[2].set_xlabel(r"$\boldsymbol{t}$ [ms]", fontsize=FONTSIZE, fontweight='bold')
    ax[2].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[2].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    ax[2].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH)
    ax[2].spines['bottom'].set_position(BOTTOM_POS)
    ax[2].spines['bottom'].set_linewidth(BOTTOM_WIDTH)
    ## y axis
    ax[2].set_yticks([])
    ax[2].tick_params(axis='y', bottom=False, labelbottom=False)
    ax[2].spines['left'].set_visible(False)
    ax[2].yaxis.set_label_coords(-0.07, 0.5)
    ax[2].set_ylabel(r"$o_i(t)$", fontsize=FONTSIZE)
    ## other axes
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()

def neuroaix_pies():
    FONTSIZE = 8
    FIGWIDTH = 6.3 # 5.6
    FIGSIZE = (FIGWIDTH, FIGWIDTH * (3/3))
    facecolor = "#eaeaf2"

    colors = [RED, BLUE, GREEN, YELLOW, VIOLET]
    colors_finegrain = [*REDS[0:2], *BLUES, *GREENS, *YELLOWS, *VIOLETS]

    font_color = GREY
    size = 0.3
    rad = 0.7
    annot_dist = 0.3
    label_dist = 0.4

    fig, ax = plt.subplots(2, 2, figsize=FIGSIZE, facecolor=facecolor)

    ## LUTs static
    labels = ["Memory", "Communication", "Neural Network", "Control"]
    vals = [35768, 63201, 77895, 75411]
    labels_finegrain = ["Spike\nDispatcher", "MiG",
                        "Scheduler", "Flow Control", "Auroras", "Router",
                        "Ring Buffers", "Load Balancing", "Local Router", "ODE Solvers",
                        "MicroBlaze", "Spike Logging", "Others", "AXI/AXIS\nInterconnect\n"]
    vals_finegrain = ["5216", "30552",
                        "637", "11028", "14013", "37523",
                        "5950", "6972", "18564", "46409",
                        "3227", "4285", "10915", "56984"]

    percs_glob = ["2.09%", "12.26%",
                    "0.26%", "4.42%", "5.62%", "15.05%",
                    "2.39%", "2.80%", "7.45%", "18.62%",
                    "1.29%", "1.72%", "4.38%", "22.86%"]
    
    patches, texts = ax[0][0].pie(vals,
                            radius=rad - size,
                            startangle=50,
                            colors=colors,
                            labels=None,
                            textprops={"color": font_color, "fontsize": 12, "weight": "bold"},
                            wedgeprops=dict(width=size, edgecolor="w"))

    for t in texts:
        t.set_horizontalalignment("center")

    patches, texts = ax[0][0].pie(vals_finegrain,
                            radius=rad,
                            startangle=50,
                            colors=colors_finegrain,
                            wedgeprops=dict(width=size, edgecolor="w"))

    kw = dict(arrowprops=dict(arrowstyle="-", color=font_color), zorder=0, va="center")

    for i, p in enumerate(patches):
        if labels_finegrain[i] == "Scheduler" or labels_finegrain[i] == "MicroBlaze":
            xdist = 0.85
            ydist = 1.3
        elif labels_finegrain[i] == "AXI/AXIS\nInterconnect\n":
            xdist = 0.9
            ydist = 0.5
        elif labels_finegrain[i] == "Load Balancing":
            xdist = 0.6
            ydist = 1.2
        elif labels_finegrain[i] == "Local Router":
            xdist = 0.6
            ydist = 1.2
        elif labels_finegrain[i] == "ODE Solvers":
            xdist = 0.6
            ydist = 1
        elif labels_finegrain[i] == "Spike Logging":
            xdist = 0.775
            ydist = 1.2
        elif labels_finegrain[i] == "Flow Control":
            xdist = 0.8
            ydist = 1.2
        elif labels_finegrain[i] == "Ring Buffers":
            xdist = 0.8
            ydist = 1.1
        else:
            xdist = 0.85
            ydist = 1.2

        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax[0][0].annotate(labels_finegrain[i] + " (" + percs_glob[i] + ")", xy=(annot_dist * x, annot_dist * y), xytext=(xdist * np.sign(x), ydist * y),
                    horizontalalignment=horizontalalignment, fontsize=FONTSIZE, **kw)

    for t in texts:
        t.set_horizontalalignment("center")    

    ## LUTs plastic
    labels = ["Memory", "Communication", "Neural Network", "Control", "Plasticity"]
    vals = [35768, 63201, 77895, 72424, 148592]
    labels_finegrain = ["Spike\nDispatcher", "MiG",
                        "Scheduler", "Flow Control", "Auroras", "Router",
                        "Ring Buffers", "Load Balancing", "Local Router", "ODE Solvers",
                        "MicroBlaze", "Spike Logging", "Others", "AXI/AXIS\nInterconnect\n",
                        "Spike Core", "Plasticity Core"]
    vals_finegrain = ["5216", "30552",
                        "637", "11028", "14013", "37523",
                        "5950", "6972", "18564", "46409",
                        "3227", "4285", "7928", "56984",
                        "102000", "46592"]

    percs_glob = ["1.31%", "7.68%",
                    "0.16%", "2.77%", "3.52%", "9.43%",
                    "1.50%", "1.75%", "4.67%", "11.66%",
                    "0.81%", "1.08%", "1.99%", "14.32%",
                    "25.64%", "11.71%"]

    patches, texts = ax[1][0].pie(vals,
                            radius=rad - size,
                            startangle=50,
                            colors=colors,
                            labels=None,
                            textprops={"color": font_color, "fontsize": 12, "weight": "bold"},
                            wedgeprops=dict(width=size, edgecolor="w"))

    for t in texts:
        t.set_horizontalalignment("center")

    patches, texts = ax[1][0].pie(vals_finegrain,
                            radius=rad,
                            startangle=50,
                            colors=colors_finegrain,
                            wedgeprops=dict(width=size, edgecolor="w"))

    kw = dict(arrowprops=dict(arrowstyle="-", color=font_color), zorder=0, va="center")

    for i, p in enumerate(patches):
        if labels_finegrain[i] == "Scheduler" or labels_finegrain[i] == "MicroBlaze":
            xdist = 0.85
            ydist = 1.3
        elif labels_finegrain[i] == "AXI/AXIS\nInterconnect\n":
            xdist = 0.9
            ydist = 1.3
        elif labels_finegrain[i] == "Load Balancing":
            xdist = 0.8
            ydist = 1.2
        elif labels_finegrain[i] == "Spike Logging":
            xdist = 0.775
            ydist = 1.2
        else:
            xdist = 0.85
            ydist = 1.2

        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax[1][0].annotate(labels_finegrain[i] + " (" + percs_glob[i] + ")", xy=(annot_dist * x, annot_dist * y), xytext=(xdist * np.sign(x), ydist * y),
                    horizontalalignment=horizontalalignment, fontsize=FONTSIZE, **kw)

    for t in texts:
        t.set_horizontalalignment("center")    


    ## BRAMS static
    labels = ["Memory", "Communication", "Neural Network", "Control"]
    vals = [40, 80, 297, 35]
    labels_finegrain = ["MiG", "Spike Dispatcher",
                        "Flow Control", "Scheduler", "Router", "Auroras",
                        "Local Router", "ODE Solvers", "Load Balancing", "Ring Buffers",
                        "AXI/AXIS Interconnect", "Others", "Spike\nLogging", "MicroBlaze"]
    vals_finegrain = ["2", "38",
                        "80", "0", "0", "0",
                        "25", "0", "32", "240",
                        "0", "2", "9", "24"]
    percs_glob = ["0.44%", "8.41%",
                    "0.00%", "0.00%", "0.00%", "17.70%",
                    "0.00%", "5.53%", "7.08%", "53.10%",
                    "0.00%", "0.44%", "1.99%", "5.31%"]

    patches, texts = ax[0][1].pie(vals,
                            radius=rad - size,
                            startangle=35,
                            colors=colors,
                            labels=None,
                            textprops={"color": font_color, "fontsize": 8, "weight": "bold"},
                            wedgeprops=dict(width=size, edgecolor="w"))

    for t in texts:
        t.set_horizontalalignment("center")

    #plt.legend(patches, labels, loc="lower left", bbox_to_anchor=(-0.64, -0.15, 0.3, 0.5), fontsize=24)

    patches, texts = ax[0][1].pie(vals_finegrain,
                            radius=rad,
                            startangle=35,
                            colors=colors_finegrain,
                            wedgeprops=dict(width=size, edgecolor="w"))

    kw = dict(arrowprops=dict(arrowstyle="-", color=font_color), zorder=0, va="center")

    for i, p in enumerate(patches):
        if percs_glob[i] == "0.00%" or labels_finegrain[i] == "Others":  # .startswith("0"):
            print("skipping " + labels_finegrain[i])
            continue

        if labels_finegrain[i] == "Ring Buffers":
            xdist = 0.3
            ydist = 0.9
        elif labels_finegrain[i] == "Spike Dispatcher":
            xdist = 0.3
        elif labels_finegrain[i] == "MicroBlaze":
            xdist = 0.75
            ydist = 1.15
        elif labels_finegrain[i] == "Spike\nLogging":
            xdist = 0.9
        elif labels_finegrain[i] == "Load Balancing":
            xdist = 0.8
        elif labels_finegrain[i] == "Auroras":
            xdist = -0.5
            ydist = 1.4
        else:
            xdist = 0.85
            ydist = 1.2

        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax[0][1].annotate(labels_finegrain[i] + " (" + percs_glob[i] + ")", xy=(annot_dist * x, annot_dist * y), xytext=(xdist * np.sign(x), ydist * y),
                    horizontalalignment=horizontalalignment, fontsize=FONTSIZE, **kw)

    for t in texts:
        t.set_horizontalalignment("center")    

    ## BRAMs plastic
    labels = ["Memory", "Communication", "Neural Network", "Control", "Plasticity"]
    vals = [40, 80, 177, 35, 234]
    labels_finegrain = ["MiG", "Spike Dispatcher",
                        "Flow Control", "Scheduler", "Router", "Auroras",
                        "Local Router", "ODE Solvers", "Load Balancing", "Ring Buffers",
                        "AXI/AXIS Interconnect", "Others", "Spike\nLogging", "MicroBlaze",
                        "Spike Core", "Plasticity Core"]
    vals_finegrain = ["2", "38",
                        "80", "0", "0", "0",
                        "25", "0", "32", "120",
                        "0", "2", "9", "24",
                        "180", "54"]
    percs_glob = ["0.35%", "6.71%",
                    "0.00%", "0.00%", "0.00%", "14.13%",
                    "0.00%", "4.42%", "5.65%", "21.20%",
                    "0.00%", "0.35%", "1.59%", "4.24%",
                    "31.80%", "9.54%"]

    patches, texts = ax[1][1].pie(vals,
                            radius=rad - size,
                            startangle=35,
                            colors=colors,
                            labels=None,
                            textprops={"color": font_color, "fontsize": 8, "weight": "bold"},
                            wedgeprops=dict(width=size, edgecolor="w"))

    for t in texts:
        t.set_horizontalalignment("center")

    #plt.legend(patches, labels, loc="lower left", bbox_to_anchor=(-0.64, -0.15, 0.3, 0.5), fontsize=24)

    patches, texts = ax[1][1].pie(vals_finegrain,
                            radius=rad,
                            startangle=35,
                            colors=colors_finegrain,
                            wedgeprops=dict(width=size, edgecolor="w"))

    kw = dict(arrowprops=dict(arrowstyle="-", color=font_color), zorder=0, va="center")

    for i, p in enumerate(patches):
        if percs_glob[i] == "0.00%" or labels_finegrain[i] == "Others":  # .startswith("0"):
            print("skipping " + labels_finegrain[i])
            continue

        if labels_finegrain[i] == "Ring Buffers":
            ydist = 0.9
            xdist = 0.8
        elif labels_finegrain[i] == "Spike Dispatcher":
            xdist = 0.5
        elif labels_finegrain[i] == "Spike\nLogging":
            xdist = 0.95
        elif labels_finegrain[i] == "Load Balancing":
            xdist = 0.8
        elif labels_finegrain[i] == "Auroras":
            xdist = -1.0
        elif labels_finegrain[i] == "ODE Solvers":
            xdist = -1.0
            ydist = 2.0
        else:
            ydist = 1.2
            xdist = 0.85

        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax[1][1].annotate(labels_finegrain[i] + " (" + percs_glob[i] + ")", xy=(annot_dist * x, annot_dist * y), xytext=(xdist * np.sign(x), ydist * y),
                    horizontalalignment=horizontalalignment, fontsize=FONTSIZE, **kw)

    for t in texts:
        t.set_horizontalalignment("center")

#    ax.set_title("Block-RAMs", fontsize=24, weight="bold", y=1.05)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def neuroaix_util():
    labels = ["LUT", "BRAM", "DSP", "IO", "GT", "MMCM", "PLL"]  # "BUFG",
    percs = ["58%", "31%", "8%", "31%", "33%", "40%", "10%"]  # "100%",
    values = [58, 31, 8, 31, 33, 40, 10]

    fig, ax = plt.subplots()
    [i.set_linewidth(1.5) for i in ax.spines.values()]
    plt.gca().yaxis.grid(True, color=GREY, linestyle="-", zorder=0)
    p = plt.bar(labels, values, color=BLUE, zorder=3)
    plt.bar_label(p, labels=percs, fontsize=18, weight="bold")
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 100)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def neuroaix_estim():
    labels = ["23.9x", "20.4x", "20.3x", "16.6x", "12.7x", "11.0x", "10.9x"]  # , "4.1x"] #["", "", "", "", "", ""]
    y = [0, 1, 2, 3, 4, 5, 6]
    percs = ["...120 ns aurora latency",
            "Latency model of neuroAIx",
            "Measured neuroAIx",
            "...256 neurons per node",
            "...no DRAM\nlat. hiding",
            "...6.4GB/s\nDRAM BW",
            "...no long-hops"]  # , "Measured\nby\nHeittmann"]
    # percs = ["27.9%", "22%", "8%", "31%", "33%", "100%", "40%", "10%"]
    values = [23.9, 20.4, 20.3, 16.6, 12.7, 11.0, 10.9]  # , 4.1]
    bl = "#6a85b0"  # 6aa8d4
    gr = "#80996e"  # 6ad4a8
    # barColors = ["grey","red","blue","grey","grey","grey","grey","blue"]
    bar_colors = [bl, bl, gr, bl, bl, bl, bl]  # ,gr]
    # hatch = "/" # "--","+","*","\\","//","/","",""
    # barHatches = [hatch, "", "", hatch, hatch, hatch, hatch] #, ""]
    fig, ax = plt.subplots()
    [i.set_linewidth(1.5) for i in ax.spines.values()]
    plt.gca().xaxis.grid(True, color="gray", linestyle="-", zorder=0)
    p = plt.barh(y, values, color=bar_colors, zorder=3, edgecolor="w")  # "#6aa8d4" , hatch=barHatches
    plt.bar_label(p, labels=percs, fontsize=14, label_type="center")  # , weight="bold")
    plt.bar_label(p, labels=labels, fontsize=14)
    plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.yticks([])
    plt.xlim(0, 31)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def neuroaix_acc_scaling():
    # params
    FONTSIZE = 12
    Y_MAJORTICKS_LABELSIZE = 12
    X_MAJORTICKS_LABELSIZE = 12
    MARKERSIZE = 8
    AXISWIDTH = 1.0
    FIGWIDTH = 6.3
    X_MAJORTICKS_LENGTH = 10
    Y_MAJORTICKS_LENGTH = 10
    X_MAJORTICKS_WIDTH = 1.0
    Y_MAJORTICKS_WIDTH = 1.0
    FIGSIZE = (FIGWIDTH, FIGWIDTH * (9/16))

    # code
    x = np.linspace(1, 80000, 100)
    x_mid = np.linspace(1, 8000, 100)
    x_small = np.linspace(1, 4000, 1000)

    x_values = [77, 155, 385, 771, 1544, 3858, 7717, 15435, 38586, 77169]

    y_nest_values = [12.26, 9.75, 3.53, 2.36, 1.82, 1.85, 1.05, 0.78, 0.47, 0.26]
    y_nest_spl = make_interp_spline(x_values, y_nest_values, k=1)
    y_nest = y_nest_spl(x)

    y_neuroaixall_values = [33.78, 33.77, 33.73, 33.17, 32.56, 31.57, 30.82, 29.37, 24.61, 20.36]
    y_neuroaixall_spl = make_interp_spline(x_values, y_neuroaixall_values, k=3)
    y_neuroaixall = y_neuroaixall_spl(x)
    
    y_neuroaixsmall_values = [124.36, 117.02, 97.52, 42.60, 24.66, 2, 1, 0]  # , 16.4 16.4 extrapolated...
    #y_neuroaixsmall_values = [124.36, 119.69, 115.02, 106.27, 97.52, 70, 42.60, 33.63, 24.66, 16.4]  #  16.4 extrapolated...
    y_neuroaixsmall_spl = make_interp_spline(x_values[0:len(y_neuroaixsmall_values)], y_neuroaixsmall_values, k=3)
    y_neuroaixsmall = y_neuroaixsmall_spl(x_small)

    y_neuroaixmid_values = [62.02, 61.48, 59.80, 50.29, 40.92, 30.61, 23.95]
    y_neuroaixmid_spl = make_interp_spline(x_values[0:len(y_neuroaixmid_values)], y_neuroaixmid_values, k=3)
    y_neuroaixmid = y_neuroaixmid_spl(x_mid)

    y_inc3000 = 4.06
    y_epyc = 1.88
    y_spinnaker = 1.0


    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax = [ax]

    ax[0].plot(x, y_nest, linewidth=LINEWIDTH, label='NEST (Intel Xeon)', color=GREY)
    ax[0].plot(x_small[0:260], y_neuroaixsmall[0:260], '-', linewidth=LINEWIDTH, label='neuroAIˣ (1x1)', color=RED)
    ax[0].plot(x_mid, y_neuroaixmid, '-', linewidth=LINEWIDTH, label='neuroAIˣ (4x1)', color=GREEN)
    ax[0].plot(x, y_neuroaixall, linewidth=LINEWIDTH, label='neuroAIˣ (5x7)', color=BLUE)

    ax[0].plot(x_values[-1], y_inc3000, 'o', markersize=MARKERSIZE, color=GREY)
    ax[0].plot(x_values[-1], y_epyc, 'o', markersize=MARKERSIZE, color=GREY)
    ax[0].plot(x_values[-1], y_spinnaker, 'o', markersize=MARKERSIZE,  color=GREY)

    ## x axis
    ax[0].set_xscale('log')
    # ax[0].set_xticks([4, 16, 64], ["4", "16", "64"])
    # ax[0].set_xlim(1.5, 80)
    ax[0].xaxis.set_label_coords(0.0, -0.08)
    ax[0].set_xlabel("Neurons", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    #ax[0].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['bottom'].set_linewidth(AXISWIDTH)

    ## y axis
    ax[0].set_yscale('log')
    # ax[0].set_yticks([0.01, 0.1, 1])
    # ax[0].set_ylim(0.005, 2)
    #ax[0].yaxis.set_label_coords(-0.11, 0.0)
    ax[0].set_ylabel("Acceleration", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    #ax[0].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['left'].set_linewidth(AXISWIDTH)

    ## other axes
    ax[0].spines['top'].set_linewidth(AXISWIDTH)
    ax[0].spines['right'].set_linewidth(AXISWIDTH)

    ax[0].annotate('INC-3000', xy=(x_values[-1]-2000, y_inc3000), xytext=(x_values[-1]-61000, y_inc3000+1), color=GREY, fontsize=FONTSIZE)
                #arrowprops=dict(arrowstyle = '-', connectionstyle = 'arc3',facecolor=GREY))
    ax[0].annotate('SpiNNaker', xy=(x_values[-1]-2000, y_spinnaker), xytext=(x_values[-1]-64000, y_spinnaker+0.2), color=GREY, fontsize=FONTSIZE)
                #arrowprops=dict(arrowstyle = '-', connectionstyle = 'arc3',facecolor=GREY))
    ax[0].annotate('NEST (AMD Epyc)', xy=(x_values[-1]-2000, y_epyc), xytext=(x_values[-1]-74000, y_epyc+0.5), color=GREY, fontsize=FONTSIZE)
                #arrowprops=dict(arrowstyle = '-', connectionstyle = 'arc3',facecolor=GREY))

    ax[0].tick_params(direction = 'in')
    plt.minorticks_off()
    # for t in ax.get_xmajorticklabels():
    #     t.set_fontsize(24)
    # for t in ax.get_ymajorticklabels():
    #     t.set_fontsize(24)

    ## grid
    ax[0].grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)

    ## legend
    ax[0].legend(loc='lower left', fontsize=FONTSIZE)

    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def neuroaix_latency(scenario=2):
    if scenario==0:  # current
        vals =   [97, 1, 2]
        vals_finegrain = ['0.03', '0.08', '0.16', '3.23', '94.46',
                        '0.36', 
                        '0.70', '0.97']
        percs_glob = ['4s', '10s', '20s',  '6m40s', '3h15m',
                    '45s', 
                    '1m27s', '2m00s']
        
        labels = ['Preprocessing', 'Simulation', 'Postprocessing']
    
        labels_finegrain = ['Init', 'Neuron Gen', 'Neuron Upload', 'Synapse Upload', 'Synapse Gen',
                        'Simulate',
                        '2nd Order', 'Spike Download']
        colors = [YELLOW, BLUE, RED]
        colors_finegrain = ['#f2cfaa', '#f2cfaa', '#f2c391', '#f2c391','#f2b879',  #YELLOW
                        '#9ea5b0',   #BLUE
                        '#d1bcbc', '#d1a7a7']  #RED

        title = "Latencies (Current) -> 3h26m26s"
    elif scenario==1:  # w/o synapse file extraction
        vals =   [65, 6, 29]
        vals_finegrain = ['0.56', '1.40', '2.79', '4.19', '55.87',
                        '6.28', 
                        '12.15', '16.76']
        percs_glob = ['4s', '10s', '20s', '30s', '6m40s',
                    '45s', 
                    '1m27s', '2m00s']
        
        labels = ['Preprocessing', 'Simulation', 'Postprocessing']
    
        labels_finegrain = ['Init', 'Neuron Gen', 'Neuron Upload', 'Synapse Gen', 'Synapse Upload',
                        'Simulate',
                        '2nd Order', 'Spike Download']
        colors = [YELLOW, BLUE, RED]
        colors_finegrain = ['#f2cfaa', '#f2cfaa', '#f2c391', '#f2c391','#f2b879',  #YELLOW
                        '#9ea5b0',   #BLUE
                        '#d1bcbc', '#d1a7a7']  #RED

        title = "Latencies (faster synapse extraction) -> 11m56s"
    elif scenario==2:  # w/o synapse file extraction and 1Gbit/s PCIe and synapse file compression from 12.6GB -> 3.4GB
        vals =   [39, 20, 41]
        vals_finegrain = ['1.74', '4.35', '8.7', '11.74', '13.04',
                        '19.57', 
                        '3.04', '37.83']
        percs_glob = ['4s', '10s', '20s', '27s','30s',
                    '45s', 
                    '7s', '1m27s']
        
        labels = ['Preprocessing', 'Simulation', 'Postprocessing']
    
        labels_finegrain = ['Init', 'Neuron Gen', 'Neuron Upload', 'Synapse Upload', 'Synapse Gen',
                        'Simulate',
                        'Spike Download', '2nd Order']
        colors = [YELLOW, BLUE, RED]
        colors_finegrain = ['#f2cfaa', '#f2cfaa', '#f2c391', '#f2c391','#f2b879',  #YELLOW
                        '#9ea5b0',   #BLUE
                        '#d1bcbc', '#d1a7a7']  #RED

        title = "Latencies (faster synapse extraction + 1G Ethernet) -> 3m50s"
    elif scenario==3:  # w/o synapse file extraction and 16Gbit/s PCIe and synapse file compression from 12.6GB -> 3.4GB
        vals =   [33, 24, 44]
        vals_finegrain = ['1.01', '2.02', '5.05', '10.10', '15.15',
                        '22.73', 
                        '0.01', '39.99']
        percs_glob = ['2s', '4s', '10s', '20s', '30s',
                    '45s', 
                    '0s', '1m27s']
        
        labels = ['Preprocessing', 'Simulation', 'Postprocessing']
    
        labels_finegrain = ['Synapse Upload', 'Init', 'Neuron Gen', 'Neuron Upload', 'Synapse Gen',
                        'Simulate',
                        'Spike Download', '2nd Order']
        colors = [YELLOW, BLUE, RED]
        colors_finegrain = ['#f2cfaa', '#f2cfaa', '#f2c391', '#f2c391','#f2b879',  #YELLOW
                        '#9ea5b0',   #BLUE
                        '#d1bcbc', '#d1a7a7']  #RED   
        
        title = "Latencies (faster synapse extraction + 16G PCIe) -> 3m18s"
    elif scenario==4:  # w/o synapse file extraction and 16Gbit/s PCIe and synapse file compression from 12.6GB -> 3.4GB
        vals =   [35.91, 20.45, 43.64]
        vals_finegrain = ['1.82', '4.55', '0.45', '13.64', '15.45',
                        '20.45', 
                        '4.09', '39.55']
        percs_glob = ['4s', '10s', '1s', '30s', '34s',
                    '45s', 
                    '9s', '1m27s']
        
        labels = ['Preprocessing', 'Simulation', 'Postprocessing']
    
        labels_finegrain = ['Init', 'Neuron Gen', 'Neuron Upload', 'Synapse Gen', 'Synapse Upload', 
                        'Simulate',
                        'Spike Download', '2nd Order']
        colors = [YELLOW, BLUE, RED]
        colors_finegrain = ['#f2cfaa', '#f2cfaa', '#f2c391', '#f2c391','#f2b879',  #YELLOW
                        '#9ea5b0',   #BLUE
                        '#d1bcbc', '#d1a7a7']  #RED   
        
        #title = "Latencies (faster synapse extraction + 800MB Ethernet) -> 3m40s"

    font_color = GREY
    size = 0.3
    rad = 0.8

    facecolor = '#eaeaf2'
    fig, ax = plt.subplots(figsize=(15,9), facecolor=facecolor)
    patches, texts = ax.pie(vals, 
        radius=rad-size, 
        startangle=90,
        counterclock=False,
        colors=colors, 
        labels=None,
        labeldistance=0.6,
        textprops={'color':font_color, 'fontsize':8, 'weight':'bold'},
        wedgeprops=dict(width=size, edgecolor='w'))
    
    for t in texts:
        t.set_horizontalalignment('center')

    if scenario==4:
        plt.legend(patches, labels, fontsize=24, bbox_to_anchor=[0.3,1])
    
    patches, texts = ax.pie(vals_finegrain, 
        radius=rad, 
        startangle=90,
        counterclock=False,
        colors=colors_finegrain,
        wedgeprops=dict(width=size, edgecolor='w'))
    
    kw = dict(arrowprops=dict(arrowstyle="-", color=font_color), zorder=0, va="center")

    for i, p in enumerate(patches):
        if percs_glob[i] == "0.00%" or labels_finegrain[i] == "Others": #.startswith('0'):
            print("skipping " + labels_finegrain[i])
            continue

        if labels_finegrain[i] == "Neuron Gen":
            ydist=1.1
            xdist=0.85
        elif labels_finegrain[i] == "Neuron Upload":
            ydist=1.0
            xdist=0.85      
        else:
            ydist=1.2
            xdist=0.85

        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels_finegrain[i]+" ("+percs_glob[i]+")", xy=(0.8*x, 0.8*y), xytext=(xdist*np.sign(x), ydist*y),
                    horizontalalignment=horizontalalignment, fontsize=22, **kw)

    for t in texts:
        t.set_horizontalalignment('center')
    
    Path(OUTPUT).mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".pdf", format='pdf', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".svg", format='svg', transparent=True)
    plt.savefig(OUTPUT+"/"+inspect.stack()[0][3]+".png", format='png', dpi=PNG_DPI, transparent=True)
    if PLOT:
        plt.show()
        plt.clf()
    plt.clf()
    plt.close()


def neuroaix_sota():
    # params
    FONTSIZE_SMALL = 10
    FONTSIZE = 12
    FONTSIZE_BIG = 12
    Y_MAJORTICKS_LABELSIZE = 12
    X_MAJORTICKS_LABELSIZE = 12
    MARKERSIZE = 8
    FIGWIDTH = 6.3
    AXISWIDTH = 1.0
    X_MAJORTICKS_LENGTH = 10
    Y_MAJORTICKS_LENGTH = 10
    X_MAJORTICKS_WIDTH = 1.0
    Y_MAJORTICKS_WIDTH = 1.0
    FIGSIZE = (FIGWIDTH, FIGWIDTH) # (9/16)
    ARROWPROPS = dict(
        facecolor=GREY,     # Arrow color
        edgecolor=GREY,     # Border color of the arrow
        arrowstyle='-|>',   # Style of the arrow (e.g., '->', '-|>', etc.)
        lw=1,               # Line width
    )


    # neuroAIx (2023)	            20,36, 0,047
    # INC-3000 (2022) [17]	        4,06, 0,783
    # AMD EPYC (2022) [22]	        1,88, 0,48
    # SpiNNaker (2019) [21]	        1,00, 0,62
    # SpiNNaker (2018) [20]	        0,05, 5,9
    # Intel Xeon (2018) [20]	    0,22, 5,8
    # Nvidia Tesla V100 (2018) [9]	0,54, 0,47
    # Nvidia 1050 Ti (2018) [9]	    0,07, 2,00
    # Nvidia Jetson (2018) [9]	    0,04, 0,30
    # Nvidia Tesla K40c (2018) [9]	0,25, 1,08
    # Nvidia RTX 2080 Ti (2021) [8]	0,95, 0,18

    data_neuroaix = np.array([20.36, 0.047])
    data_inc = np.array([4.06, 0.783])
    data_epyc = np.array([1.88, 0.48])
    data_spinn19 = np.array([1.00, 0.62])
    data_spinn18 = np.array([0.05, 5.9])
    data_xeon = np.array([0.22, 5.8])
    data_v100 = np.array([0.54, 0.47])
    data_1050 = np.array([0.07, 2.00])
    data_jetson = np.array([0.04, 0.30])
    data_k40c = np.array([0.25, 1.08])
    data_2080 = np.array([0.95, 0.18])

    # experiment data (from https://docs.google.com/spreadsheets/d/1l406Oo9PhM4TjKejtO-YHJUwxebz8lgmVTP1ErljXjI/edit?usp=sharing (SotA sheet))
    # data_fpgas = np.array([[0.047, 20.36], [0.783, 4.06]])
    # data_cpus  = np.array([[0.48, 1.88], [0.62, 1.00], [5.9, 0.05], [5.8, 0.22]])
    # data_gpus  = np.array([[0.47, 0.54], [2.00, 0.07], [0.3, 0.04], [1.08, 0.25], [0.18, 0.95]])

    data_fpgas = np.array([data_neuroaix, data_inc])
    data_cpus  = np.array([data_epyc, data_spinn18, data_spinn19, data_xeon])
    data_gpus  = np.array([data_v100, data_1050, data_jetson, data_k40c, data_2080])

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE) #, figsize=(10,7))
    fig.subplots_adjust(top=0.8, bottom=0.15)
    if type(ax) is not list and type(ax) is not np.ndarray:
        ax = [ax]

    # refs & arrows
    ax[0].annotate("neuroAIx [999]", xytext=(data_neuroaix[0]*0.2, data_neuroaix[1]*1.2), xy=(data_neuroaix[0]*0.2, data_neuroaix[1]*1.2), fontsize=FONTSIZE_BIG, weight='bold')
    ax[0].annotate("INC-3000 [999]", xytext=(data_inc[0]*0.5, data_inc[1]*1.2), xy=(data_inc[0]*0.5, data_inc[1]*1.2), fontsize=FONTSIZE_SMALL)
    ax[0].annotate("AMD Epyc [999]", xytext=(data_epyc[0]*0.6, data_epyc[1]*0.7), xy=(data_epyc[0]*0.6, data_epyc[1]*0.7), fontsize=FONTSIZE_SMALL)
    ax[0].annotate("SpiNNaker [999]", xytext=(data_spinn19[0]*0.4, data_spinn19[1]*1.2), xy=(data_spinn19[0]*0.4, data_spinn19[1]*1.2), fontsize=FONTSIZE_SMALL)
    ax[0].annotate("SpiNNaker [999]", xytext=(data_spinn18[0]*0.5, data_spinn18[1]*1.2), xy=(data_spinn18[0]*0.5, data_spinn18[1]*1.2), fontsize=FONTSIZE_SMALL)
    ax[0].annotate("Intel Xeon [999]", xytext=(data_xeon[0]*0.5, data_xeon[1]*0.7), xy=(data_xeon[0]*0.5, data_xeon[1]*0.7), fontsize=FONTSIZE_SMALL)
    ax[0].annotate("Nvidia V100 [999]", xytext=(data_v100[0]*0.2, data_v100[1]*0.7), xy=(data_v100[0]*0.2, data_v100[1]*0.7), fontsize=FONTSIZE_SMALL)
    ax[0].annotate("Nvidia 1050 Ti [999]", xytext=(data_1050[0]*0.5, data_1050[1]*1.2), xy=(data_1050[0]*0.5, data_1050[1]*1.2), fontsize=FONTSIZE_SMALL)
    ax[0].annotate("Nvidia Jetson [999]", xytext=(data_jetson[0]*0.3, data_jetson[1]*0.7), xy=(data_jetson[0]*0.3, data_jetson[1]*0.7), fontsize=FONTSIZE_SMALL)
    ax[0].annotate("Nvidia K40v [999]", xytext=(data_k40c[0]*0.5, data_k40c[1]*1.2), xy=(data_k40c[0]*0.5, data_k40c[1]*1.2), fontsize=FONTSIZE_SMALL)
    ax[0].annotate("Nvidia RTX 2080 Ti [999]", xytext=(data_2080[0]*0.2, data_2080[1]*0.7), xy=(data_2080[0]*0.2, data_2080[1]*0.7), fontsize=FONTSIZE_SMALL)

    # mnist
    ax[0].scatter(data_fpgas[:, 0], data_fpgas[:, 1], color=GREEN, marker='o', s=50, label="FPGA")
    ax[0].scatter(data_cpus[:, 0], data_cpus[:, 1], color=YELLOW, marker='o', s=50, label="CPU")
    ax[0].scatter(data_gpus[:, 0], data_gpus[:, 1], color=BLUE, marker='o', s=50, label="GPU")

    # x axis
    ax[0].set_xscale('log', base=10)
    ax[0].set_xticks([0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0], ["0.05", "0.10", "0.50", "1.00", "5.00", "10.00", "50.00"])
    ax[0].set_xlim(0.01, 50)
    #ax[0].xaxis.set_label_coords(0.0, -0.11)
    ax[0].set_xlabel("Acceleration", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    #ax[0].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['bottom'].set_linewidth(AXISWIDTH)

    ## y axis
    ax[0].set_yscale('log', base=10)
    ax[0].set_yticks([0.05, 0.1, 0.5, 1.0, 5.0, 10.0], ["0.05", "0.10", "0.50", "1.00", "5.00", "10.00"])
    ax[0].set_ylim(0.01, 10.0)
    #ax[0].yaxis.set_label_coords(-0.11, 0.0)
    ax[0].set_ylabel("Energy per Synaptic Operation [µJ]", fontsize=FONTSIZE) # , fontweight='bold'
    #ax[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    #ax[0].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['left'].set_linewidth(AXISWIDTH)

    ## other axes
    ax[0].spines['top'].set_linewidth(AXISWIDTH)
    ax[0].spines['right'].set_linewidth(AXISWIDTH)
    plt.minorticks_off()

    ## grid
    ax[0].grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.5)

    ## legend
    # plt.legend(ncol=4, frameon=True, loc='upper center', bbox_to_anchor=(0.5, 1.327), fontsize=FONTSIZE, borderpad=0.1, columnspacing=0.01) #, borderpad=0.1, columnspacing=0.01, labelspacing=0.01)
    ax[0].legend(loc='upper right', fontsize=FONTSIZE)

    #plt.tight_layout()
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
