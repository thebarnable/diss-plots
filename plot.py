import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator
from matplotlib.patches import FancyArrow
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import euclidean, correlation #cosine, cityblock
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes
from scipy.interpolate import interp1d
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
    FONTSIZE = 25
    X_MAJORTICKS_LABELSIZE = 25
    Y_MAJORTICKS_LABELSIZE = 25
    MARKERSIZE = 8
    FIGSIZE = (10, 8)

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
    ax[0].set_xlabel("Vector Length (N)", fontsize=FONTSIZE, fontweight='bold')
    #ax[0].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    #ax[0].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].spines['bottom'].set_linewidth(BOTTOM_WIDTH)

    ## y axis
    ax[0].set_yticks([0.01, 0.1, 1])
    ax[0].set_ylim(0.005, 2)
    #ax[0].yaxis.set_label_coords(-0.11, 0.0)
    ax[0].set_ylabel("Normalized Energy Cost", fontsize=FONTSIZE, fontweight='bold')
    #ax[0].yaxis.set_minor_locator(AutoMinorLocator(10))
    ax[0].tick_params(axis='y', length=Y_MAJORTICKS_LENGTH, width=Y_MAJORTICKS_WIDTH, labelsize=Y_MAJORTICKS_LABELSIZE, right=True, top=True, direction='in')
    ax[0].tick_params(axis='y', which='minor', length=Y_MINORTICKS_LENGTH, width=Y_MINORTICKS_WIDTH, right=True, top=True, direction='in')
    ax[0].set_yscale('log')
    ax[0].spines['left'].set_linewidth(LEFT_WIDTH)

    ## other axes
    ax[0].spines['top'].set_linewidth(BOTTOM_WIDTH)
    ax[0].spines['right'].set_linewidth(BOTTOM_WIDTH)

    ## grid
    ax[0].grid(True, which='both', linestyle='-', linewidth=0.4)

    ## legend
    ax[0].legend(frameon=True, loc='lower right',bbox_to_anchor=(1.0, 0.0),fontsize=FONTSIZE)

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
    # generate random data
    # Parameters
    h=0.001 # timestep width
    t_sim=100 # timesteps
    lambda_j = 100 # presyn decay rate
    lambda_i = 100 # postsyn decay rate
    spike_train_j = [1,5,10,20,21]
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
    ax[0].scatter(np.where(spikes_j > 0)[0], spikes_j[spikes_j > 0], label="j spikes", s=100, marker='.', color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False)
    ## y axis
    ax[0].set_yticks([])
    ax[0].set_ylim(0, 2)
    ax[0].yaxis.set_label_coords(*Y_AXIS_COORDS)
    ax[0].set_ylabel("j", fontsize=FONTSIZE, fontweight='bold')
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
    ax[1].set_ylabel("kj", fontsize=FONTSIZE, fontweight='bold')
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
    ax[2].scatter(np.where(spikes_i > 0)[0], spikes_i[spikes_i > 0], label="i spikes", s=100, marker='.', color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False)
    ## y axis
    ax[2].set_yticks([])
    ax[2].set_ylim(0, 2)
    ax[2].yaxis.set_label_coords(*Y_AXIS_COORDS)
    ax[2].set_ylabel("i", fontsize=FONTSIZE, fontweight='bold')
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
    ax[3].set_yticks([])
    ax[3].yaxis.set_label_coords(*Y_AXIS_COORDS)
    ax[3].set_ylabel("ki", fontsize=FONTSIZE, fontweight='bold')
    ## x axis
    ax[3].set_xticks([])
    ax[3].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[3].tick_params(axis='x', which='minor', bottom=False, labelbottom=False)    
    ## other axes
    ax[3].spines['left'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)

    # dw
    ax[4].plot(w, label="ki exact, time-based", color=BLUE, linewidth=LINEWIDTH, linestyle="solid", clip_on=False, drawstyle='steps-post')
    ## y axis
    ax[4].set_yticks([])
    ax[4].yaxis.set_label_coords(*Y_AXIS_COORDS)
    ax[4].set_ylabel("w", fontsize=FONTSIZE, fontweight='bold')
    ## x axis
    ax[4].set_xticks([0, len(w)])
    ax[4].set_xlim(0, len(w))
    ax[4].xaxis.set_label_coords(*X_AXIS_COORDS)
    ax[4].set_xlabel("Time [ms]", fontsize=FONTSIZE, fontweight='bold')
    #ax[4].xaxis.set_minor_locator(AutoMinorLocator(10))
    ax[4].tick_params(axis='x', length=X_MAJORTICKS_LENGTH, width=X_MAJORTICKS_WIDTH, labelsize=X_MAJORTICKS_LABELSIZE)
    ax[4].tick_params(axis='x', which='minor', length=X_MINORTICKS_LENGTH, width=X_MINORTICKS_WIDTH)
    ax[4].spines['bottom'].set_position(BOTTOM_POS)
    ax[4].spines['bottom'].set_linewidth(BOTTOM_WIDTH)
    ## other axes
    ax[4].spines['left'].set_visible(False)
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
    fig.subplots_adjust(hspace=0.05)
    if type(ax) is not list and type(ax) is not np.ndarray:
        ax = [ax]

    FONTSIZE = 20
    Y_MAJORTICKS_LABELSIZE = 25
    X_MAJORTICKS_LABELSIZE = 25
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
    ax[0].yaxis.set_label_coords(-0.06, 0.5)
    ax[0].set_ylabel(r"$\sum_j \boldsymbol{o}_\boldsymbol{j}\boldsymbol{(t)}$", fontsize=FONTSIZE, fontweight='bold')
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
    ax[1].set_ylabel(r"$\boldsymbol{u(t)}/\boldsymbol{u}_\boldsymbol{t}$", fontsize=FONTSIZE) # , fontweight='bold') # uₜ
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
    ax[2].set_ylabel(r"$\boldsymbol{o}_\boldsymbol{i}\boldsymbol{(t)}$", fontsize=FONTSIZE, fontweight='bold')
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot script')
    parser.add_argument('--function', '-f', default='lif', help='Plot function to call')
    args = parser.parse_args()

    locals()[args.function]()