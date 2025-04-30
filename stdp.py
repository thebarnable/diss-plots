import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# stdp for two neurons (j = pre and i = post)
def main():
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
    kj,ki = np.zeros(t_sim),np.zeros(t_sim)
    ki_event, kj_event = np.zeros(t_sim),np.zeros(t_sim)
    w = np.zeros(t_sim)
    last_spiketime_j, last_spiketime_i = 0, 0
    for t in tqdm(range(1,t_sim-1), desc="# Time-driven exact integration"):
        # integration of spike traces for reference
        kj[t] = np.exp(-h*lambda_j)*kj[t-1] + ((1-np.exp(-h*lambda_j))/lambda_j)*spikes_j[t]
        ki[t] = np.exp(-h*lambda_i)*ki[t-1] + ((1-np.exp(-h*lambda_i))/lambda_i)*spikes_i[t]

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

    # Plot
    RED = "#D17171"
    YELLOW = "#F3A451"
    GREEN = "#7B9965"
    BLUE = "#5E7DAF"
    DARKBLUE = "#3C5E8A"
    DARKRED = "#A84646"
    VIOLET = "#886A9B"
    GREY = "#636363"

    plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
    fig, axs = plt.subplots(5, 1, sharex=True, gridspec_kw={'height_ratios': [0.5, 1, 0.5, 1, 1]})
    fig.subplots_adjust(hspace=0)
    axs[0].scatter(np.where(spikes_j > 0)[0], spikes_j[spikes_j > 0], color=BLUE, label="j spikes", s=5000, linewidth=2, marker='|')
    #axs[0].legend()
    axs[1].plot(kj, color=BLUE, label="kj exact, time-based", linewidth=0.5, linestyle='--')
    axs[1].scatter(np.where(kj_event > 0)[0], kj_event[kj_event > 0], color=BLUE, label="kj exact, event-based", s=20, marker='o')
    axs[1].legend()
    axs[2].scatter(np.where(spikes_i > 0)[0], spikes_i[spikes_i > 0], color=RED, label="i spikes", s=5000, linewidth=2, marker='|')
    #axs[2].legend()
    axs[3].plot(ki, color=RED, label="ki exact, time-based", linewidth=0.5, linestyle='--')
    axs[3].scatter(np.where(ki_event > 0)[0], ki_event[ki_event > 0], color=RED, label="ki exact, event-based", s=20, marker='o')
    axs[3].legend()
    axs[4].plot(w, color=YELLOW, label="w", linewidth=2, linestyle='-', drawstyle='steps-post')
    axs[4].legend()

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.grid(True, color='lightgray', linestyle='--', linewidth=2)

    plt.show() 
    plt.close()

if __name__ == '__main__':
    main()