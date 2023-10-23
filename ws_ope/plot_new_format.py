import plot_utilities as pu
import os
import matplotlib.pyplot as plt

def plot_bars(data, target=None, gamma=None):
    """
    model_dict, target_values, target_stds = data
    keys_sub = list(target_values.keys())
    keys_main = ["Target", "DR", "MB", "FQE"]
    
    bar_width = 0.15
    num_bars = len(keys_sub)
    
    # Create a range for the primary x-axis labels (i.e., Target, DR, MB, FQE)
    r = np.arange(len(keys_main))
    
    # Start plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plotting bars for target values
    ax.bar(r[0]-bar_width, target_values[keys_sub[0]], yerr=target_stds[keys_sub[0]], width=bar_width, label=keys_sub[0], alpha=0.8, capsize=7)
    ax.bar(r[0], target_values[keys_sub[1]], yerr=target_stds[keys_sub[1]], width=bar_width, label=keys_sub[1], alpha=0.8, capsize=7)
    ax.bar(r[0]+bar_width, target_values[keys_sub[2]], yerr=target_stds[keys_sub[2]], width=bar_width, label=keys_sub[2], alpha=0.8, capsize=7)
    
    # Plotting bars for DR, MB, FQE
    for i, model in enumerate(keys_main[1:]):  # Start from 1 to exclude 'Target'
        means = model_dict[model]['means']
        stds = model_dict[model]['std_devs']
        for j, key_sub in enumerate(keys_sub):
            ax.bar(r[i+1] + j*bar_width - (bar_width*(num_bars-1)/2), means[key_sub], yerr=stds[key_sub], width=bar_width, label=(model + '_' + key_sub if i == 0 else ""), alpha=0.8, capsize=7)
    
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Target, DR, MB, and FQE')
    ax.set_xticks(r)
    ax.set_xticklabels(keys_main)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    """
    model_dict, target_values, target_stds = data
    keys_sub = list(target_values.keys())
    keys_main = ["Target", "DR", "MB", "FQE"]
    
    bar_width = 0.15
    num_bars = len(keys_sub)
    
    # Create a range for the primary x-axis labels (i.e., Target, DR, MB, FQE)
    r = np.arange(len(keys_main))
    
    # Set a color palette
    colors = ['C0', 'C1', 'C2'] #plt.cm.rainbow(np.linspace(0, 1, num_bars))
    
    # Start plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plotting bars for each of the main groups (Target, DR, MB, FQE)
    for i, model in enumerate(keys_main):
        # Identify appropriate data for the model
        if model == "Target":
            means = target_values
            stds = target_stds
        else:
            means = model_dict[model]['means']
            stds = model_dict[model]['std_devs']
            
        for j, key_sub in enumerate(keys_sub):
            label = key_sub if i == 0 else None  # Set label only for the first main group to avoid label repetition in the legend
            ax.bar(r[i] + j*bar_width - (bar_width*(num_bars-1)/2), means[key_sub], yerr=stds[key_sub], width=bar_width, label=label, alpha=0.8, capsize=7, color=colors[j])
    
    ax.set_xlabel('OPE Type')
    ax.set_ylabel('Discounted Reward')
    ax.set_title(f"Comparison of Target, DR, MB, and FQE \n (target: {target}, gamma: {gamma})")
    ax.set_xticks(r)
    ax.set_xticklabels(keys_main)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"plots2/{target}/{gamma}/bar_plot.png")




def plot_discount(gamma=0.85,algo = "dr", date="926", path = "/mnt/hdd2/fabian/f1tenth_dope/ws_ope/f1tenth_orl_dataset/data/trajectories.zarr",min_trajectory_length=600, date_2=None,target=None):
    precomputed_returns, precomputed_stds,mean_precomputed, std_precomputed, dataset = pu.compute_target_rewards(
                            gamma=gamma, 
                            #skip_inital=50, 
                            # split_trajectories=100,
                            alternate_reward=True, 
                            # remove_short_trajectories=True,
                            min_trajectory_length=min_trajectory_length,
                            # without_agents=["raceline", "min_lida"]
                            path =path,
                            )

    # importlib.reload(pu)
    logdir = f"logdir/f110_rl_{str(gamma)}_{algo}_{target}_{date}/"
    logdir_mb = f"logdir/f110_rl_{str(gamma)}_{algo}_{target}_{date}/" # f"logdir/f110_rl_{str(gamma)}_mb_{date}/"

    values = pu.plot_everything(logdir,
                    precomputed_returns,
                    precomputed_stds,
                    dataset,
                    logdir_mb=logdir_mb,
                    plt_path = f"plots2/{target}/{gamma}",
                    gamma=gamma)
    plt.show()
    return values

import numpy as np
from scipy import stats
import pickle

def process_values(values, path=None):
    models_dict = values[0]
    precomputed_returns = values[1]
    print("----")
    print(precomputed_returns)
    # loop over the models (the keys in models_dict)
    mse = dict()
    mse = {"DR": dict(), "MB": dict(), "FQE": dict()}
    for model in models_dict.keys():
        for target in models_dict[model]['means'].keys():
            if target not in precomputed_returns.keys():
                continue
            else:
                print(target)
                print(models_dict[model])
                print("xs")
                print(models_dict[model]['means'][target])
                print(precomputed_returns[target])
                mse_ = abs(models_dict[model]['means'][target]- precomputed_returns[target]) # **2
                mse[model][target] = mse_
    print(mse)

    # now check which target is closest to the precomputed returns
    bestk = dict()
    bestk = {"DR": dict(), "MB": dict(), "FQE": dict()}
    print("associated key maximum in precomputed returns")
    key_with_max_value = max(precomputed_returns, key=precomputed_returns.get)
    print(key_with_max_value)
    for model in models_dict.keys():
        means = models_dict[model]['means']
        bestk[model] = max(means, key= means.get)
        #for target in models_dict[model]['means'].keys():
        #    if target not in precomputed_returns.keys():
        #        continue
        #    else:
        #        #mse_ = (models_dict[model]['means'][target]- precomputed_returns[target])**2
        #        bestk[model] = np.argsort(mse_)[0]
    # create directory plots
    # set bestk to true where its equivalent to key_with_max_value
    for model in bestk.keys():
        if bestk[model] == key_with_max_value:
            bestk[model] = True
        else:
            bestk[model] = False
    
    print("rank correlation")
    keys = precomputed_returns.keys()
    targets = []
    for key in keys:
        targets.append(precomputed_returns[key])
    targets = np.array(targets)
    spearman_per_model = {}
    for model in models_dict.keys():
        y = []
        for key in keys:
            y.append(models_dict[model]['means'][key])
        y = np.array(y)

        res = stats.spearmanr(y, targets)
        # print(res.correlation.shape)
        # print(res.pvalue)
        spearman_per_model[model] = float(res.correlation)
    
    with open(f"{path}/spearman_per_model.pkl", 'wb') as f:
        pickle.dump(spearman_per_model, f)

    with open(f"{path}/bestk.pkl", 'wb') as f:
        pickle.dump(bestk, f)

    with open(f"{path}/mse.pkl", 'wb') as f:
        pickle.dump(mse, f)

if not os.path.exists("plots2"):
    os.mkdir("plots2")


target_datasets = ["trajectories_min_act.zarr", 
                "trajectories_td_prog.zarr", 
                "trajectories_raceline.zarr"]

gammas = [0.95, 0.99]
#logdir/f110_rl_0.85_dr_922/"
# logdir_mb = "logdir/f110_rl_0.85_mb_925/"
benchmark_date = "1020"
min_trajectory_length = 0

for i, target in enumerate(target_datasets):
    for gamma in gammas:
        targ_dir = f"plots2/{target}"
        if not os.path.exists(targ_dir):
            os.mkdir(targ_dir)
        if not os.path.exists(f"{targ_dir}/{gamma}"):
            os.mkdir(f"{targ_dir}/{gamma}")
        if not os.path.exists(f"{targ_dir}/{gamma}/time"):
            os.mkdir(f"{targ_dir}/{gamma}/time")
        
        
        values = plot_discount(gamma=gamma, date=benchmark_date, 
                               min_trajectory_length=min_trajectory_length,
                               path = f"/mnt/hdd2/fabian/f1tenth_dope/ws_ope/f1tenth_orl_dataset/data/{target}",
                               target=target)


        print(values)
        plot_bars(values, target=target, gamma=gamma)
        process_values(values, path=f"{targ_dir}/{gamma}") # compute mse, bestk@1, and rank correlation
        # exit()
        # exit()