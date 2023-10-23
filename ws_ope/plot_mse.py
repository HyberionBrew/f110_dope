import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

paths = ["plots/trajectories_td_prog", "plots/trajectories_raceline", "plots/trajectories_min_act"]
discounts = [0.85, 0.95, 0.99]
keys_main = ['DR', 'MB', 'FQE']
cmap = colors.LinearSegmentedColormap.from_list("mse_colormap", ['white', 'red'])
sub_keys = ["min_action_weight","progress_weight", "raceline_delta_weigh"]
for path in paths:
    # Prepare the data
    mse_data = {key: [] for key in keys_main}

    for discount in discounts:
        # Load the data
        with open(f"{path}/{discount}/mse.pkl", 'rb') as f:
            mse = pickle.load(f)

        # Organize the data
        for key in keys_main:
            sub_dict_values = [mse[key][sub_key] for sub_key in sub_keys]
            mse_data[key].append(sub_dict_values)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, key in zip(axs, keys_main):
        # Convert data for the current key to an array
        data = np.array(mse_data[key])
        cax = ax.matshow(data, cmap=cmap, vmin=0, vmax=0.2)

        # Annotating cells with their values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="black")

        ax.set_xticks(np.arange(len(mse[key].keys())))
        ax.set_yticks(np.arange(len(discounts)))

        ax.set_xticklabels([key_name[:-6] for key_name in sub_keys])
        ax.set_yticklabels(discounts)

        ax.set_title(key)
        ax.set_xlabel('Models')
        ax.set_ylabel('Discounts')
        plt.colorbar(cax, ax=ax, ticks=[0, 1], label='MSE')
    plt.suptitle(f"Aboslute Error (target: {path[6:]})")
    plt.tight_layout()
    plt.savefig(f"{path}/mse_comparison.png")
    plt.clf()
