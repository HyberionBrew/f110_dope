import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

paths = ["trajectories_td_prog.zarr", "trajectories_raceline.zarr", "trajectories_min_act.zarr"]
# preppend plots to paths
paths = ["plots2/" + path for path in paths]
discounts = [0.95, 0.99]

keys = ['DR', 'MB', 'FQE']
for path in paths:

    # Prepare an empty matrix for the results
    data_matrix = []

    for discount in discounts:
        # Construct the full path to the .pkl file
        full_path = f"{path}/{discount}/spearman_per_model.pkl"
        
        with open(full_path, 'rb') as f:
            spearman_data = pickle.load(f)
        
        # Extract values for models 0-3
        model_values = [spearman_data[key] for key in keys]
        data_matrix.append(model_values)

    data_matrix = np.array(data_matrix)

    # Plotting
    fig, ax = plt.subplots()

    cmap = colors.ListedColormap(['red', 'green'])
    bounds = [-1.0, 0.6, 1.1]  # Adjusting upper bound to include all values >0.5
    norm = colors.BoundaryNorm(bounds, cmap.N)

    cax = ax.matshow(data_matrix, cmap=cmap, norm=norm)

    # Annotating the actual values
    for i in range(len(discounts)):
        for j in range(len(spearman_data.keys())):
            ax.text(j, i, f"{data_matrix[i, j]:.2f}", va='center', ha='center', color='black' if data_matrix[i, j] < 0.7 else 'white')

    ax.set_xticks(np.arange(len(spearman_data.keys())))
    ax.set_yticks(np.arange(len(discounts)))

    ax.set_xticklabels([str(key) for key in keys])
    ax.set_yticklabels(discounts)

    ax.set_xlabel('Models')
    ax.set_ylabel('Discounts')

    plt.colorbar(cax, ticks=[0, 0.5, 1.5])

    plt.show()
    plt.savefig(f"{path}/spearman.png")
    # clear figure
    plt.clf()
