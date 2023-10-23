import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

paths = ["trajectories_td_prog", "trajectories_raceline", "trajectories_min_act"]
# preppend plots to paths
paths = ["plots/" + path for path in paths]
discounts = [0.85, 0.95, 0.99]

for path in paths:

    # Prepare an empty matrix for the results
    data_matrix = []

    for discount in discounts:
        # Construct the full path to the .pkl file
        full_path = f"{path}/{discount}/bestk.pkl"
        
        with open(full_path, 'rb') as f:
            bestk = pickle.load(f)
        
        # Extract values for models 0-3
        print(bestk)
        model_values = [bestk[key] for key in bestk.keys()]
        data_matrix.append(model_values)

    data_matrix = np.array(data_matrix)

    # Plotting
    fig, ax = plt.subplots()

    cmap = colors.ListedColormap(['red', 'green'])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    cax = ax.matshow(data_matrix, cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(len(discounts)))

    ax.set_xticklabels([str(key) for key in bestk.keys()])
    ax.set_yticklabels(discounts)

    ax.set_xlabel('Models')
    ax.set_ylabel('Discounts')
    plt.title(f"Best k@1 ({path[6:]})")

    plt.colorbar(cax, ticks=[0, 1])

    plt.show()
    plt.savefig(f"{path}/bestk.png")
    #clear figure
    plt.clf()