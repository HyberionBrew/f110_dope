import os
from tensorflow.python.summary.summary_iterator import summary_iterator
import struct
import sys
import io
import os


class SuppressPrints:
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout

# Function to extract target_policy from folder name
def get_target_policy(folder_name):
    for part in folder_name.split(","):
        key, value = part.split("=")
        if key == "target_policy":
            return value
    return None


def get_dict(logdir, tag="train/pred returns"):
    # Dictionary to store returns
    all_returns = {}
    all_folders = get_folders(logdir)
    for folder in all_folders:
        dir_path = os.path.join(logdir, folder)
        
        # We'll assume there's only one file per directory as in your code. 
        # If there are multiple files, this will take the first file found.
        file_path = os.path.join(dir_path, os.listdir(dir_path)[0])
        
        target_policy = get_target_policy(folder)
        
        returns = []
        for e in summary_iterator(file_path):    
            for v in e.summary.value:
                if v.tag == tag:            
                    returns.append(struct.unpack('f', v.tensor.tensor_content)[0])
        
        # Append returns to the respective key in the dictionary
        if target_policy in all_returns:
            all_returns[target_policy].append(returns)
        else:
            all_returns[target_policy] = [returns]
    return all_returns

def get_folders(logdir):
    return [d for d in os.listdir(logdir) if os.path.isdir(os.path.join(logdir, d))]


import numpy as np
def compute_statistics(data):
    """
    Compute mean and standard deviation for each list across the outer list.
    
    Args:
    - data (list of lists): Input data where inner lists contain numbers.

    Returns:
    - means (list): Mean of each list across the outer list.
    - std_devs (list): Standard deviation of each list across the outer list.
    """
    # Convert to numpy array for easier calculations
    # remove all lists that are not the same length
    max_length = max([len(l) for l in data])
    data = [l for l in data if len(l) == max_length]

    np_data = np.array(data)
    
    #print(np_data.shape)
    #print(np_data)
    # If the inner lists aren't of equal length, this will raise a ValueError
    means = np_data.mean(axis=0).tolist()
    std_devs = np_data.std(axis=0).tolist()
    
    return means, std_devs


def compute_means_std(policy_returns):
    computed_means = {}
    computed_std_devs = {}
    for target_policy, returns in policy_returns.items():
        means, std_devs = compute_statistics(returns)
        computed_means[target_policy] = means
        computed_std_devs[target_policy] = std_devs
    for target_policy in computed_means.keys():
        computed_means[target_policy] = np.mean(np.array(computed_means[target_policy])[len(computed_means[target_policy])//2:])
        computed_std_devs[target_policy] = np.mean(np.array(computed_std_devs[target_policy][len(computed_std_devs[target_policy])//2:]))
    return computed_means, computed_std_devs

import matplotlib.pyplot as plt

def plot_data(policy, means, std_devs):
    x = list(range(len(means)))
    
    
    plt.plot(x, means, label=f"{policy}", marker='o')
    plt.fill_between(x, np.array(means) - np.array(std_devs), 
                     np.array(means) + np.array(std_devs), color='gray', alpha=0.2)
    

def plot_time(returns, title="DR", mean_std=False, plt_path=None, gamma=None):
    plt.figure(figsize=(7, 4))
    computed_means = {}
    computed_std_devs = {}
    if mean_std:
        for target_policy, values in returns.items():
                # each values is a list of lists
                #get the max list length
                max_length = max([len(l) for l in values])
                # remove each list from values that is less than max_length
                values = [l for l in values if len(l) == max_length]
                #print(values)
                #print("---")
                means, std_devs = compute_statistics(values)
                computed_means[target_policy] = means
                computed_std_devs[target_policy] = std_devs
                plot_data(target_policy, means, std_devs)
    else:
        # Get the number of unique target policies
        num_policies = len(returns.keys())

        # Choose a colormap
        cmap = plt.get_cmap('viridis')

        # Generate a list of colors based on the number of unique target policies
        colors = [cmap(i) for i in np.linspace(0, 1, num_policies)]
        # Create a dictionary to store the color for each target_policy
        policy_colors = {policy: color for policy, color in zip(returns.keys(), colors)}

        for target_policy in returns.keys():
            for values in returns[target_policy]:
                # print(values)
                # values = values[len(values)//2:]
                # give all of the same target policy the same color

                plt.plot(values, label=f"{target_policy}", marker='o', color=policy_colors[target_policy])
                # max y value is 0.6 min is 0.45
                # plt.ylim(0.5, 0.65)
    plt.xlabel("Evaluation Step")
    plt.ylabel("Reward")
    # in the legend only print each label once
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    #plt.legend()
    plt.title(f"Reward vs. Evaluation Steps during Training \n {plt_path} " + title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plt_path}/time/{title}{'_mean' if mean_std else ''}.png")
    plt.show()

import f110_gym
import f110_orl_dataset
import gymnasium as gym
import numpy as np

def get_change_indices(model_names):
    change_indices = []
    current_name = model_names[0]
    for idx, name in enumerate(model_names):
        if name != current_name:
            change_indices.append(idx)
            current_name = name
    return change_indices

def count_collision(dataset):
    model_names = dataset["infos"]["model_name"]
    change_indices = get_change_indices(model_names) + [len(model_names)]

    # done_or_truncated = np.logical_or(done, truncated)
    # split into segments at done or truncated
    # change_indices = np.where(done_or_truncated)[0] + 1 
    print(dataset.keys())
    # collisions = datasets['collision']
    print("Number of collisions for each model:")
    # loop over models:
    start_idx = 0
    for idx, change_idx in enumerate(change_indices):
        # count number of collisions for this model
        num_dones = np.sum(dataset['timeouts'][start_idx:change_idx])
        num_collisions = np.sum(dataset['terminals'][start_idx:change_idx])
        print(f'{model_names[start_idx]}: {num_collisions}/{num_dones+num_collisions}')
        start_idx = change_idx

def calculate_discounted_reward(rewards, done, truncated,end=0, gamma=0.99 ,only_consider_full_trajectories=False):
    # combine done and truncated
    done_or_truncated = np.logical_or(done, truncated)
    # split into segments at done or truncated
    change_indices = np.where(done_or_truncated)[0] + 1 # catch the last segment
    # ensure that we dont access out of bounds, by checking if we are at the end
    change_indices[-1] = min(change_indices[-1], len(rewards)-1)
    # calculate discounted reward for each segment
    start_idx = 0
    discounted_rewards = []
    for end_idx in change_indices:
        if end!=0:
            end_idx = min(start_idx + end, end_idx)
        segment_rewards = rewards[start_idx:end_idx]
        #print(len(segment_rewards))
        discounted_reward = np.sum(segment_rewards * gamma ** np.arange(len(segment_rewards)))
        # print(f'Discounted reward: {discounted_reward}')
        discounted_rewards.append(discounted_reward)
        start_idx = end_idx
    return np.mean(discounted_rewards), np.std(discounted_rewards)


def calculate_mean_reward(rewards, done, truncated, end=0, only_consider_full_trajectories=False):
    # combine done and truncated
    done_or_truncated = np.logical_or(done, truncated)
    # split into segments at done or truncated
    change_indices = np.where(done_or_truncated)[0] + 1 # catch the last segment
    # ensure that we dont access out of bounds, by checking if we are at the end
    change_indices[-1] = min(change_indices[-1], len(rewards)-1)
    # calculate discounted reward for each segment
    start_idx = 0
    discounted_rewards = []
    for end_idx in change_indices:
        if end!=0:
            end_idx = min(start_idx + end, end_idx)
        segment_rewards = rewards[start_idx:end_idx]
        #print(len(segment_rewards))
        discounted_reward = np.sum(segment_rewards * 1 ** np.arange(len(segment_rewards))) # / len(segment_rewards)
        # print(f'Discounted reward: {discounted_reward}')
        discounted_rewards.append(discounted_reward)
        start_idx = end_idx
    return np.mean(discounted_rewards), np.std(discounted_rewards)

def compute_mean_rewards_interval(inital=50, 
                                  timesteps = 50, 
                                  alternate_reward=True,
                                  path="/mnt/hdd2/fabian/f1tenth_dope/ws_ope/f1tenth_orl_dataset/data/trajectories.zarr"):
    F110Env = gym.make('f110_with_dataset-v0',
        # only terminals are available as of tight now 
            **dict(name='f110_with_dataset-v0',
                config = dict(map="Infsaal", num_agents=1),
                render_mode="human")
        )
    root = F110Env.get_dataset(zarr_path=path,
                            skip_inital=inital,
                            split_trajectories=False,
                            alternate_reward=alternate_reward,
                            remove_short_trajectories=False,
                            min_trajectory_length=600,

                            )
    model_names = root["infos"]["model_name"]
    print(len(model_names))
    change_indices = get_change_indices(model_names) + [len(model_names)]
    print(change_indices)

    change_indices = np.array(change_indices)-1
    # for each model calculate the mean discounted reward
    start_idx = 0
    #print("Discounted TD Reward")
    #print(change_indices)
    names = model_names[change_indices]
    #print(names)
    gamma = 1.0
    precomputed_returns = { name: [] for name in names}
    precomputed_stds = { name: [] for name in names}

    precomputed_mean_returns = { name: [] for name in names}
    precomputed_mean_stds = { name: [] for name in names}
    #print(precomputed_returns.keys())
    means = []
    stds = []
    # model_names_.append(model_names[start_idx])
    start_idx = 0
    change_indices = np.array(change_indices)+1
    start_idx = 0
    # calculate mean rewards
    for idx, change_idx in enumerate(change_indices):
        mean_reward_episodes, std_episodes = calculate_mean_reward(root['rewards'][start_idx:change_idx],
                                    root['terminals'][start_idx:change_idx],
                                    root['timeouts'][start_idx:change_idx],
                                    end = timesteps,
                                    only_consider_full_trajectories=True,)
        #mean_reward = np.mean(root['rewards'][start_idx:change_idx]) #/len(root['rewards'][start_idx:change_idx]))
        #std_reward = np.std(root['rewards'][start_idx:change_idx]) #/len(root['rewards'][start_idx:change_idx]))
        print(mean_reward_episodes, std_episodes)
        precomputed_mean_returns[model_names[start_idx]] = mean_reward_episodes
        precomputed_mean_stds[model_names[start_idx]]  = std_episodes
        start_idx = change_idx
    return precomputed_mean_returns, precomputed_mean_stds , root
def get_data(skip_inital=0, split_trajectories=0, 
            max_timesteps=50,
            alternate_reward=False, 
            remove_short_trajectories=False,
            without_agents = [],
            only_consider_full_trajectories = False,
            min_trajectory_length = 0,
            path = "/mnt/hdd2/fabian/f1tenth_dope/ws_ope/f1tenth_orl_dataset/data/trajectories.zarr",
            ):

    # import gymnasium as gym
    F110Env = gym.make('f110_with_dataset-v0',
        # only terminals are available as of tight now 
            **dict(name='f110_with_dataset-v0',
                config = dict(map="Infsaal", num_agents=1),
                render_mode="human")
        )
    root = F110Env.get_dataset(zarr_path=path,
                            skip_inital=skip_inital,
                            split_trajectories=split_trajectories,
                            alternate_reward=alternate_reward,
                            remove_short_trajectories=remove_short_trajectories,
                            min_trajectory_length=min_trajectory_length,
                            )
    return root
def compute_target_rewards(gamma = 0.85, 
                           skip_inital=0, split_trajectories=0, 
                           max_timesteps=50,
                           alternate_reward=False, 
                           remove_short_trajectories=False,
                           without_agents = [],
                           only_consider_full_trajectories = False,
                           min_trajectory_length = 0,
                           path = "/mnt/hdd2/fabian/f1tenth_dope/ws_ope/f1tenth_orl_dataset/data/trajectories.zarr",
                           ):

    # import gymnasium as gym
    F110Env = gym.make('f110_with_dataset-v0',
        # only terminals are available as of tight now 
            **dict(name='f110_with_dataset-v0',
                config = dict(map="Infsaal", num_agents=1),
                render_mode="human")
        )
    root = F110Env.get_dataset(zarr_path=path,
                            skip_inital=skip_inital,
                            split_trajectories=split_trajectories,
                            alternate_reward=alternate_reward,
                            remove_short_trajectories=remove_short_trajectories,
                            min_trajectory_length=min_trajectory_length,
                            )

    model_names = root["infos"]["model_name"]
    print(len(model_names))
    change_indices = get_change_indices(model_names) + [len(model_names)]
    print(change_indices)

    change_indices = np.array(change_indices)-1
    # for each model calculate the mean discounted reward
    start_idx = 0
    #print("Discounted TD Reward")
    #print(change_indices)
    names = model_names[change_indices]
    #print(names)
    gamma = gamma
    precomputed_returns = { name: [] for name in names}
    precomputed_stds = { name: [] for name in names}

    precomputed_mean_returns = { name: [] for name in names}
    precomputed_mean_stds = { name: [] for name in names}
    #print(precomputed_returns.keys())
    means = []
    stds = []
    # model_names_.append(model_names[start_idx])
    start_idx = 0
    change_indices = np.array(change_indices)+1
    #print(change_indices)
    for idx, change_idx in enumerate(change_indices): # this is for each model
        # count number of collisions for this model
        #try:
        # print(start_idx, change_idx)
        mean_discounted_reward, std_discounted = calculate_discounted_reward(root['rewards'][start_idx:change_idx],
                                    root['terminals'][start_idx:change_idx],
                                    root['timeouts'][start_idx:change_idx], gamma=gamma,
                                    # end=max_timesteps,
                                    only_consider_full_trajectories=only_consider_full_trajectories,)
        # print(f'{model_names[start_idx]}: {mean_discounted_reward} +- {std_discounted} ({mean_discounted_reward* (1-gamma)} +- {std_discounted * (1-gamma)})')
        means.append(mean_discounted_reward) #* (1-gamma))
        stds.append(std_discounted* (1-gamma))
        precomputed_returns[model_names[start_idx]] = mean_discounted_reward* (1-gamma)
        precomputed_stds[model_names[start_idx]] = std_discounted* (1-gamma)
        start_idx = change_idx
            
        #except:
        #    print(0)
    #print(precomputed_returns)
    start_idx = 0
    # calculate mean rewards
    for idx, change_idx in enumerate(change_indices):
        mean_reward_episodes, std_episodes = calculate_mean_reward(root['rewards'][start_idx:change_idx],
                                    root['terminals'][start_idx:change_idx],
                                    root['timeouts'][start_idx:change_idx],
                                    end=max_timesteps,
                                    only_consider_full_trajectories=only_consider_full_trajectories,)
        #mean_reward = np.mean(root['rewards'][start_idx:change_idx]) #/len(root['rewards'][start_idx:change_idx]))
        #std_reward = np.std(root['rewards'][start_idx:change_idx]) #/len(root['rewards'][start_idx:change_idx]))
        precomputed_mean_returns[model_names[start_idx]] = mean_reward_episodes
        precomputed_mean_stds[model_names[start_idx]]  = std_episodes
        start_idx = change_idx

    return precomputed_returns, precomputed_stds, precomputed_mean_returns, precomputed_mean_stds, root

def plot_reward_vs_discount(gammas, alternate_reward, min_trajectory_length=0, skip_inital=0):
    for gamma in gammas:
        
        with SuppressPrints():
            precomputed_returns, precomputed_stds, mean_precomputed, std_precomputed, dataset = compute_target_rewards(
                                    gamma=gamma, 
                                    skip_inital=skip_inital, 
                                    # split_trajectories=100,
                                    max_timesteps=0,
                                    alternate_reward=alternate_reward, 
                                    remove_short_trajectories=False,
                                    # without_agents=["raceline", "min_lida"]
                                    min_trajectory_length=min_trajectory_length,
                                    )
        plot_bar_precomputed(precomputed_returns, precomputed_stds, name = "gamma="+str(gamma))
    plot_bar_precomputed(mean_precomputed, std_precomputed, name = "Mean Rewards")


def compute_trajectory_lengths(done, truncated):
    done_or_truncated = np.logical_or(done, truncated)
    # Add an ending point to ensure the last trajectory is considered
    # done_or_truncated = np.append(done_or_truncated, [1])
    lengths = []
    count = 0
    for flag in done_or_truncated:
        if flag:
            lengths.append(count + 1)
            count = 0
        else:
            count += 1
    return lengths

def dataset_statistics(root, path=None):
    done = root['terminals']
    truncated = root['timeouts']

    start_idx = 0

    # Lists for plotting
    means = []
    std_devs = []
    mins = []
    maxs = []
    model_names = root["infos"]["model_name"]
    change_indices = get_change_indices(model_names) + [len(model_names)]
    
    print("Statistics for each model:")
    total_lengths = []
    for idx, change_idx in enumerate(change_indices):
        current_lengths = compute_trajectory_lengths(done[start_idx:change_idx], truncated[start_idx:change_idx])

        mean_length = np.mean(current_lengths)
        std_dev = np.std(current_lengths)
        min_length = np.min(current_lengths)
        max_length = np.max(current_lengths)
        total_length = np.sum(current_lengths)
        #print(f"{model_names[start_idx]}:")
        #print(f"\tMean Length: {mean_length}")
        #print(f"\tStandard Deviation: {std_dev}")
        #print(f"\tMin Length: {min_length}")
        #print(f"\tMax Length: {max_length}\n")
        #print(f"\tTotal Length: {total_length}\n")
        # Add values to lists for plotting
        means.append(mean_length)
        std_devs.append(std_dev)
        mins.append(min_length)
        maxs.append(max_length)
        total_lengths.append(total_length)
        start_idx = change_idx

    # Plotting
    model_names_ = [model_names[i-1] for i in change_indices]
    x = np.arange(len(np.unique(model_names)))
    #print(len(means))
    #print(len(std_devs))
    #print(len(x))
    plt.bar(x, means, yerr=std_devs, label='Mean with STD', alpha=0.7)
    plt.scatter(x, mins, color='red', label='Min Length', zorder=3)
    plt.scatter(x, maxs, color='green', label='Max Length', zorder=3)
    
    plt.xlabel('Model')
    plt.ylabel('Trajectory Length')
    plt.title('Trajectory Length Statistics for Each Model')
    plt.xticks(x, model_names_, rotation=45)
    plt.legend()
    plt.tight_layout()
    if path is not None:
        print("Saved figure!")
        plt.savefig(path + "/trajectory_length_statistics.png")

    plt.show()
    # clear plt
    plt.clf()
    # bar chart with total_lengths
    plt.bar(x, total_lengths, label='Total Length', alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('# of Transitions')
    plt.title('Total # of Transitions present in the dataset for each model')
    plt.xticks(x, model_names_, rotation=45)
    plt.legend()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path + "/total_trajectory_length.png")
    plt.show()

def plot_bar_precomputed(precomputed_returns, precomputed_stds, name="DR", keys= None, plt_path=None, gamma=None):
    # Assuming the order of precomputed_returns as the standard order
    if keys is None:
        keys = list(precomputed_returns.keys())
    precomputed_means_values = [precomputed_returns[key] for key in keys]
    precomputed_std_values = [precomputed_stds[key] for key in keys]

    bar_width = 0.7
    index = np.arange(len(keys))
    
    fig, ax = plt.subplots(figsize=(5, 4))
    bar1 = ax.bar(index, precomputed_means_values, bar_width, yerr=precomputed_std_values, label='Precomputed Means', alpha=0.8, capsize=7)
    
    ax.set_xlabel('Agent')
    ax.set_ylabel(f"Discounted Reward ")
    ax.set_title(f"Means and Std Deviations: {name} \n {gamma}")
    ax.set_xticks(index)
    ax.set_xticklabels(keys, rotation=45)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    if plt_path is not None:
        plt.savefig(plt_path + "/precomputed_gt.png")
    plt.show()

def plot_multiple_models(precomputed_returns, precomputed_stds, models_dict, name="DR", plt_path=None, gamma=None):
    keys = list(precomputed_returns.keys())
    
    precomputed_means_values = [precomputed_returns[key] for key in keys]
    precomputed_std_values = [precomputed_stds[key] for key in keys]
    
    num_models = len(models_dict) + 1  # +1 for the precomputed
    bar_width = 0.35 / num_models  # Adjusting bar width based on the number of models
    
    index = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plotting bars for precomputed values
    ax.bar(index, precomputed_means_values, bar_width, yerr=precomputed_std_values, label='Precomputed', alpha=0.8, capsize=7)
    
    # Plotting bars for additional models
    for i, (model_name, model_values) in enumerate(models_dict.items()):
        model_means_values = [model_values['means'][key] for key in keys]
        model_std_values = [model_values['std_devs'][key] for key in keys]
        ax.bar(index + (i+1)*bar_width, model_means_values, bar_width, yerr=model_std_values, label=model_name, alpha=0.8, capsize=7)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Model')
    ax.set_ylabel('Discounted Reward')
    ax.set_title(f"Means and Std Deviations: {name} \n {gamma} \n {plt_path}")
    ax.set_xticks(index + bar_width * num_models / 2)  # Adjusting x-ticks based on the number of models
    ax.set_xticklabels(keys, rotation=45)
    ax.legend()
    plt.tight_layout()
    if plt_path is not None:
        plt.savefig(plt_path + "/multiple_models.png")
    plt.show()



def plot_bar_vs(precomputed_returns, precomputed_stds, computed_means, computed_std_devs ,name = "DR", plt_path=None, gamma=None):
    # Assuming the order of precomputed_returns as the standard order
    
    keys = list(precomputed_returns.keys())
    precomputed_means_values = [precomputed_returns[key] for key in keys]
    precomputed_std_values = [precomputed_stds[key] for key in keys]
    computed_means_values = [computed_means[key] for key in keys]
    computed_std_values = [computed_std_devs[key] for key in keys]
    print(computed_means_values)
    bar_width = 0.35
    index = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(12, 7))
    bar1 = ax.bar(index, precomputed_means_values, bar_width, yerr=precomputed_std_values, label='Precomputed Means', alpha=0.8, capsize=7)
    bar2 = ax.bar(index + bar_width, computed_means_values, bar_width, yerr=computed_std_values, label='Computed Means', alpha=0.8, capsize=7)

    ax.set_xlabel('Model')
    ax.set_ylabel('Discounted Returns')
    ax.set_title(f"Means and Std Deviations: {name} ({gamma})")
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(keys, rotation=45)
    ax.legend()
    # plt.title("Precomputed vs. Computed Means and Std Deviations")
    plt.tight_layout()
    if plt_path is not None:
        plt.savefig(plt_path + "/precomputed_vs_computed.png")
    plt.show()

def plot_dataset_statistics(dataset, path="plots"): #, precomputed_returns, precomputed_stds):
    print("DATASET STATISTICS")
    
    dataset_statistics(dataset, path=path)

    model_names = dataset["infos"]["model_name"]
    change_indices = get_change_indices(model_names) + [len(model_names)]
    model_names_ = [model_names[i-1] for i in change_indices]
    print(model_names_)
    #print(precomputed_returns.keys())
    # plot_bar_precomputed(precomputed_returns, precomputed_stds, "Precomputed" ,keys=model_names_)

def plot_everything(logdir,
                    precomputed_returns,
                    precomputed_stds,
                    dataset,
                    logdir_mb=None,
                    plt_path=None,
                    gamma= None):

    # read the returns from the logdir
    returns_dr = get_dict(logdir, tag="train/pred returns")
    returns_fqe = get_dict(logdir, tag="train/pred returns (fqe)")
    std_fqe = get_dict(logdir, tag="std_deviation returns (fqe)")
    returns_mb = get_dict(logdir_mb, tag="train/pred returns")
    print(returns_dr)
    #print(f"# of runs recorded: {len(returns_dr['raceline']) }" )
    #print(f"# of runs recorded: {len(returns_dr['min_lida']) }" )

    # time series plots
    print(returns_dr)
    plot_time(returns_dr, mean_std=True , plt_path=plt_path, gamma=gamma)
    plot_time(returns_dr, mean_std=False , plt_path=plt_path, gamma=gamma)

    plot_time(returns_fqe, mean_std=True,title="FQE"  , plt_path=plt_path, gamma=gamma)
    plot_time(returns_fqe, mean_std=False, title="FQE" , plt_path=plt_path, gamma=gamma)

    plot_time(returns_mb, mean_std=True,title="MB"  , plt_path=plt_path, gamma=gamma)
    plot_time(returns_mb, mean_std=False, title="MB" , plt_path=plt_path, gamma=gamma)

    returns_dr_means, returns_dr_std = compute_means_std(returns_dr)
    std_fqe_means, std_fqe_std = compute_means_std(std_fqe)
    mean_fqe, mean_fqe_std = compute_means_std(returns_fqe)
    
    mean_mb, mean_mb_std = compute_means_std(returns_mb)
    print("DATASET STATISTICS")
    dataset_statistics(dataset)
    plot_bar_precomputed(precomputed_returns, precomputed_stds, "Precomputed", plt_path=plt_path, gamma=gamma)
    # from precomputed_returns remove all keys that are not in returns_dr_means
    for key in list(precomputed_returns.keys()):
        if key not in returns_dr_means.keys():
            del precomputed_returns[key]
            del precomputed_stds[key]

    plot_bar_vs(precomputed_returns, precomputed_stds,returns_dr_means,returns_dr_std , plt_path=plt_path, gamma=gamma)
    plot_bar_vs(precomputed_returns, precomputed_stds,mean_fqe,std_fqe_means, name="FQE",plt_path=plt_path, gamma=gamma)
    plot_bar_vs(precomputed_returns, precomputed_stds,mean_mb,mean_mb_std, name="MB",plt_path=plt_path, gamma=gamma)
    # create dict for plot_multiple_models
    models_dict = {
        "DR": {
            "means": returns_dr_means,
            "std_devs": returns_dr_std,
        },
        "FQE": {
            "means": mean_fqe,
            "std_devs": std_fqe_means,
        },
        "MB": {
            "means": mean_mb,
            "std_devs": mean_mb_std,
        } 
    }
    
    plot_multiple_models(precomputed_returns, precomputed_stds, models_dict, name="DR, FQE and MB", plt_path=plt_path, gamma=gamma)
    return models_dict, precomputed_returns, precomputed_stds


    
def plot_mb_mean(logdir,
                    precomputed_returns,
                    precomputed_stds,
                    dataset,
                    logdir_mb=None):


    retruns_std = get_dict(logdir, tag="train/pred std returns (MB)")
    returns_mb = get_dict(logdir, tag="train/pred mean returns (MB)")
    # print(returns_dr)
    #print(f"# of runs recorded: {len(returns_dr['raceline']) }" )
    #print(f"# of runs recorded: {len(returns_dr['min_lida']) }" )

    # time series plots
    plot_time(returns_mb, mean_std=True,title="MB" )
    plot_time(returns_mb, mean_std=False, title="MB")

    std_mb, std_fqe_std = compute_means_std(retruns_std)
    mean_mb, mean_fqe_std = compute_means_std(returns_mb)
    
    mean_mb, mean_mb_std = compute_means_std(returns_mb)
    print("DATASET STATISTICS")
    dataset_statistics(dataset)
    plot_bar_precomputed(precomputed_returns, precomputed_stds, "Precomputed")
    # from precomputed_returns remove all keys that are not in returns_dr_means
    for key in list(precomputed_returns.keys()):
        if key not in returns_mb.keys():
            del precomputed_returns[key]
            del precomputed_stds[key]


    plot_bar_vs(precomputed_returns, precomputed_stds,mean_mb,std_mb, name="MB")
    # create dict for plot_multiple_models
    models_dict = {
        "MB": {
            "means": mean_mb,
            "std_devs": mean_mb_std,
        } 
    }
    
    plot_multiple_models(precomputed_returns, precomputed_stds, models_dict, name="DR and FQE")