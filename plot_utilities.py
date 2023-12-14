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


def get_dict_from_tf(logdir, tag="train/pred returns"):
    returns = []
    #print("-----")
    #print(logdir)
    # discover the file in the folder
    run = os.listdir(logdir)[0]
    logdir = os.path.join(logdir,run)

    for e in summary_iterator(logdir):
        for v in e.summary.value:
            if v.tag == tag:
                content = v.tensor.tensor_content
                #print(len(content))
                if len(content)==0:
                    if hasattr(v, 'simple_value'):
                        returns.append(float(v.simple_value))
                    else:
                        # fallback to 0
                        returns.append(0.0)
                elif len(content)==4:
                    returns.append(struct.unpack('f', content)[0])
                elif len(content)==8:
                    returns.append(float(struct.unpack('d', content)[0]))
    return returns

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


def compute_returns(dataset, 
                   gamma=0.85):
    model_names = dataset["infos"]["model_name"]
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
        mean_discounted_reward, std_discounted = calculate_discounted_reward(dataset['rewards'][start_idx:change_idx],
                                    dataset['terminals'][start_idx:change_idx],
                                    dataset['timeouts'][start_idx:change_idx], gamma=gamma,
                                    # end=max_timesteps,
                                    only_consider_full_trajectories=False)
        # print(f'{model_names[start_idx]}: {mean_discounted_reward} +- {std_discounted} ({mean_discounted_reward* (1-gamma)} +- {std_discounted * (1-gamma)})')
        means.append(mean_discounted_reward) #* (1-gamma))
        stds.append(std_discounted* (1-gamma))
        precomputed_returns[model_names[start_idx]] = mean_discounted_reward* (1-gamma)
        precomputed_stds[model_names[start_idx]] = std_discounted* (1-gamma)
        start_idx = change_idx
            
    return precomputed_returns, precomputed_stds

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

def plot_bar_dict(mean_std_dict: dict, name="DR", key_order= None, plt_path=None):
    assert 'mean' and 'std' in mean_std_dict
    # Assuming the order of precomputed_returns as the standard order
    if key_order is None:
        keys = list(mean_std_dict['mean'].keys())
    else:
        keys = key_order
    precomputed_means_values = [mean_std_dict['mean'][key] for key in keys]
    precomputed_std_values = [mean_std_dict['std'][key] for key in keys]

    bar_width = 0.7
    index = np.arange(len(keys))
    
    fig, ax = plt.subplots(figsize=(5, 4))
    bar1 = ax.bar(index, precomputed_means_values, bar_width, yerr=precomputed_std_values, label='Precomputed Means', alpha=0.8, capsize=7)
    
    ax.set_xlabel('Agent')
    ax.set_ylabel(f"Discounted Reward ")
    ax.set_title(f"Means and Std Deviations: {name} \n")
    ax.set_xticks(index)
    ax.set_xticklabels(keys, rotation=45)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    if plt_path is not None:
        plt.savefig(plt_path + "/precomputed_gt.png")
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

def traverse_logdir(logdir,method='dr', tag_pred='train/pred returns',tag_std='train/pred stds', minimum_evals=3):
    dict = {}
    # first append the method to the logdir
    logdir = os.path.join(logdir, method)
    # now lets check the target rewards available
    target_rewards = os.listdir(logdir)
    # for each target reward we now descent into the directory
    for reward in target_rewards:
        dict[reward] = {}
        # now we check the available lengths
        for length in os.listdir(os.path.join(logdir, reward)):
            dict[reward][length] = {}
            for agent in os.listdir(os.path.join(logdir, reward, length)):
                dict[reward][length][agent] = {'mean': [], 'std': []}
                # now we check the available runs
                isFilled = False
                for run in os.listdir(os.path.join(logdir, reward, length, agent)):
                    # each of the runs is a folder, that we have to descent into and hand 
                    # over to get dict from tf
                    folder_path = os.path.join(logdir, reward, length, agent, run)
                    # make sure to only use folders not files
                    if not os.path.isdir(folder_path):
                        continue
                    data = pu.get_dict_from_tf(folder_path, tag=tag_pred)
                    data_std = pu.get_dict_from_tf(folder_path, tag=tag_std)
                    # print(len(data))
                    if len(data) >= minimum_evals:
                        dict[reward][length][agent]['mean'].append(data)
                        dict[reward][length][agent]['std'].append(data_std)
                        isFilled = True
                if not isFilled:
                    dict[reward][length].pop(agent)
    return dict

def post_process_rewards(rewards_dict):
    for method in rewards_dict.keys():
        if method=="ground_truth":
            continue
        for target in rewards_dict[method].keys():
            for timestep in rewards_dict[method][target].keys():
                for agent in rewards_dict[method][target][str(timestep)].keys():
                    unprocessed_means = rewards_dict[method][target][str(timestep)][agent]['mean']
                    unprocessed_stds = rewards_dict[method][target][str(timestep)][agent]['std']
                    processed_means = []
                    processed_stds = []
                    if len(unprocessed_means) == 0:
                        # del rewards_dict[method][target][str(timestep)][agent]
                        continue
                    if unprocessed_means[0] == []:
                        continue
                    for mean in unprocessed_means:
                        if len(mean) == 0:
                            processed_means.append(0)
                        else:
                            processed_means.append(mean[-1])

                    processed_means = np.array(processed_means)
                    rewards_dict[method][target][str(timestep)][agent]['mean'] = np.mean(processed_means)
                    processed_means_std = np.std(processed_means)
                    stds_zero =False
                    for std in unprocessed_stds:
                        if len(std) == 0:
                            stds_zero = True
                            break
                        processed_stds.append(std[-1])
                    if stds_zero == True:
                        rewards_dict[method][target][str(timestep)][agent]['std'] = processed_means_std
                    else:
                        processed_stds = np.array(processed_stds)
                        rewards_dict[method][target][str(timestep)][agent]['std'] = np.mean(processed_stds)
    return rewards_dict

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


def plot_bars_from_dict(reward_dict, target, length, methods, sub_keys=["progress_weight"], path=None):
    print(methods)
    print(reward_dict.keys())
    assert np.all([method in reward_dict.keys() for method in methods])
    assert "ground_truth" in reward_dict.keys()
    keys_main = methods
    keys_sub = sub_keys

    bar_width = 0.15
    num_bars = len(keys_sub)

    r = np.arange(len(keys_main))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_bars))
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, method in enumerate(methods):
        print(reward_dict[method][target].keys())
        mtl = reward_dict[method][target][length]
        # print(mtl)
        for j, key_sub in enumerate(keys_sub):
            label = key_sub if i == 0 else None  # Set label only for the first main group to avoid label repetition in the legend
            try:
                mean = mtl[key_sub]["mean"]
                std = mtl[key_sub]["std"]
            except KeyError:
                mean = 0.0
                std = 0.0
                print("Not all values yet available")
            #print(mean)
            ax.bar(r[i] + j*bar_width - (bar_width*(num_bars-1)/2), 
                    mean, 
                    yerr=std, 
                    width=bar_width, 
                    label=label, alpha=0.8, 
                    capsize=7, color=colors[j])
    ax.set_xlabel('OPE Type')
    ax.set_ylabel('Discounted Reward')
    ax.set_title(f"Comparison of {reward_dict.keys()} \n (target: {target}, length: {length})")
    ax.set_xticks(r)
    ax.set_xticklabels(keys_main)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()

from scipy import stats

def extract_ordered_gt(all_rewards, target_rewards, target, methods):
    ordered_per_model = {}
    #print(all_rewards.keys())
    #print(target_rewards)
    #print(target)
    set_method_keys = False
    for length in all_rewards[methods[0]][target].keys():
        ordered_per_model[length] = {}
        for method in methods:
            gt = all_rewards['ground_truth'][target][length]
            gt_keys = gt.keys()
            method_rewards = all_rewards[method][target][length]
            if not(set_method_keys):
                method_keys = method_rewards.keys()
                set_method_keys = True
            #print(gt)
            #print(method_rewards)
            method_ordered = []
            gt_ordered = []
            for key in method_keys:
                gt_ordered.append(gt[key]["mean"])
                method_ordered.append(method_rewards[key]["mean"])
            ordered_per_model[length][method] = (gt_ordered, method_ordered)
            #print(f"keys ordered: ", method_keys)
            #print(f"gt ordered: ", gt_ordered)
            #print(f"method ordered: ", method_ordered)
    print(method_keys)
    return ordered_per_model, method_keys
            #res = stats.spearmanr(method_ordered, gt_ordered)
            #spearman_per_model[length][method] = float(res.correlation)

import matplotlib.colors as colors

def compute_data_matrix(all_rewards, target_rewards, target, methods, fn, length_order=["50","250","1000"]):
    ordered_dict, method_keys = extract_ordered_gt(all_rewards, target_rewards, target, methods)
    # print(ordered_dict)
    # print("----")
    data_matrix = []
    for length in length_order:
        data_row = []
        for method in methods:
            targets = ordered_dict[length][method][0]
            y = ordered_dict[length][method][1]
            res = fn(targets, y)
            data_row.append(res)
        data_matrix.append(data_row)
    data_matrix = np.array(data_matrix)
    return data_matrix, length_order

def spearmanr(targets, y):
    res = stats.spearmanr(y, targets)
    return float(res.correlation)
            # print(res.correlation.shape)
            # print(res.pvalue)

def plot_spearman(all_rewards, target_rewards, methods, lengths=["50","250","1000"]):
    assert 'ground_truth' in all_rewards.keys()
    for target in target_rewards:
        data_matrix, length_order = compute_data_matrix(all_rewards, 
                            target_rewards, 
                            target, 
                            methods, 
                            spearmanr, length_order=lengths)
        #print(data_matrix)
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = colors.ListedColormap(['red', 'green'])
        bounds = [-1.0, 0.6, 1.1]  # Adjusting upper bound to include all values >0.5
        norm = colors.BoundaryNorm(bounds, cmap.N)

        cax = ax.matshow(data_matrix, cmap=cmap, norm=norm)

        # Annotating the actual values
        for i in range(data_matrix.shape[0]):
            for j in range(len(methods)):
                ax.text(j, i, f"{data_matrix[i, j]:.2f}", va='center', ha='center', color='black' if data_matrix[i, j] < 0.7 else 'white')

        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(data_matrix.shape[0]))
        #print(data_matrix.shape)

        ax.set_xticklabels([str(key) for key in methods])
        #print(ordered_dict.keys())
        ax.set_yticklabels(length_order)

        ax.set_xlabel('Models')
        ax.set_ylabel('Discounts')
        # set title the target
        ax.set_title(f"Spearman Correlation: {target}")
        plt.colorbar(cax, ticks=[0, 0.5, 1.5])

        plt.show()
        # plt.savefig(f"{path}/spearman.png")
        # clear figure
        plt.clf()

def mse(targets, y):
    #print(targets)
    #print(y)
    res = np.sum(abs(np.array(targets) - np.array(y))/np.array(targets))/ len(targets)
    return res


def plot_mse(all_rewards, target_rewards, methods, lengths=["50","250","1000"]):
    assert 'ground_truth' in all_rewards.keys()
    for target in target_rewards:
        ordered_dict = extract_ordered_gt(all_rewards, target_rewards, target, methods)
        data_matrix, length_order = compute_data_matrix(all_rewards, 
                            target_rewards, 
                            target, 
                            methods, 
                            mse,
                            length_order=lengths)
        #print(data_matrix)
        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = colors.LinearSegmentedColormap.from_list('white_red', ['white', 'red'], N=256)
        max_error_bound = 0.4 #max(0.2, np.max(data_matrix))
        #bounds = [0, max_error_bound / 2, max_error_bound]
        norm = colors.PowerNorm(gamma=0.5, vmin=0, vmax=max_error_bound)

        cax = ax.matshow(data_matrix, cmap=cmap, norm=norm)


        # Annotating the actual values
        for i in range(len(length_order)):
            for j in range(len(methods)):
                ax.text(j, i, f"{data_matrix[i, j]:.2f}", va='center', ha='center', color='black')

        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(length_order)))

        ax.set_xticklabels([str(key) for key in methods])
        ax.set_yticklabels(length_order)

        ax.set_xlabel('Models')
        ax.set_ylabel('Discounts')
        ax.set_title(f"Relative error: {target}")
        plt.colorbar(cax)
        plt.show()
        plt.clf()

def k1(targets, y):
    # is one if the largest in targets is a the same spot as in y
    max_targets = np.argmax(targets)
    max_y = np.argmax(y)
    return int(max_targets == max_y)

def plot_k1(all_rewards, target_rewards, methods, lengths=["50","250","1000"]):
    assert 'ground_truth' in all_rewards.keys()
    for target in target_rewards:
        data_matrix, length_order = compute_data_matrix(all_rewards, 
                    target_rewards, 
                    target, 
                    methods, 
                    k1,
                    length_order=lengths)
        print(data_matrix)

        fig, ax = plt.subplots(figsize=(12, 7))
        cmap = colors.ListedColormap(['red', 'green'])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        cax = ax.matshow(data_matrix, cmap=cmap, norm=norm)

        # Annotating the actual values
        for i in range(len(length_order)):
            for j in range(len(methods)):
                ax.text(j, i, f"{data_matrix[i, j]:.2f}", va='center', ha='center', color='black')

        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(length_order)))

        ax.set_xticklabels([str(key) for key in methods])
        print(length_order)
        ax.set_yticklabels(length_order)

        ax.set_xlabel('Models')
        ax.set_ylabel('Discounts')
        ax.set_title(f"K@1: {target}")
        plt.colorbar(cax, ticks=[0, 1])
        plt.show()
        plt.clf()

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