from torch.utils.data import Dataset
import numpy as np
import torch

def compute_raw_action(state, action,normalize_fn,unnormalize_fn):
    state = unnormalize_fn(state)
    prev_action = state[..., 7:9]
    raw_action = prev_action + np.clip(action, -1, 1) * 0.05
    state[..., 7:9] = raw_action
    state = normalize_fn(state)
    return state[..., 7:9]

class F110Dataset(Dataset):
    def __init__(self, 
               d4rl_env,
               normalize_states = False,
               normalize_rewards = False,
               path=None,
               exclude_agents = [],
               only_agents = [],
               alternate_reward = False,
               state_mean=None,
               state_std = None,
               reward_mean = None,
               reward_std = None,
               include_timesteps_in_obs = False,
               only_terminals=False,
               clip_trajectory_length= None,):
        # Load the dataset
        d4rl_dataset = d4rl_env.get_dataset(
            zarr_path=path, 
            without_agents=exclude_agents, 
            only_agents=only_agents,
            alternate_reward=alternate_reward,
            include_timesteps_in_obs=include_timesteps_in_obs,
            only_terminals=only_terminals,
            clip_trajectory_length=clip_trajectory_length
        )
        # Assuming 'observations' and 'next_observations' are keys in the dataset
        #self.observations = self.data['observations']
        self.timestep_constant = None
        if 'timesteps_constant' in d4rl_dataset.keys():
            self.timestep_constant = d4rl_dataset['timesteps_constant']
        
        self.states = torch.from_numpy(d4rl_dataset['observations'].astype(np.float32))
        self.scans = torch.from_numpy(d4rl_dataset['scans'].astype(np.float32))
        self.actions = torch.from_numpy(d4rl_dataset['actions'].astype(np.float32))
        self.raw_actions = torch.from_numpy(d4rl_dataset['raw_actions'].astype(np.float32))
        self.rewards = torch.from_numpy(d4rl_dataset['rewards'].astype(np.float32))
        self.masks = torch.from_numpy(1.0 - d4rl_dataset['terminals'].astype(np.float32))
        self.log_probs = torch.from_numpy(d4rl_dataset['log_probs'].astype(np.float32))
        self.timesteps = torch.from_numpy(d4rl_dataset['timesteps'].astype(np.float32))
        
        # now we need to do next states and next scans
        # first check where timeout and where terminal
        finished = np.logical_or(d4rl_dataset['terminals'], d4rl_dataset['timeouts'])
        # rolled finished to the right by 1
        finished = torch.from_numpy(finished)
        self.mask_inital = torch.roll(finished, 1)
        assert(self.mask_inital[0] == True)
        # now lets loop over the [finished[i-1], finished[i]] and set the next states
        finished_indices = torch.where(finished)[0]
        start = 0
        self.states_next = torch.zeros_like(self.states)
        self.scans_next = torch.zeros_like(self.scans)
        # zeros like (len(finished_indices), obs_shape)
        self.initial_states = torch.zeros((len(finished_indices), self.states.shape[-1]))
        # unused inital_weights
        self.initial_weights = torch.ones(len(self.initial_states))
        for i, stop in enumerate(finished_indices):
            # append to dim 0
            next_states = torch.cat((self.states[start+1:stop+1], self.states[stop].unsqueeze(0)), dim=0)
            next_scans = torch.cat((self.scans[start+1:stop+1], self.scans[stop].unsqueeze(0)), dim=0)

            self.states_next[start:stop+1] = next_states
            self.scans_next[start:stop+1] = next_scans
            
            self.initial_states[i] = self.states[start]
            start = stop + 1
        print("initial states", self.initial_states.shape)
        # now perform intelligent normalization from the dataset
        if normalize_states == True:
            if state_mean is None:
                self.state_mean = torch.mean(self.states, axis=0)
                self.state_std = torch.std(self.states, axis=0)
            else:
                self.state_mean = state_mean
                self.state_std = state_std
            self.states = self.normalize_states(self.states)
            self.initial_states = self.normalize_states(self.initial_states)
            self.states_next = self.normalize_states(self.states_next)
        
        if normalize_rewards == True:
            # do this here
            if reward_mean is None:
                self.reward_mean = torch.mean(self.rewards)
                self.reward_std = torch.std(self.rewards)
            else:  
                self.reward_mean = reward_mean
                self.reward_std = reward_std

            self.reward_mean = torch.mean(self.rewards)
            self.reward_std = torch.std(self.rewards)
            self.rewards = self.normalize_rewards(self.rewards) 
        else:
            print("am i a joke?")
            self.reward_mean = 0.0
            self.reward_std = 1.0
    def normalize_states(self, states):
        # only normalize columns [0,1,4,5,6,7,8]
        states_return = states.clone()
        # if only 1d add a fake first dim
        if len(states.shape) == 1:
            states_return = np.expand_dims(states_return, axis=0)
        for i in range(states_return.shape[-1]): # #[0,1,4,5,6,7,8]:#
            states_return[...,i] = (states_return[...,i] - self.state_mean[i]) / max(self.state_std[i],1e-8)
        if len(states.shape) == 1:
            states_return = states_return[0]
        return states_return
    
    def normalize_rewards(self, rewards):
        return (rewards - self.reward_mean) / max(self.reward_std, 1e-8)
    
    def unnormalize_states(self, states, eps=1e-8):
        states_return = states.clone()
        # 2,3 are theta already between -1 and 1
        # 9,10 are progress already between -1 and 1
        # if only 1d add a fake first dim
        if len(states.shape) == 1:
            states_return = torch.expand_dims(states_return, axis=0)

        for i in range(states_return.shape[-1]): # [0,1,4,5,6,7,8]: #
            states_return[...,i] = states_return[...,i] * max(self.state_std[i],eps) + self.state_mean[i]
        # remove the fake dim
        if len(states.shape) == 1:
            states_return = states_return[0]
        return states_return
    
    def unnormalize_rewards(self, rewards):
        return rewards * self.reward_std + self.reward_mean

    def __len__(self):
        return len(self.states)

    # returns (states, scans, actions, next_states, next_scans, rewards, masks, weights,
    # log_prob, timesteps)
    def __getitem__(self, idx):
        current_state = self.states[idx]
        next_state = self.states_next[idx]
        action = self.actions[idx]
        scan = self.scans[idx]
        next_scan = self.scans_next[idx]
        reward = self.rewards[idx]
        mask = self.masks[idx]
        log_prob = self.log_probs[idx]
        timestep = self.timesteps[idx]
        # Include other components like actions, rewards, etc. if needed
        return current_state, scan, action, next_state, next_scan, reward, mask, 1.0, log_prob, timestep 

if __name__ == "__main__":
    import f110_gym
    import f110_orl_dataset
    import gymnasium as gym
    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of right now 
      **dict(name='f110_with_dataset-v0',
          config = dict(map="Infsaal", num_agents=1,
          params=dict(vmin=0.5, vmax=2.0)),
            render_mode="human")
        )
    env = F110Env
    dataset = F110Dataset(
            env,
            path = f"/home/fabian/msc/f110_dope/rollouts/f110-sb3/trajectories_new.zarr", #trajectories.zarr",
            normalize_rewards=True,
            normalize_states=True,
            only_agents = [], #['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
            alternate_reward=False,
            include_timesteps_in_obs = True,
    )

    # do some testing here