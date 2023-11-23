import numpy as np
from policy_eval_torch.dataset import F110Dataset
import torch 

class DR_estimator:
    def __init__(self, model, behavior_dataset: F110Dataset, discount):
        self.model = model
        self.behavior_dataset = behavior_dataset
        self.discount = discount
    
    def estimate_returns(self,get_target_actions, get_target_logprobs):
        with torch.no_grad():
            behavior_log_probs = self.behavior_dataset.log_probs
            # print the devices of the tensors
            print("behavior_log_probs", behavior_log_probs.device)
            print("behavior_dataset.states", self.behavior_dataset.states.device)
            print("behavior_dataset.actions", self.behavior_dataset.actions.device)
            print("behavior_dataset.scans", self.behavior_dataset.scans.device)
            # send all to cuda
            #behavior_log_probs = behavior_log_probs.to(self.model.device)

            target_log_probs = get_target_logprobs(self.behavior_dataset.states, 
                                                self.behavior_dataset.actions,
                                                scans = self.behavior_dataset.scans)
            ############# DR part #############
            offset, std_deviation = self.model.estimate_returns(self.behavior_dataset.initial_states,
                                                            self.behavior_dataset.initial_weights,
                                                            get_action=get_target_actions,)
            if False:
                q_values = self.model(self.behavior_dataset.states,
                                self.behavior_dataset.actions,
                                timesteps = self.behavior_dataset.timesteps)
            else:
                q_values = self.model.compute_q_values(self.behavior_dataset)
            n_samples = 10
            next_actions = [get_target_actions(self.behavior_dataset.states_next, 
                                            scans=self.behavior_dataset.scans_next)
                    for _ in range(n_samples)]
            
            next_q_values = [self.model(self.behavior_dataset.states_next,
                                        next_actions[i],
                                        #scans = behavior_dataset.next_scans,
                                        timesteps = self.behavior_dataset.timesteps + self.behavior_dataset.timestep_constant)
                            for i in range(n_samples)] # / n_samples
            next_q_values = torch.stack(next_q_values, dim=0).mean(dim=0)

            rewards = self.behavior_dataset.rewards

            rewards = rewards + self.discount * next_q_values - q_values
            
            ############# Importance Weighting #############
            num_trajectories = len(self.behavior_dataset.initial_states)
            # max trajectory length
            ones_indices = np.where(self.behavior_dataset.finished==1)[0]
            differences = np.diff(ones_indices)
            max_trajectory_length = differences.max() if len(differences) > 0 else 0
            print(max_trajectory_length)
            trajectory_weights = self.behavior_dataset.initial_weights.numpy()
            trajectory_starts = np.where(self.behavior_dataset.mask_inital==1)[0]
            
            batched_rewards = np.zeros([num_trajectories, max_trajectory_length])
            batched_masks = np.zeros([num_trajectories, max_trajectory_length])
            batched_log_probs = np.zeros([num_trajectories, max_trajectory_length])

            for traj_idx, traj_start in enumerate(trajectory_starts):
                traj_end = trajectory_starts[traj_idx+1] if traj_idx+1 < len(trajectory_starts) else len(rewards)
                #print(traj_end)
                traj_len = traj_end - traj_start
                #print(traj_len)
                batched_rewards[traj_idx, :traj_len] = rewards[traj_start:traj_end]
                # this is not completely correct here
                batched_masks[traj_idx, :traj_len] = 1.
                #print()
                assert self.behavior_dataset.masks[traj_end-1] == 0
                batched_log_probs[traj_idx, :traj_len] = (
                    -behavior_log_probs[traj_start:traj_end] +
                    target_log_probs[traj_start:traj_end])
            
            assert np.sum(batched_masks) == len(rewards)
            # discounted weights
            batched_weights = (batched_masks *
                            (self.discount **
                                np.arange(max_trajectory_length))[None, :])

            clipped_log_probs = np.clip(batched_log_probs, -6., 2.)
            # how much is clipped here?

            import matplotlib.pyplot as plt
            cum_log_probs = batched_masks * np.cumsum(clipped_log_probs, axis=1)

            cum_log_prob_offset = np.max(cum_log_probs, axis=0)
            cum_probs = np.exp(cum_log_probs- cum_log_prob_offset[None,:])

            avg_cum_probs = (
                np.sum(cum_probs * trajectory_weights[:, None], axis=0) /
                ( np.sum(batched_masks * trajectory_weights[:, None],
                                axis=0)+1e-10)) # 
            norm_cum_probs = cum_probs / (1e-10 + avg_cum_probs[None, :])

            #average the cum probs along dim=1 and plot

            weighted_rewards = batched_weights * batched_rewards * norm_cum_probs
            trajectory_values = np.sum(weighted_rewards, axis=1)
            mean_trajectory_value = np.sum(trajectory_values * trajectory_weights) / (
                    np.sum(trajectory_weights))
            std_trajectory_value = np.std(trajectory_values) # assumes only 1s in weights
            pred_return = offset + mean_trajectory_value
            pred_std = std_deviation + std_trajectory_value
            return pred_return, pred_std