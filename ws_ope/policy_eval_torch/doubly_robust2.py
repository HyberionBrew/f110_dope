import numpy as np
from policy_eval_torch.dataset import F110Dataset
import torch 
import matplotlib.pyplot as plt


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
            """
            DR_Estimate = 0
            for each trajectory H in D
                Weight = 1
                Trajectory_Estimate = 0
                for each timestep t in H
                    s, a, r = state, action, and reward at timestep t in H
                    Importance_Weight_t = exp(logprob_π_e(a|s) - logprob_π_b(a|s))
                    Weight *= Importance_Weight_t
                    if t < length(H) - 1
                        Direct_Method = Q_hat(s, a) - γ * V_hat(next_state(s, a))
                    else
                        Direct_Method = 0
                    Trajectory_Estimate += γ^t * (Weight * r + Direct_Method - Weight * Q_hat(s, a))
                DR_Estimate += Trajectory_Estimate
            DR_Estimate /= number of trajectories in D
            return DR_Estimate
            """

            num_trajectories = len(self.behavior_dataset.initial_states)
            trajectory_starts = np.where(self.behavior_dataset.mask_inital==1)[0]
            rewards = self.behavior_dataset.rewards
            # compute the iw
            states = self.behavior_dataset.states
            actions = self.behavior_dataset.actions
            scans = self.behavior_dataset.scans
            behavior_log_probs = self.behavior_dataset.log_probs

            
            ######### calculate IWs ######### 
            target_log_probs = get_target_logprobs(states, actions, scans=scans)
            num_trajectories = len(self.behavior_dataset.initial_states)
            ones_indices = np.where(self.behavior_dataset.finished==1)[0]
            differences = np.diff(ones_indices)
            max_trajectory_length = differences.max() if len(differences) > 0 else 0

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
                batched_log_probs[traj_idx, :traj_len] = (
                    -behavior_log_probs[traj_start:traj_end] +
                    target_log_probs[traj_start:traj_end])
                
            clipped_log_probs = np.clip(batched_log_probs, -6., 2.)
            cum_log_probs = np.cumsum(clipped_log_probs, axis=1)
            cum_log_prob_offset = np.max(cum_log_probs, axis=0)
            cum_probs = np.exp(cum_log_probs- cum_log_prob_offset[None,:])
            avg_cum_probs = (
                np.sum(cum_probs , axis=0) / cum_probs.shape[0])
            importance_weight = cum_probs # / (avg_cum_probs[None, :])

            for i in range(900):
                plt.plot(importance_weight[i])
            print(avg_cum_probs.shape)
            print(avg_cum_probs)
            plt.show()
            """
            for traj_idx, traj_start in enumerate(trajectory_starts):

                traj_end = trajectory_starts[traj_idx+1] if traj_idx+1 < len(trajectory_starts) else len(rewards)
                weigth = 1
                Trajectory_Estimate = 0
                states = self.behavior_dataset.states[traj_start:traj_end]
                actions = self.behavior_dataset.actions[traj_start:traj_end]
                log_probs = behavior_log_probs[traj_start:traj_end]
                rewards = self.behavior_dataset.rewards[traj_start:traj_end]
                scans = self.behavior_dataset.scans[traj_start:traj_end]
                target_log_probs = get_target_logprobs(states, actions, scans=scans)
                batched_log_probs = target_log_probs- log_probs
                clipped_log_probs = np.clip(batched_log_probs, -6., 2.)
                print(clipped_log_probs.shape)
                print(clipped_log_probs)
                cum_logprobs = np.cumsum(clipped_log_probs)
                print(cum_logprobs)
                cum_logprobs_offset = max(cum_logprobs)
                print(cum_logprobs_offset)
                print(cum_logprobs-cum_logprobs_offset)
                importance_weight = np.exp(cum_logprobs- cum_logprobs_offset)
                importance_weight = importance_weight # /max(importance_weight)
                plt.plot(importance_weight)
                plt.show()
                #importance_weight = np.cumprod(importance_weight)
                exit()
            """