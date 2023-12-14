import numpy as np
from policy_eval_torch.dataset import F110Dataset
import torch 

class DR_estimator:
    def __init__(self, model, behavior_dataset: F110Dataset, discount, max_trajectory_length=250):
        self.model = model
        self.behavior_dataset = behavior_dataset
        self.discount = discount
        #self.max_trajectory_length = max_trajectory_length
    
    def estimate_returns(self,get_target_actions, get_target_logprobs, algo="fqe"):
        import matplotlib.pyplot as plt
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
                                                #added
                                                self.behavior_dataset.timesteps,
                                                scans = self.behavior_dataset.scans)
            print("nans, behavior_log_probs", np.isnan(behavior_log_probs).sum())
            ############# DR part #############
            offset, std_deviation = self.model.estimate_returns(self.behavior_dataset.initial_states,
                                                            self.behavior_dataset.initial_weights,
                                                            get_action=get_target_actions,
                                                            action_timesteps = torch.zeros_like(self.behavior_dataset.initial_states[:,0]),)
            if algo=="fqe":
                q_values = self.model(self.behavior_dataset.states,
                                self.behavior_dataset.actions,
                                timesteps = self.behavior_dataset.timesteps)
                assert np.isnan(q_values).sum() == 0
            else:
                # next_q_values = self.model.compute_q_values(self.behavior_dataset, next_q_values=True)
                print("Computing Q values")
                q_values = self.model.compute_q_values(self.behavior_dataset)
            n_samples = 5
            
            if algo=="fqe":
                next_actions = [get_target_actions(self.behavior_dataset.states_next,
                                                  (self.behavior_dataset.timesteps + self.behavior_dataset.timestep_constant), 
                                            scans=self.behavior_dataset.scans_next)
                    for _ in range(n_samples)]
                #assert np.isnan(next_actions).sum() == 0
                
                next_q_values = [self.model(self.behavior_dataset.states_next,
                                            next_actions[i],
                                            #scans = behavior_dataset.next_scans,
                                            timesteps = self.behavior_dataset.timesteps + self.behavior_dataset.timestep_constant)
                                for i in range(n_samples)] # / n_samples
               
                next_q_values = torch.stack(next_q_values, dim=0).mean(dim=0)
                assert np.isnan(next_q_values).sum() == 0
            else:
                print("Computing Next Q values")
                next_q_values = [self.model.compute_q_values(self.behavior_dataset, next_q_values=True)
                                for i in range(n_samples)] # / n_samples
                #print(next_q_values)
                next_q_values = torch.stack(next_q_values, dim=0).mean(dim=0)
                #print(next_q_values)
                #print(next_q_values.shape)
                #exit()
            print("Finished Q value computation")
            rewards = self.behavior_dataset.rewards

            rewards = rewards + self.discount * next_q_values - q_values
            
            ############# Importance Weighting #############
            num_trajectories = len(self.behavior_dataset.initial_states)
            # max trajectory length
            ones_indices = np.where(self.behavior_dataset.finished==1)[0]
            differences = np.diff(ones_indices)
            max_trajectory_length = differences.max() if len(differences) > 0 else 0
            #print(max_trajectory_length)
            trajectory_weights = self.behavior_dataset.initial_weights.numpy()
            trajectory_starts = np.where(self.behavior_dataset.mask_inital==1)[0]
            
            batched_rewards = np.zeros([num_trajectories, max_trajectory_length])
            batched_masks = np.zeros([num_trajectories, max_trajectory_length])
            # batched_log_probs that are not getting filled up are crashes, set them to -10.0
            batched_log_probs = np.zeros([num_trajectories, max_trajectory_length]) - 10.0
            # what should happen for log_probs with crashes?
            # i.e. 0?
            #print("batched_log_probs nans", np.isnan(batched_log_probs).sum())
            for traj_idx, traj_start in enumerate(trajectory_starts):
                traj_end = trajectory_starts[traj_idx+1] if traj_idx+1 < len(trajectory_starts) else len(rewards)
                #print(traj_end)
                #if traj_idx == 364:
                #    print(traj_start)
                #    print(traj_end)
                #    print(self.behavior_dataset.rewards[traj_start:traj_end])
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
                #if traj_idx == 364:
                #    print(batched_log_probs[traj_idx, :traj_len])
                #    print("---")
                #    print(batched_log_probs[traj_idx,:])

                #print(traj_idx, traj_start, traj_end)
                #print("state;", self.behavior_dataset.states[traj_start:3])
                #print("scans", self.behavior_dataset.scans[traj_start:3])
                #print("action", self.behavior_dataset.actions[traj_start:3])
                #print("unnormed",self.behavior_dataset.unnormalize_states(self.behavior_dataset.states[traj_start:3]))
                #print(behavior_log_probs[traj_start :traj_end])
                #print(target_log_probs[traj_start:traj_end])
                
                #print(batched_log_probs[traj_idx, :10])
                assert np.isnan(batched_log_probs).sum() == 0
            
            assert np.sum(batched_masks) == len(rewards)
            # discounted weights
            batched_weights = (batched_masks *
                            (self.discount **
                                np.arange(max_trajectory_length))[None, :])

            clipped_log_probs = np.clip(batched_log_probs, -6., 2.)
            """
            print("0",clipped_log_probs[0])
            for i in range(600): #len(clipped_log_probs)):
                plt.plot(clipped_log_probs[i], alpha=0.1)
            plt.plot(clipped_log_probs[0],color="red")
            print(clipped_log_probs[0])
            plt.title("clipped_log_probs")
            plt.show()
            
            # how much is clipped here?
            #count nans in clipped log probs
            print("clipped_log_probs", np.isnan(clipped_log_probs).sum())
            """



            
            #for j in range(5):
            #    for i in range(50):#len(clipped_log_probs)):

            #        plt.plot(clipped_log_probs[i+j*50])
            #    plt.title("norm_cum_probs")
            #    plt.show()
            #if True:
            #    clipped_log_probs = (clipped_log_probs+ 6)/8
            #    cum_log_probs = batched_masks * np.cumsum(clipped_log_probs, axis=1)
            #    # now divide by 1,2,3,4,5... 
                
            #print(batched_masks.shape)
            #print(np.cumsum(clipped_log_probs, axis=1).shape)

            cum_log_probs = np.cumsum(clipped_log_probs, axis=1) #* batched_masks
            # print where cum_log_probs is zero
            # print("where zero",np.where(np.isclose(batched_masks,0.0)))

            for i in range(len(cum_log_probs)):

                plt.plot(cum_log_probs[i],alpha=0.1) #240 
            plt.plot(cum_log_probs[0],color="red")
            plt.title("cum_probs not normalized")
            plt.show()
            cum_log_prob_offset = np.max(cum_log_probs, axis=0)
            """
            print("offset shape", cum_log_prob_offset.shape)
            print("cum_log probs shape", cum_log_probs.shape)
            print("cum_log_probs", cum_log_probs[0])
            print("cum_log_prob_offset", cum_log_prob_offset)
            print("cum_log_probs - cum_log_prob_offset", cum_log_probs[0]- cum_log_prob_offset)
            
            plt.plot(cum_log_probs[0]- cum_log_prob_offset)
            plt.show()
            """
            cum_probs = np.exp(cum_log_probs- cum_log_prob_offset[None,:])
            """
            print(cum_probs)
            print("last not zero:",len(cum_probs)-np.sum(np.isclose(cum_probs[:,-1],0)))
            # print indices of trajectories where last non zero
            print("last non zero:",np.where(~np.isclose(cum_probs[:,-1],0.0))[0])
            # plot all where last non zero
            for i in np.where(~np.isclose(cum_probs[:,-1],0.0))[0]:
                plt.plot(cum_probs[i])
            plt.title("cum_probs no zero")
            plt.show()
            """
            for i in range(len(cum_probs)):

                plt.plot(cum_probs[i], alpha =0.1) #240 
            #for i in range(100):
            #    plt.plot(cum_probs[i],color="red")
            plt.title("cum_probs")
            plt.show()
            
            # count how many are non zero at the last timestep

            avg_cum_probs = (
                np.sum(cum_probs * trajectory_weights[:, None], axis=0) /
                ( np.sum(batched_masks * trajectory_weights[:, None],
                                axis=0)+1e-10)) # 
            #average the cum probs along dim=1 
            # plot avg_cum_probs
            # calculate avg_cum prob of the first 100 cum probs
            first_avg_cum_probs = (
                np.sum(cum_probs[:100] * trajectory_weights[:100, None], axis=0) /
                ( np.sum(batched_masks[:100] * trajectory_weights[:100, None],
                                axis=0)+1e-10))
            
            
            # plot with transperency all cum_probs
            for i in range(len(cum_probs)):

                plt.plot(cum_probs[i], alpha=0.1)
            # color red
            """
            plt.plot(avg_cum_probs, color="red")
            plt.plot(first_avg_cum_probs, color="green")
            plt.title("avg_cum_probs")
            plt.show()
            """
            norm_cum_probs = cum_probs / (1e-10 + avg_cum_probs[None, :])

            for i in range(len(batched_rewards)):

                plt.plot(batched_rewards[i])
            """
            plt.title("batched_rewards")
            plt.show()
            """
            norm_cum_probs = np.clip(norm_cum_probs, -20., 20.)

            for i in range(len(norm_cum_probs)):

                plt.plot(norm_cum_probs[i], alpha=0.1)
            plt.plot(norm_cum_probs[20],color="red")
            plt.title("norm_cum_probs 2")
            plt.show()


            weighted_rewards = batched_weights * batched_rewards * norm_cum_probs
            # plot the weighted rewards for trajectories plot 240-245
            
            
            """
            for i in range(len(weighted_rewards)):

                plt.plot(weighted_rewards[i])
            plt.show()
            """
            trajectory_values = np.sum(weighted_rewards, axis=1)
            # plot all trajectory values
            plt.plot(trajectory_values)
            plt.title("trajectory_values")
            plt.show()

            mean_trajectory_value = np.sum(trajectory_values * trajectory_weights) / (
                    np.sum(trajectory_weights))
            std_trajectory_value = np.std(trajectory_values) # assumes only 1s in weights
            pred_return = offset + mean_trajectory_value
            pred_std = std_deviation + std_trajectory_value #TODO! this is not really accurate
            """
            print("mean",mean_trajectory_value)
            print("offs",offset)
            print("std_trajectory_value", std_trajectory_value)
            """
            return pred_return, pred_std