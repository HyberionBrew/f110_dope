import torch

class FQEMB:
    def __init__(self, model_fqe,model_mb, discount, timestep_constant, max_timestep,
                  rollouts = 10,
                  mb_steps=20,
                  single_step_fqe = True,
                  min_reward=0,
                    max_reward=100,
                  writer=None,
                  target_actions = None,):
        self.rollouts=rollouts
        self.model_fqe = model_fqe
        self.model_mb = model_mb
        self.discount = discount
        self.mb_steps = mb_steps
        self.single_step_fqe = single_step_fqe
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.writer = writer
        self.timestep_constant = timestep_constant
        self.device = self.model_mb.device
        # send fqe model to device
        self.target_actions = target_actions
        self.model_fqe = self.model_fqe.to(self.device)
        self.max_timestep = max_timestep
    def update():
        # FQE model and MB model should already be trained
        # they are just loaded in before handing to FQEMB
        pass
    # TODO make this work for next_q_values as well, what is missing to make this work?
    # just needs some index shifting and stuff like that!
    def compute_q_values(self,behavior_dataset, next_states=False):
        states = behavior_dataset.states
        actions = behavior_dataset.actions
        timesteps = behavior_dataset.timesteps#[:100]
        rewards = behavior_dataset.rewards

        states = states#[:100]
        actions = actions#[:100]
        states = states.unsqueeze(1).to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards#[:100] # .to(self.device)

        
        all_states, all_actions = self.model_mb.fast_rollout(states,
                                        actions, 
                                        get_target_action=None,
                                        skip_first_action=True,
                                        horizon=2) # this is not correct horizon is +1 apparently
        new_inital_states = all_states[:,-1].unsqueeze(1)
        new_inital_actions = all_actions[:,-1].unsqueeze(1)

        all_states_t, all_actions_t = self.model_mb.fast_rollout(new_inital_states,
                                new_inital_actions, 
                                get_target_action=self.target_actions,
                                skip_first_action=True,
                                horizon=self.mb_steps-1)
        # now join the actual inital states to the new states
        #print(all_states[0])
        #print(all_states.shape)
        #print(all_states_t.shape)
        all_states = torch.cat([all_states, all_states_t[:,1:]], dim=1)
        all_actions = torch.cat([all_actions, all_actions_t[:,1:]], dim=1)


        #print(all_states[0])
        #print(all_states.shape)
        #exit()
        # now we need to zero the rewards of the states that are to close to the horizon
        # figure out where there are timestep == 0
        all_rewards = self.model_mb.reward_model(all_states.clone(), all_actions.clone())
        #print(rewards.shape)
        #print(all_rewards)
         # set the 0th reward to the reward we actually already know, needed for td-progress
        all_rewards[:,0] = rewards
        #print(all_rewards)
        #print(behavior_dataset.rewards[0:100])
        # in the case we want to compute normal q_values we can figure out the starts easily
        starts = torch.where(behavior_dataset.mask_inital==1)[0]
        ends = starts - 1
        # remove the -1 and add the last timestep
        ends = ends[1:].to('cpu')
        ends = torch.cat([ends, torch.tensor([len(behavior_dataset.states)-1])])
        #ends = torch.tensor([49,99])
        #print(ends)
        # assert 0 not in ends
        assert 0 not in ends
        
        
        final_states = all_states_t[:,-1]
        timesteps_shifted = timesteps + self.timestep_constant * (self.mb_steps-1)
        timesteps_shifted = timesteps_shifted.to(self.device)
        #print(final_states.shape)
        #print(timesteps_shifted.shape)
        if self.single_step_fqe:
            # for every final_state that is not OOB/terminal we compute the according fqe_value
            weights = torch.ones((final_states.shape[0])).to(self.device)
            print()
            all_fqe_reward, _ = self.model_fqe.estimate_returns(final_states,
                                weights,
                            get_action=None,#self.target_actions,
                            action= all_actions[:,-1],
                            timesteps=timesteps_shifted,
                            reduction=False)
            # TODO! figure out why FQE yields negative values? OOD?
            # print(timesteps_shifted[44:52])
            # set the reward at position [:,-1] to all_fqe_reward
            all_rewards[:,-1] = all_fqe_reward.cpu().numpy()
            # zero out where it is OOB
            # zero out everything that is "outside" the trajectory
            for i in range(self.mb_steps):
                all_rewards[ends - i,i+1:] = 0.0
            print(all_rewards[0:10])
            # now we need to compute the correct estimates for each state with correct discounts
            # loop over all the ends
            all_rewards = torch.tensor(all_rewards).float()
            final_rewards = torch.zeros((len(states)))
            start = 0
            j = 0
            #for num_trajectories in range(len(states)):
            """
            for i, rewards in enumerate(all_rewards):
                trajectory_discounts = torch.tensor([self.discount**(j) for j in range(len(rewards))])
                #print(trajectory_discounts)
                #print(rewards)
                temp = torch.mul(rewards ,trajectory_discounts)
                #print(temp)
                final_rewards[i] = temp.sum()
            # final rewards are the q_values
            print(final_rewards)
            """
            # this is not correct, it should be smaller, but whatever
            # TODO!
            max_length = max(len(rewards) for rewards in all_rewards)
            print(max_length)
            # Create a matrix of discount factors
            discounts = torch.tensor([self.discount**i for i in range(max_length)])
            discount_matrix = discounts.unsqueeze(0).repeat(len(all_rewards), 1)

            # Adjust the shape of each rewards tensor and apply the discount
            final_rewards = torch.zeros(len(all_rewards))
            # we can easily remove this loop, TODO!
            for i, rewards in enumerate(all_rewards):
                len_rewards = len(rewards)
                discounted_rewards = rewards * discount_matrix[i, :len_rewards]
                # print(discount_matrix[i, :len_rewards])
                final_rewards[i] = discounted_rewards.sum()

            start = 0
            for end in ends:
                trajectory = final_rewards[start:end+1]
                trajectory_discounts = torch.tensor([self.discount**(j) for j in range(len(trajectory))])
                trajectory_discounted = torch.mul(trajectory ,trajectory_discounts)
                final_rewards[start:end+1] = trajectory_discounted
                start = end + 1
            return final_rewards
        else:
            raise NotImplementedError

        exit()
    
    
    """
    Returns for each input state the mean and std of the returns
    """
    # TODO integrate this function into estimate returns
    def __call__(self, states, actions, timesteps=None, batch_size=1000):
        n_data = states.shape[0]
        results = []
        if timesteps is None:
            timesteps = torch.zeros_like(states[:, -1]).unsqueeze(-1)
        else:
            print(states.shape)
            print(actions.shape)
            for i in range(self.rollouts):
                ###### RUN MB #######
                action = actions.to(self.device)
                states = states.to(self.device)
                states = states.unsqueeze(1)
                action = action.unsqueeze(1)
                timesteps = timesteps.to(self.device)[0:100]
                # For debugging! TODO! remove
                states = states[:100]
                action = action[:100]
                

                # do one step rollout with known actions

                all_states, all_actions = self.model_mb.fast_rollout(states,
                                        action, 
                                        get_target_action=None,
                                        skip_first_action=True,
                                        horizon=2) # this is not correct horizon is +1 apparently
                new_inital_states = all_states[:,-1].unsqueeze(1)
                new_inital_actions = all_actions[:,-1].unsqueeze(1)

                all_states_t, all_actions_t = self.model_mb.fast_rollout(new_inital_states,
                                        new_inital_actions, 
                                        get_target_action=self.target_actions,
                                        skip_first_action=True,
                                        horizon=self.mb_steps-1)
                # now join the actual inital states to the new states
                #print(all_states[0])
                #print(all_states.shape)
                #print(all_states_t.shape)
                all_states = torch.cat([all_states, all_states_t[:,1:]], dim=1)
                all_actions = torch.cat([all_actions, all_actions_t[:,1:]], dim=1)


                #print(all_states[0])
                #print(all_states.shape)
                #exit()
                # now we need to zero the rewards of the states that are to close to the horizon
                # figure out where there are timestep == 0
                all_rewards = self.model_mb.reward_model(all_states.clone(), all_actions.clone())
                print(all_rewards.shape)
                
                exit()
                starts = torch.where(timesteps == 0)[0]
                ends = starts - 1
                # remove the -1 and add the last timestep
                ends = ends[1:].to('cpu')
                ends = torch.cat([ends, torch.tensor([len(states)-1])])
                print(ends)
                # zero out rewards at the end
                for i in range(self.mb_steps):
                    all_rewards[ends - i,i+1:] = 0.0
                # now lets do the fqe rewards
                fqe_rewards = torch.zeros_like(all_rewards)
                if self.single_step_fqe:
                    # we actually have to be careful here for the all states
                    # lets only apply fqe to the last step, iff
                    fqe_reward, _ = self.model_fqe.estimate_returns(all_states[:,-1],
                                weights[:],
                            get_action=get_action,
                            action= all_actions[:,-1],
                            timesteps=timesteps[:],
                            reduction=False)

                
                exit()
                



    # This always starts at timestep 0
    def estimate_returns(self, inital_states, inital_weights, get_action):
        # first we run the MB model for mb_steps to collect a bunch of steps
        # then we run the FQE model on all the collected steps and compute
        # discounted averages
        gathered_rewards = []
        gathered_stds = []
        for _ in range(self.rollouts):
            ######## RUN MB MODEL ########
            # need to provide dummy actions
            action = torch.zeros((inital_states.shape[0],1,2)).to(self.device)
            inital_states_ = inital_states.unsqueeze(1).to(self.device)
            print(inital_states_.shape)
            all_states, all_actions = self.model_mb.fast_rollout(inital_states_,
                                        action, 
                                        get_target_action=get_action,
                                        skip_first_action=True,
                                        horizon=self.mb_steps) # fixes first action being 0,0
            #print(all_states)
            #print(all_states.shape)
            #print(inital_states)
            #print(all_states[:,0,:])
            # TODO! make sure assertion is correct
            # assert torch.is_close(all_states[:,0,:],inital_states.cpu().numpy()
            # maybe do some visualization here
            
            # calculate the rewards
            all_rewards = self.model_mb.reward_model(all_states.clone(), all_actions.clone())

            discount_factors = torch.tensor([self.discount**i for i in range(self.mb_steps)])
            all_rewards = torch.tensor(all_rewards).float()
            #all_rewards = torch.tensor(all_rewards) * discount_factors
            print(all_rewards.shape)
            print(all_rewards)
            ######## RUN FQE MODEL ########
            # now run the FQE model on all the collected steps
            print(all_states.shape) # should be (batch_size, mb_steps, 11)
            print(all_actions.shape) # should be (batch_size, mb_steps, 2)
            assert all_states.shape[1] == self.mb_steps
            assert all_actions.shape[1] == self.mb_steps
            assert len(all_states.shape) == 3
            assert len(all_actions.shape) == 3
            # create 


            if self.single_step_fqe:
                # run fqe only on the last step and replace the value there
                # create the timestep array
                timesteps = torch.ones((all_states.shape[0],1)) * self.mb_steps * self.timestep_constant
                timesteps = timesteps.to(self.device)
                weights = torch.ones((all_states.shape[0])).to(self.device)
                #print(all_states[0,-1])
                #all_states_input = all_states[:,-1]
                #print(all_states_input.shape)
                fqe_reward, _ = self.model_fqe.estimate_returns(all_states[:,-1],
                                                weights[:],
                                            get_action=get_action,
                                            action= all_actions[:,-1],
                                            timesteps=timesteps[:],
                                            reduction=False)
                #print(fqe_reward)
                all_rewards[:,-1] = fqe_reward
                #print(all_rewards)
                # now apply discount factor
                all_rewards = all_rewards * discount_factors
                all_rewards = all_rewards.sum(dim=1)
                #print(all_rewards.shape)
                # now return the mean and std
                gathered_rewards.append(all_rewards.mean().clone())
                gathered_stds.append(all_rewards.std().clone())
            else:
                # Multi step FQE
                fqe_rewards = torch.zeros_like(all_rewards)
                for step in range(self.mb_steps):
                    timesteps = torch.ones((all_states.shape[0],1)) * step * self.timestep_constant
                    timesteps = timesteps.to(self.device)
                    weights = torch.ones((all_states.shape[0])).to(self.device)
                    fqe_reward, _ = self.model_fqe.estimate_returns(all_states[:,step],
                                                    weights[:],
                                                get_action=get_action,
                                                action= all_actions[:,step],
                                                timesteps=timesteps[:],
                                                reduction=False)
                    fqe_rewards[:,step] = fqe_reward
                # now 
                cum_rewards = all_rewards * discount_factors
                cum_rewards = torch.cumsum(cum_rewards,dim=1)
                new_cum = torch.zeros_like(cum_rewards)
                # shift by 1, FQE(t) already contains the reward at that timestep
                new_cum[:,1:] = cum_rewards[:,:-1] 
                # add the new_cum to the fqe_rewards
                rewards = fqe_rewards + new_cum
                gathered_rewards.append(rewards.mean().clone())
                gathered_stds.append(rewards.std().clone())
        # now calculate the mean and std
        print(len(gathered_rewards))
        print(len(gathered_stds))
        gathered_rewards = torch.tensor(gathered_rewards)
        gathered_stds = torch.tensor(gathered_stds)
        print("gathered_rewards", gathered_rewards)
        print("gathered_stds", gathered_stds)
        return gathered_rewards.mean(), gathered_stds.mean()