import torch
class FQEMB:
    def __init__(self, model_fqe,model_mb, discount, timestep_constant, 
                  mb_steps=20,
                  single_step_fqe = True,
                  min_reward=0, max_reward=100,
                  writer=None):
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
        self.model_fqe = self.model_fqe.to(self.device)
    def update():
        # FQE model and MB model should already be trained
        # they are just loaded in before handing to FQEMB
        pass
    def __call__(self, )
    def estimate_returns(self, inital_states, inital_weights, target_actions):
        # first we run the MB model for mb_steps to collect a bunch of steps
        # then we run the FQE model on all the collected steps and compute
        # discounted averages
        ######## RUN MB MODEL ########
        # need to provide dummy actions
        action = torch.zeros((inital_states.shape[0],1,2)).to(self.device)
        inital_states = inital_states.unsqueeze(1).to(self.device)
        print(inital_states.shape)
        all_states, all_actions = self.model_mb.fast_rollout(inital_states,
                                    action, 
                                    get_target_action=target_actions,
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
                                          get_action=target_actions,
                                          action= all_actions[:,-1],
                                          timesteps=timesteps[:],
                                          reduction=False)
            #print(fqe_reward)
            all_rewards[:,-1] = fqe_reward
            print(all_rewards)
            # now apply discount factor
            all_rewards = all_rewards * discount_factors
            all_rewards = all_rewards.sum(dim=1)
            print(all_rewards.shape)
            # now return the mean and std
            return all_rewards.mean(), all_rewards.std()
        else:
            # Multi step FQE
            fqe_rewards = torch.zeros_like(all_rewards)
            for step in range(self.mb_steps):
                timesteps = torch.ones((all_states.shape[0],1)) * step * self.timestep_constant
                timesteps = timesteps.to(self.device)
                weights = torch.ones((all_states.shape[0])).to(self.device)
                fqe_reward, _ = self.model_fqe.estimate_returns(all_states[:,step],
                                                weights[:],
                                              get_action=target_actions,
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
            return rewards.mean(), rewards.std()