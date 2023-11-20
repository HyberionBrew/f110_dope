class DR_estimator:
    def __init__(self, model, behavior_dataset):
        self.model = model
        self.behavior_dataset = behavior_dataset
    
    def estimate_returns(self,get_target_actions, get_target_logprobs):
        behavior_log_probs = self.behavior_dataset.log_probs
        target_log_probs = get_target_logprobs(self.behavior_dataset.states, 
                                               self.behavior_dataset.actions,
                                               scans = self.behavior_dataset.scans)
        rewards, std_deviation = self.model.estimate_returns(self.behavior_dataset.inital_states,
                                                        self.behavior_dataset.initial_weights,
                                                        get_target_actions=get_target_actions,)
        q_values = self.model(self.behavior_dataset.states,
                         self.behavior_dataset.actions,
                         #scans = behavior_dataset.scans,
                         timesteps = self.behavior_dataset.timesteps)
        n_samples = 10
        next_actions = [get_target_actions(self.behavior_dataset.next_states, 
                                           scans=self.behavior_dataset.next_scans)
                for _ in range(n_samples)]
        next_q_values = [self.model(self.behavior_dataset.next_states,
                                    next_actions[i],
                                    #scans = behavior_dataset.next_scans,
                                    timesteps = self.behavior_dataset.timesteps+ self.behavior_dataset.timestep_constant)
                         for i in range(n_samples)] / n_samples
        # got the next q values
        # now calculate the 