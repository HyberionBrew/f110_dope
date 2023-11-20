class FQEMB:
    def __init__(self, model_fqe,model_mb, discount, 
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

    def update():
        # FQE model and MB model should already be trained
        # they are just loaded in before handing to FQEMB
        pass
    
    def estimate_returns(self, inital_states, inital_weights, target_actions):
        # first we run the MB model for mb_steps to collect a bunch of steps
        # then we run the FQE model on all the collected steps and compute
        # discounted averages
        ######## RUN MB MODEL ########
        inital_states