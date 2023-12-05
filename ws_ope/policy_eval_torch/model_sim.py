import gymnasium as gym
import f110_gym
from f110_sim_env.base_env import make_base_env
import numpy as np
from policy_eval_torch.dataset import F110Dataset
import torch
from tqdm import tqdm

from f110_orl_dataset.normalize_dataset import Normalize
from f110_orl_dataset.config_new import Config
from policy_eval_torch.model_based_2 import GroundTruthRewardFast

class SimBased:
    def __init__(self, dataset, fn_unnormalize, fn_normalize, normalizer, target_reward, map= "Infsaal"):
        self.env = make_base_env(random_start =False,
                                 pose_start=True,)
        
                #gym.make("f110_gym:f110-v0",
                #config = dict(map=map,
                #num_agents=1, 
                #params=dict(vmin=0.5, vmax=2.0)),
                #render_mode="human")
        self.device ="cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = normalizer
        self.fn_unnormalize = fn_unnormalize
        self.fn_normalize = fn_normalize
        if target_reward=="trajectories_td_prog.zarr":
            print("[mb] Using progress reward")
            config = Config('config/td_prog_config.json')
            #self.reward_model = GroundTruthRewardFast(dataset,20, config)
        elif target_reward=="trajectories_raceline.zarr":
            print("[mb] Using raceline reward")
            config = Config('config/raceline_config.json')
            #self.reward_model = GroundTruthRewardFast(dataset,20, config)
        elif target_reward=="trajectories_min_act.zarr":
            print("[mb] Using min action reward")
            config = Config('config/min_act_config.json')
            
        else:
            raise NotImplementedError
        print(config)
        self.reward_model = GroundTruthRewardFast(dataset,20, config)

    # input are normalized states and normal actions
    def fast_rollout(self, states:torch.tensor, actions:torch.tensor, 
                     get_target_action=None,
                     horizon = 10,
                     batch_size = 256,
                     use_dynamics = True,
                     skip_first_action = False,):
        assert (len(states.shape) == 3) # (batch, timestep, state_dim)
        assert (len(actions.shape) == 3) # (batch, timestep, action_dim)
        assert (states.shape[0] == actions.shape[0])
        assert (states.shape[1] == actions.shape[1])
        assert (states.shape[2] == 11)
        assert (actions.shape[2] == 2)
        # lets start by reseting the env
        inital_states = states[:,0,:]
        # we need to unnormalize the inital states
        inital_states = self.fn_unnormalize(inital_states)
        # print(inital_states)
        #compute the theta from the inital states
        # sin is inital_states[:,2], cos is inital_states[:,3]
        theta = np.arctan2(inital_states[:,2], inital_states[:,3])
        #print(theta)
        #while True:
        all_states =  torch.zeros((0, horizon, states[0].shape[-1]))#.to(self.device)
        all_actions = torch.zeros((0, horizon, 2))#.to(self.device)
        
        for i, inital_state in tqdm(enumerate(inital_states)):
            
            rollout_states = torch.zeros((1, 0, inital_state.shape[-1]))#.to(self.device)
            rollout_actions = torch.zeros((1, 0, 2))#.to(self.device)
            # print(inital_state)
            reset_options = dict(poses=np.array([[inital_state[0],
                                                  inital_state[1],
                                                  theta[i]]]),
                                velocity=inital_state[5].item(),)
            self.env.reset(options = reset_options)
            # action = np.array([[0.0,0.0]])#actions[i,0,:]
            state = inital_state
            ## add dim 0 to state
            state = state.unsqueeze(0)
            # print(inital_state)
            for j in range(horizon):

                if get_target_action is not None:
                    # need to normalize the state
                    norm_state = self.fn_normalize(state)
                    action = get_target_action(norm_state)
                else:
                    #print(actions)
                    #print(actions.shape)
                    # we enter here for 0 and 1, but 1 action should be discarded later
                    action = np.array([actions[i,0,:].cpu().numpy()])
                    action = torch.Tensor(action)
                #print(action)
                #action_torch = torch.tensor(action.clone().detach()).float()

                #self.env.render()
                rollout_actions = torch.cat([rollout_actions, action.unsqueeze(1)], dim=1)
                rollout_states = torch.cat([rollout_states, state.unsqueeze(1)], dim=1)
                #print(action)
                #print(action.shape)
                #print(action.dtype)
                obs, reward, done, truncated, info = self.env.step(action.cpu().numpy())
                unnormalized_obs = info["observations"]
                batched_obs = self.normalizer.flatten_batch(unnormalized_obs)
                state = torch.tensor(batched_obs).float()
                
            all_states = torch.cat([all_states, rollout_states], dim=0)
            all_actions = torch.cat([all_actions, rollout_actions], dim=0)
        states_normalized = self.fn_normalize(all_states)
        return states_normalized, all_actions
if __name__ == "__main__":
    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of right now 
      **dict(name='f110_with_dataset-v0',
          config = dict(map="Infsaal", num_agents=1,
          params=dict(vmin=0.5, vmax=2.0)),
            render_mode="human")
    )
    env = F110Env

    behavior_dataset = F110Dataset(
        env,
        normalize_states=True,
        normalize_rewards=False,#FLAGS.normalize_rewards,
        path = f"/home/fabian/msc/f110_dope/ws_ope/f1tenth_orl_dataset/data/trajectories_min_act.zarr",
        exclude_agents = ['progress_weight', 'raceline_delta_weight', 'min_action_weight'],#['det'], #+ [FLAGS.target_policy] , #+ ["min_lida", "raceline"],
        alternate_reward=True,
        include_timesteps_in_obs = True,
        only_terminals=True,
        clip_trajectory_length= (0,50),
        sample_from_trajectories= 50,
        )
    
    sim = SimBased(fn_normalize=behavior_dataset.normalize_states,
                   fn_unnormalize=behavior_dataset.unnormalize_states,
                   normalizer = Normalize(),
                   map="Infsaal")
    #print((1-behavior_dataset.masks).sum())
    initial_states = behavior_dataset.initial_states#[:100]
    initial_states = initial_states.unsqueeze(1)
    actions = behavior_dataset.actions
    actions = actions.unsqueeze(1)
    # at dim 1 of the initial states add 0s as many as actions
    print(actions.shape)
    print(initial_states.shape)
    print(behavior_dataset.states[:10])
    initial_states = torch.cat([initial_states, torch.zeros(19,49,11)], dim=1)
    #print(initial_states.shape)
    #print(actions.shape)
    actions = actions.reshape(19,50,2)
    sim.fast_rollout(initial_states, actions)
    pass