import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib.cm as cm
from policy_eval_torch.dynamics_new import NewDynamicsModel

def tf_to_torch(tf_tensor):
    """
    Convert a TensorFlow tensor to a PyTorch tensor.
    """
    # Convert TensorFlow tensor to NumPy
    numpy_array = tf_tensor.numpy()
    
    # Convert NumPy array to PyTorch tensor
    torch_tensor = torch.from_numpy(numpy_array)
    
    return torch_tensor

import torch.nn.init as init


"""
A network that takes in the current x,y state and outputs the sin and cos of the progress
"""
class ProgressNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super(ProgressNetwork, self).__init__()
        # Define the architecture here
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size) # Second fully connected layer
        self.fc3 = nn.Linear(hidden_size, output_size) # Output layer

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer without an activation function
        x = self.fc3(x)
        # Normalize the output to lie on the unit circle
        # This enforces the sin^2(theta) + cos^2(theta) = 1 constraint
        x = F.normalize(x, p=2, dim=1)
        return x

class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dt, min_state, max_state, lr=1e-4, weight_decay=1e-5):
        
        super().__init__()
        self.min_state = min_state
        self.max_state = max_state
        self.state_size, self.action_size, self.dt = state_size, action_size, dt
        A_size, B_size = state_size * state_size, state_size * action_size
        #self.A1 = nn.Linear(state_size + action_size, hidden_size)
        #self.A2 = nn.Linear(hidden_size, A_size)
        #self.B1 = nn.Linear(state_size + action_size, hidden_size)
        #self.B2 = nn.Linear(hidden_size, B_size)
        self.A_layers = nn.ModuleList()
        self.A_layers.append(self._make_layer(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.A_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.A_layers.append(self._make_layer(hidden_size[-1], A_size))

        # Construct hidden layers for B
        self.B_layers = nn.ModuleList()
        self.B_layers.append(self._make_layer(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.B_layers.append(self._make_layer(hidden_size[i-1], hidden_size[i]))
        self.B_layers.append(self._make_layer(hidden_size[-1], B_size))
        
        self.STATE_X, self.STATE_Y = 0, 1
        self.STATE_PROGRESS_SIN = 9
        self.STATE_PROGRESS_COS = 10
        # maybe also remove this
        self.STATE_PREVIOUS_ACTION_STEERING = 7
        self.STATE_PREVIOUS_ACTION_VELOCITY = 8

        self.progress_model = ProgressNetwork(input_size=2, hidden_size=64, output_size=2)
        self.optimizer_dynamics = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    def _make_layer(self, in_dim, out_dim):
        layer = nn.Linear(in_dim, out_dim)
        init.orthogonal_(layer.weight)
        return layer

    def forward(self, x, u):
        """
            Predict x_{t+1} = f(x_t, u_t)
        :param x: a batch of states
        :param u: a batch of actions
        """
        #in order to make learning easier apply to the u the clipping and scaling
        # u = torch.clip(u, -1, 1) * 0.05
        xu = torch.cat((x, u), -1)
        xu[:, self.STATE_X:self.STATE_Y+1] = 0  # Remove dependency in (x,y)
        xu[:, self.STATE_PROGRESS_SIN:self.STATE_PROGRESS_COS+1] = 0  # Remove dependency in progress
        # calculate the actual input action by using the previous action ob + the cliped and scaled action
        # but would also need unnormalization and then renomalization, so not at this point in time
        
        #A = self.A2(F.relu(self.A1(xu)))
        #A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
        #B = self.B2(F.relu(self.B1(xu)))
        #B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        for layer in self.A_layers[:-1]:  # All but the last layer
            xu = F.relu(layer(xu))
        A = self.A_layers[-1](xu)  # Last layer
        A = torch.reshape(A, (x.shape[0], self.state_size, self.state_size))
        # Reset and pass through B hidden layers
        xu = torch.cat((x, u), -1)
        xu[:, self.STATE_X:self.STATE_Y+1] = 0
        xu[:, self.STATE_PROGRESS_SIN:self.STATE_PROGRESS_COS+1] = 0
        for layer in self.B_layers[:-1]:  # All but the last layer
            xu = F.relu(layer(xu))
        B = self.B_layers[-1](xu)  # Last layer
        B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        
        dx = A @ x.unsqueeze(-1) + B @ u.unsqueeze(-1)
        x = x + dx.squeeze()*self.dt

        # now apply the progress model to x
        progress = self.progress_model(x[:, self.STATE_X:self.STATE_Y+1])
        # Create a mask for the indices that you want to update
        x_new = x.clone()
        x_new[:, self.STATE_PROGRESS_SIN:self.STATE_PROGRESS_COS+1] = progress
        
        #x[:, self.STATE_PROGRESS_SIN:self.STATE_PROGRESS_COS+1] = progress
        # clip the state between min and maxstate
        x_new = torch.clamp(x_new, self.min_state, self.max_state)
        return x_new , 0
    def update(self, states, actions, next_states):
        self.optimizer_dynamics.zero_grad()
        pred_states, _ = self(states, actions)
        dyn_loss = F.mse_loss(pred_states, next_states, reduction='none')
        dyn_loss = (dyn_loss).mean()
        dyn_loss.backward()
        self.optimizer_dynamics.step()
        return dyn_loss.item(), 0, 0 , 0

class DynamicsModelPolicy(object):
    def __init__(self, state_dim, action_dim, hidden_size, dt,writer,
                 learning_rate=1e-3, weight_decay=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dynamics_model = DynamicsModel(state_dim, action_dim, hidden_size, dt)
        self.dynamics_model.to(self.device)
        self.optimizer_dynamics = optim.Adam(self.dynamics_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.writer=writer

    def update(self, states,actions,next_states, step):
        self.optimizer_dynamics.zero_grad()  # Reset gradients
        pred_states = self.dynamics_model(states, actions)
        dyn_loss = F.mse_loss(pred_states, next_states, reduction='none')
        dyn_loss = (dyn_loss).mean()
        dyn_loss.backward()  # Compute gradients
        self.optimizer_dynamics.step()
        self.writer.add_scalar('cond/train/dyn_loss', dyn_loss.item(), global_step=step)

    def __call__(self, states, actions):
        return self.dynamics_model(states, actions)

    def save(self, path, filename):
        torch.save(self.dynamics_model.state_dict(), os.path.join(path, filename))


class RewardModel(nn.Module):
    """A class that implements a reward model."""

    def __init__(self, state_dim, action_dim):
        super(RewardModel, self).__init__()

        # Define the neural network layers
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 1)

        # Initialize the layers with orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.orthogonal_(self.fc4.weight)
        nn.init.orthogonal_(self.fc5.weight)

    def forward(self, x, a):
        x_a = torch.cat([x, a], dim=-1)
        x = F.relu(self.fc1(x_a))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x).squeeze(-1)

from f110_orl_dataset.reward import MixedReward
import gymnasium as gym

from f110_orl_dataset.normalize_dataset import Normalize
from f110_orl_dataset.fast_reward import StepMixedReward
from f110_orl_dataset.fast_reward import MixedReward as MixedReward_fast
from f110_orl_dataset.config_new import Config


class GroundTruthRewardFast(object):
    def __init__(self, dataset,  subsample_laser, config):
       
        self.env = gym.make('f110_with_dataset-v0',
        # only terminals are available as of right now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1,
            params=dict(vmin=0.5, vmax=2.0)),
              render_mode="human")
    )
        
        # config = Config('reward_config.json')
        self.reward = MixedReward_fast(self.env, config)
        self.model_input_normalizer = Normalize()
        self.dataset= dataset
        self.subsample_laser = subsample_laser
    
    def __call__(self, observation, action):
        # the input observation and action are tensors

        collision = np.zeros((observation.shape[0],observation.shape[1]), dtype=np.bool)
        done = collision
        #print(observation)
        #print(action)
        states = self.dataset.unnormalize_states(observation).cpu().numpy()
        action = action.cpu().numpy()
        #print(states)
        # we have to extract the previous action from the states
        # states are: batch_dim, timesteps, observation_dim
        previous_action = states[..., 7:9] 
        assert(action.shape[0]==observation.shape[0])
        #assert()

        raw_action = previous_action + np.clip(action, -1, 1) * 0.05
        #print(raw_action.shape)
        #print(previous_action.shape)
        #print(action.shape)
        #assert(raw_action.shape == previous_action.shape)
        
        #print(f"calcuated raw_action {raw_action}")
        #print(np.array([observation_dict['poses_x'][0], observation_dict['poses_y'][0]]))
        #print("previours_action", previous_action[0])
        #print("current action:", np.clip(action[0], -1, 1) * 0.05 )
        #print("Raw_action infered:", raw_action[0])
        reward, _ = self.reward(states, raw_action, 
                                      collision, done)
        #print("R:", reward)
        return reward
    
    def reset(self, pose , velocity=1.5):
        # arguments to this function are deprecated
        self.reward.reset()



class GroundTruthReward(object):
    def __init__(self, map, dataset,  subsample_laser, **reward_config):
       
        self.env = gym.make('f110_with_dataset-v0',
        # only terminals are available as of right now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1,
            params=dict(vmin=0.5, vmax=2.0)),
              render_mode="human")
    )
        
        self.reward = MixedReward(self.env, self.env.track, **reward_config)
        self.model_input_normalizer = Normalize()
        self.dataset= dataset
        self.subsample_laser = subsample_laser
    
    def __call__(self, observation, action):
        # the input observation and action are tensors
        collision = False
        done = False
        #print(observation)
        #print(action)
        states = self.dataset.unnormalize_states(observation)
        #print(states)
        # need to add the laser observation
        #print("unnormalized")
        #print(states)
        observation_dict = self.model_input_normalizer.unflatten_batch(states)
        laser_scan = self.env.get_laser_scan(states, self.subsample_laser) # TODO! rename f110env to dataset_env
        laser_scan = self.model_input_normalizer.normalize_laser_scan(laser_scan)
        observation_dict['lidar_occupancy'] = laser_scan
        # have to unsqueeze the batch dimension
        observation_dict = {key: value.squeeze(0) if value.ndim > 1 and value.shape[0] == 1 else value for key, value in observation_dict.items()}
        #print("previous action:", observation_dict["previous_action"])
        #print("current action:", np.clip(action, -1, 1) * 0.05)
        raw_action = observation_dict['previous_action'] + np.clip(action, -1, 1) * 0.05
        #print(f"calcuated raw_action {raw_action}")
        #print(np.array([observation_dict['poses_x'][0], observation_dict['poses_y'][0]]))
        reward, _ = self.reward(observation_dict, raw_action, 
                                      collision, done)
        # print("R:", reward)
        return reward
    
    def reset(self, pose , velocity=1.5):
        # add 9 empty states such that we can use the unnormalize function
        states = np.zeros((1, 11), dtype=np.float32)
        states[0, 0] = pose[0]
        states[0, 1] = pose[1]
        pose = self.dataset.unnormalize_states(states)[0][:2]
        self.reward.reset(pose, velocity=velocity)

progress_config = {
    "collision_penalty": 0.0,
    "progress_weight": 1.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "min_action_weight" : 0.0,
    "min_lidar_ray_weight" : 0.0, #missing
    "inital_velocity": 1.5,
    "normalize": False,
}

raceline_config = {
    "collision_penalty": 0.0,
    "progress_weight": 0.0,
    "raceline_delta_weight": 1.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "min_action_weight" : 0.0,
    "min_lidar_ray_weight" : 0.0, #missing
    "inital_velocity": 1.5,
    "normalize": False,
}


min_action_config = {
    "collision_penalty": 0.0,
    "progress_weight": 0.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "min_action_weight" : 1.0,
    "min_lidar_ray_weight" : 0.0, #missing
    "inital_velocity": 1.5,
    "normalize": False,
}


def dynamic_xavier_init(scale):
    def _initializer(m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight, gain=scale)
            if m.bias is not None:
                init.zeros_(m.bias)
    return _initializer

"""
@input: previous_action, target_action
"""
def get_raw_action(previous_action, target_action):
    # we first need to unnormalize the previous action
    # then we can add the target action
    pass


"""
The new dynamics model works like this:
0. the target action is computed from the previous action and the new action
1. delta x, delta y are predicted from previous theta, lin_vel, & target action
2. delta_theta_s, delta_theta_c are predicted with from the same states as above
(),out normalized to fullfill cos^2 + sin^2 = 1
3. Linear_vels are predicted from the same states as above (so no x,y, prev_action (rather target action))
4. Prev_action is computed outside the dynamics model
5. progress is computed by (x,y) 
"""


class DynamicsModelAction(object):
    
    def __init__(self, state_normalizer, state_unnormalizer, device):
        # self.model = model
        self.state_normalizer = state_normalizer
        self.state_unnormalizer = state_unnormalizer
        self.device = device

    def get_raw_action(self, x, target_action):
        # send x and target_action to numpy from torch
        x = x.cpu().numpy()
        target_action = target_action.cpu().numpy()
        x_unnormalized = self.state_unnormalizer(x)
        # to numpy array
        x_unnormalized = x_unnormalized.cpu().numpy()
        previous_action = x_unnormalized[:, 7:9]
        # we have recovered the raw action
        raw_action = previous_action + np.clip(target_action, -1, 1) * 0.05
        
        # normalize the raw_action back to get the new prev_action
        x_unnormalized[:, 7:9] = raw_action
        new_x = self.state_normalizer(x_unnormalized)
        # now we recover the new prev action
        new_prev_action = new_x[:,7:9]
        new_prev_action = new_prev_action.cpu().numpy()
        return new_prev_action
    
    def __call__(self, x, u):
        # recover raw_action
        norm_action = self.get_raw_action(x, u)
        norm_action = torch.tensor(norm_action).to(self.device)
        output = torch.zeros_like(x)
        # input is theta, vel, norm_act
        output[:,:2] = self.XYmodel(x[:,:2],x[:,2:7],norm_action)        
        # theta only gets vels and act
        output[:,2:4] = self.ThetaModel(x[:,2:4],x[:,4:7],norm_action)
        # gets theta, vel, act
        output[:,4:7] = x[:,4:7] + self.VelModel(x[2:7],norm_action)
        output[:,7:9] = norm_action
        output[:,9:11] = self.ProgressModel(output[:,:2])

        return output


    #def update(self,state, action, next_state):
    #    raw_action, prev_action_x = get_raw_action(state, action)
    #    raw_action = raw_action.to(self.device)
    #    prev_action_x = prev_action_x.to(self.device)
    #    x_pred = self.model(state,raw_action)
    #    x_pred[:,7:9] = prev_action_x
class XYModel(nn.Module):
    def __init__(self, dt, state_size=5, action_size=2, hidden_size=64, output_size=2):
        super().__init__()
        self.state_size = state_size
        self.dt = dt
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.A_layers = nn.ModuleList()
        self.A_layers.append(self._make_layer(state_size + action_size, hidden_size[0]))


    def forward(self, target, state, action):
        xu = torch.cat((state, action), -1)
        for layer in self.A_layers:
            xu = F.relu(layer(xu))
        A = self.A_layers[-1](xu)  # Last layer
        A = torch.reshape(A, (state.shape[0], self.state_size, self.state_size))
        # Reset and pass through B hidden layers
        xu = torch.cat((state, action), -1)
        for layer in self.B_layers:
            xu = F.relu(layer(xu))
        B = self.B_layers[-1](xu)  # Last layer
        B = torch.reshape(B, (x.shape[0], self.state_size, self.action_size))
        dx = A @ state.unsqueeze(-1) + B @ action.unsqueeze(-1)
        x = target + dx.squeeze()*self.dt
        return x


class ModelBasedEnsemble(object):
    def __init__(self, state_dim, action_dim, hidden_size,  dt, min_state, max_state,
                 logger, fn_normalize, fn_unnormalize,
                 learning_rate=1e-3, weight_decay=1e-4, N=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim, self.action_dim, self.hidden_size, self.dt = state_dim, action_dim, hidden_size, dt
        self.N = N  # Number of models in the ensemble
        self.models = []  # This will store each of the ensemble members
        self.optimizers = []
        self.min_state = min_state.to(self.device)
        self.max_state = max_state.to(self.device)
        for _ in range(self.N):
            model = DynamicsModel(state_dim, action_dim, 
                                  hidden_size, dt, self.min_state,self.max_state).to(self.device)
            # TODO! add min and max state to the model
            # wyh does it work so much worse than the original model?
            # TODO! add additional regularization on theta, we can force consitency (calculate theta from x,y moves and then backprop with that target as well, but is for t-1)
            # model = NewDynamicsModel(fn_normalize=fn_normalize, fn_unnormalize=fn_unnormalize, device=self.device) 
            self.models.append(model)

        self.writer = logger
        self.dynamics_optimizer_iterations = 0

    def forward(self, x, u):
        """
        Forward pass through all models of the ensemble.
        Returns predictions from all ensemble members.
        """
        predictions = []
        for model in self.models:
            prediction, _ = model(x, u)
            predictions.append(prediction)
        
        return predictions
    
    def update(self, states, actions, next_states, step):
        batch_size = states.size(0)
        
        for i, model in enumerate(self.models):
            # Sample with replacement to create a bootstrapped batch
            indices = torch.randint(0, batch_size, (batch_size,)).to(self.device)
            bootstrapped_states = states[indices]
            bootstrapped_actions = actions[indices]
            bootstrapped_next_states = next_states[indices]
            #print(bootstrapped_actions.device)

            # Reset gradients for the current model
            loss = model.update(bootstrapped_states, bootstrapped_actions, bootstrapped_next_states)
            
            if self.dynamics_optimizer_iterations % 10 == 0:
                self.writer.add_scalar(f"train/loss{i}/xy", loss[0], global_step=self.dynamics_optimizer_iterations)
                self.writer.add_scalar(f"train/loss{i}/theta", loss[1], global_step=self.dynamics_optimizer_iterations)
                self.writer.add_scalar(f"train/loss{i}/vel", loss[2], global_step=self.dynamics_optimizer_iterations)
                self.writer.add_scalar(f"train/loss{i}/progr", loss[3], global_step=self.dynamics_optimizer_iterations)
                # add mean of all loses
                #self.writer.add_scaler(f"train/loss{i}/mean", loss[0], global_step=self.dynamics_optimizer_iterations)
            self.dynamics_optimizer_iterations += 1
        #self.dynamics_optimizer_iterations = 0
    def __call__(self,x,u):
        predictions = self.forward(x, u)
        #print(predictions)
        avg_prediction = torch.mean(torch.stack(predictions), dim=0)
        return avg_prediction
    
    def save(self, path, filename):
        #print("saving model right now not implemented")
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(path, f"{filename}_{i}.pth"))

        #pass
        #for model in self.models:
    def load(self, path, filename):
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(path, f"{filename}_{i}.pth")))

class ModelBased2(object):
    def __init__(self, state_dim, action_dim, hidden_size, dt, min_state,max_state,
                 logger, dataset, fn_normalize, fn_unnormalize, use_reward_model=False,
                 learning_rate=1e-3, weight_decay=1e-4,target_reward=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim, self.action_dim, self.hidden_size, self.dt = state_dim, action_dim, hidden_size, dt
        #self.dynamics_model = DynamicsModelAction()
        
        self.dynamics_model = ModelBasedEnsemble(state_dim, 
                                                   action_dim,
                                                    hidden_size,
                                                    dt,
                                                    min_state,
                                                    max_state,
                                                    #tf_to_torch(min_state).to(self.device),
                                                    #tf_to_torch(max_state).to(self.device),
                                                    logger, fn_normalize, fn_unnormalize,
                                                    learning_rate=learning_rate,
                                                    weight_decay=weight_decay,
                                                    N=3)
        
        #DynamicsModelPolicy(state_dim, action_dim, hidden_size, dt, logger,
                              #                    learning_rate=learning_rate, weight_decay=weight_decay)
        #self.min_state = 
        #self.max_state = 
        self.use_reward_model = use_reward_model
        if use_reward_model:
            self.reward_model = RewardModel(state_dim, action_dim)
            self.done_model = RewardModel(state_dim, action_dim)
            self.reward_model.to(self.device)
            self.done_model.to(self.device)
        
        else:
            # TODO! think about how to do better here
            if target_reward=="trajectories_td_prog.zarr":
                print("[mb] Using progress reward")
                config = Config('config/td_prog_config.json')
                #self.reward_model = GroundTruthRewardFast(dataset,20, config)
            elif target_reward=="trajectories_raceline.zarr":
                print("[mb] Using raceline reward")
                config = Config('config/raceline_config.json')
                #self.reward_model = GroundTruthRewardFast(dataset,20, config)
            elif target_reward=="trajectories_min_act.zarr":
                print("[mb]Using min action reward")
                config = Config('config/min_act_config.json')
                
            else:
                raise NotImplementedError
            print(config)
            self.reward_model = GroundTruthRewardFast(dataset,20, config)
        self.writer=logger
        
        # self.dynamics_model.to(self.device)
        
        #self.optimizer_dynamics = optim.Adam(self.dynamics_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if use_reward_model:
            self.optimizer_reward = optim.Adam(self.reward_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.optimizer_done = optim.Adam(self.done_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.dynamics_optimizer_iterations = 0

    def update(self, states, actions, next_states, rewards, masks, weights):
        #states = tf_to_torch(states)
        #actions = tf_to_torch(actions)
        #next_states = tf_to_torch(next_states)
        #rewards = tf_to_torch(rewards)
        #masks = tf_to_torch(masks)
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        masks = masks.to(self.device)

        self.dynamics_model.update(states, actions, next_states, self.dynamics_optimizer_iterations)
        # Update model parameters
        
        if self.use_reward_model:
            # Update reward model
            self.optimizer_reward.zero_grad()  # Reset gradients
            pred_rewards = self.reward_model(states, actions)
            reward_loss = F.mse_loss(pred_rewards, rewards, reduction='none')
            reward_loss = (reward_loss).mean()
            reward_loss.backward()  # Compute gradients
            self.optimizer_reward.step()  # Update model parameters

            # Update done model
            self.optimizer_done.zero_grad()  # Reset gradients
            pred_dones = self.done_model(states, actions)
            done_loss = F.binary_cross_entropy_with_logits(pred_dones, masks, reduction='none')
            done_loss = (done_loss).mean()
            done_loss.backward()  # Compute gradients
            self.optimizer_done.step()
        
        if self.dynamics_optimizer_iterations % 1000 == 0:
            
            if self.use_reward_model:
                self.writer.add_scalar('cond/train/rew_loss', reward_loss.item(), global_step=self.dynamics_optimizer_iterations)
                self.writer.add_scalar('cond/train/done_loss', done_loss.item(), global_step=self.dynamics_optimizer_iterations)
        self.dynamics_optimizer_iterations += 1
    

    def sample_initial_states(self, initial_mask, min_distance=50, num_samples=20):
        # Find all initial state indices
        initial_indices = np.where(initial_mask == 1)[0]

        # Filter out the indices that don't have at least min_distance before the next initial state
        valid_indices = [idx for i, idx in enumerate(initial_indices[:-1]) if initial_indices[i+1] - idx >= min_distance]

        # If the last state is also valid (i.e., it has more than min_distance states until the end of the array)
        if len(initial_mask) - initial_indices[-1] >= min_distance:
            valid_indices.append(initial_indices[-1])

        # If there are fewer or equal valid indices than num_samples, return all valid indices
        if len(valid_indices) <= num_samples:
            return valid_indices
        
        # Determine the step size to take to get num_samples from the list of valid_indices
        step_size = len(valid_indices) // num_samples

        # Select the indices
        sampled_indices = valid_indices[::step_size][:num_samples]
        
        return sampled_indices

    def plot_rollouts_fast(self, dataset,
                      unnormalize_fn,
                      batch_size = 256, 
                      horizon=100, 
                      num_samples=256, 
                      discount=1.0, 
                      get_target_action=None,
                      use_dynamics=True,
                      path="rollouts_mb_fast.png"):
        
        with torch.no_grad():
            states, actions, rewards, inital_mask = dataset.states, dataset.actions, dataset.rewards, dataset.mask_inital
            raw_actions = dataset.raw_actions
            raw_actions = tf_to_torch(raw_actions).to(self.device)
            states = tf_to_torch(states).to(self.device)
            actions = tf_to_torch(actions).to(self.device)
            rewards = tf_to_torch(rewards).to(self.device)

            sampled_initial_indices = self.sample_initial_states(inital_mask, 
                                                                 min_distance=horizon, 
                                                                 num_samples=num_samples)
            
            sampled_initial_indices = torch.tensor(sampled_initial_indices)

            sampled_states = torch.stack([states[idx:idx + horizon] for idx in sampled_initial_indices])
            sampled_actions = torch.stack([actions[idx:idx + horizon] for idx in sampled_initial_indices])
            # Use a colormap to generate colors
            num_samples = len(sampled_initial_indices)
            colors = cm.viridis(np.linspace(0, 1, num_samples))

            all_states, all_actions = self.fast_rollout(sampled_states, sampled_actions, 
                                                        get_target_action=get_target_action,
                                                        horizon=horizon,
                                                        batch_size=batch_size,
                                                        use_dynamics=use_dynamics)  
            #print(all_states.shape)
            #print(all_states[0])
            #print(states.shape)
            #print(states[sampled_initial_indices[0]:sampled_initial_indices[0]+horizon, 0:2])
            all_states = all_states.cpu().numpy()
            states = states.cpu().numpy()
            plt.scatter(states[:, 0].reshape(-1)
                        ,states[:, 1].reshape(-1), 
            color='grey', s=5, label='All states', alpha=0.5)

            for i,idx in enumerate(sampled_initial_indices):
                trajectory = all_states[i,:, :2]
                # print(trajectory.shape)
                plt.scatter(trajectory[0, 0], trajectory[0, 1], color=colors[i], marker='x', s=60, label=f'Start {idx + 1}')
                plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"Rollout {idx + 1}", color=colors[i])
                plt.plot(states[idx:idx+horizon, 0], states[idx:idx+horizon, 1],'--', label=f"Ground Truth {idx + 1}", color=colors[i])
            plt.title("Rollouts using Given Actions (torch)")
            # plt.legend()
            plt.savefig(f"{path}")
            plt.clf()


    def estimate_returns_fast():
        pass

    def evaluate_fast(self, dataset,
                      unnormalize_fn,
                      batch_size = 256, 
                      horizon=100, 
                      num_samples=256, 
                      discount=1.0, 
                      get_target_action=None,
                      tag = "test"):

        with torch.no_grad():
            states, actions, rewards, inital_mask = dataset.states, dataset.actions, dataset.rewards, dataset.mask_inital
            raw_actions = dataset.raw_actions
            raw_actions = raw_actions.to(self.device)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)

            sampled_initial_indices = [0] #self.sample_initial_states(inital_mask, 
                                          #                       min_distance=horizon, 
                                          #                       num_samples=num_samples)
            
            sampled_initial_indices = torch.tensor(sampled_initial_indices)
            # calculate the ground truth of the rollouts
            discount_factors = torch.tensor([discount**i for i in range(horizon)]).to(self.device)
            gt_rewards_segments = rewards[sampled_initial_indices[:, None] + torch.arange(horizon)].to(self.device)
            gt_rewards = (gt_rewards_segments * discount_factors).sum(dim=1)
            mean_gt_rewards = unnormalize_fn(gt_rewards.mean().item())

            # create sampled_states
            sampled_states = torch.stack([states[idx:idx + horizon] for idx in sampled_initial_indices])
            sampled_actions = torch.stack([actions[idx:idx + horizon] for idx in sampled_initial_indices])


            all_states, all_actions = self.fast_rollout(sampled_states, sampled_actions, 
                                                        get_target_action=get_target_action,
                                                        horizon=horizon,
                                                        batch_size=batch_size,
                                                        use_dynamics=True)
            

            all_rewards = self.reward_model(all_states, all_actions)
            
            discount_factors = torch.tensor([discount**i for i in range(horizon)]).to(self.device)
            all_rewards = torch.tensor(all_rewards).to(self.device) * discount_factors
            all_rewards = all_rewards.sum(dim=1)
            all_rewards = all_rewards.cpu().numpy()
            mean_model_rewards = np.mean(np.asarray(all_rewards))

            mean_gt_rewards = mean_gt_rewards.cpu().numpy()

            self.writer.add_scalar(f"test/mean_gt_rewards_{tag}", mean_gt_rewards, global_step=self.dynamics_optimizer_iterations)
            self.writer.add_scalar(f"test/mean_model_rewards_{tag}", mean_model_rewards, global_step=self.dynamics_optimizer_iterations)
            self.writer.add_scalar(f"test/diff_rewards_{tag}", mean_gt_rewards-mean_model_rewards, global_step=self.dynamics_optimizer_iterations)
    """
    @input states: (batch, timestep, state_dim)
    @input actions: (batch, timestep, action_dim)
    performs parallel rollouts and returns
    a rollout matrix of (batch, timestep, state_dim)
    """
    def fast_rollout(self, states, actions, 
                     get_target_action=None, 
                     horizon = 10,
                     batch_size = 256,
                     use_dynamics=True,
                     skip_first_action=False):
        
        assert (len(states.shape) == 3) # (batch, timestep, state_dim)
        assert (len(actions.shape) == 3) # (batch, timestep, action_dim)
        assert (states.shape[0] == actions.shape[0])
        assert (states.shape[1] == actions.shape[1])
        assert (states.shape[2] == 11)
        assert (actions.shape[2] == 2)
        with torch.no_grad():
            states_initial = states[:,0,:] # get the first timesteps from the states
            state_batches = torch.split(states_initial, batch_size) # only do rollouts from timestep 0

            all_states = torch.zeros((0, horizon, states[0].shape[-1])).to(self.device)
            all_actions = torch.zeros((0, horizon, 2)).to(self.device)
            for num_batch, state_batch in enumerate(state_batches):
                # (batch,11)
                all_state_batches = torch.zeros((state_batch.shape[0], 0, state_batch[0].shape[-1])).to(self.device)
                # now includes the first action that is always a (0,0)
                all_actions_batches = torch.zeros((state_batch.shape[0], 0, 2)).to(self.device)
                # add the first state to all_state_batches
                # already handled by the for loop
                # all_state_batches = torch.cat([all_state_batches, state_batch.unsqueeze(1)], dim=1)
                #print(state_batch.shape)
                # print("ji")
                for i in range(horizon):
                    assert len(state_batch.shape) == 2
                    assert state_batch.shape[0] <= batch_size
                    assert state_batch.shape[1] == 11
                    if get_target_action is None:
                        if i == 0:
                            action = np.zeros((state_batch.shape[0],2),dtype=np.float32)
                            
                            action = torch.tensor(action).to(self.device)
                        else:
                            action = actions[num_batch*batch_size:batch_size*(num_batch+1),i -1,:] # (batch,2)
                            action = action.float()
                        #print(action.shape)
                        assert(action.shape[0] == state_batch.shape[0])
                        assert(action.shape[1]==2)
                    else:
                        if i == 0 and not skip_first_action: #TODO! this is missing in the dataset!
                            action = np.zeros((state_batch.shape[0],2),dtype=np.float32)
                            # action to torch
                            action = torch.tensor(action).to(self.device)
                        else:
                            action = get_target_action(state_batch) #.to('cpu').numpy())

                            assert(action.shape[0] == state_batch.shape[0])
                            assert(action.shape[1]==2)
                            action = tf_to_torch(action).to(self.device)
                            #make dtype float32
                            action = action.float()
                    # add the action to all_batch_actions along dim=1
                    all_actions_batches = torch.cat([all_actions_batches, action.unsqueeze(1)], dim=1)
                    all_state_batches = torch.cat([all_state_batches, state_batch.unsqueeze(1)], dim=1)
                    if use_dynamics:
                        #print(state_batch)
                        state_batch = self.dynamics_model(state_batch, action)
                        #print("-----")
                        #print(state_batch)
                    elif horizon-1 != i:
                        state_batch = states[:,i+1,:]
                all_states = torch.cat([all_states, all_state_batches], dim=0)
                all_actions = torch.cat([all_actions, all_actions_batches], dim=0)
            return all_states, all_actions
    
    def estimate_returns(self, inital_states,
                         inital_weights,
                         get_target_action,
                            horizon=50,
                            discount=0.99,
                            batch_size=256):
        
        with torch.no_grad():
            states = tf_to_torch(inital_states)
            # add timestep dimension
            states = states.unsqueeze(1).to(self.device)
            #states = states[0].unsqueeze(0).unsqueeze(0).to(self.device)
            #print(states.shape)
            actions = torch.zeros((states.shape[0], 1, 2)).to(self.device)
            all_states, all_actions = self.fast_rollout(states, actions, 
                                            get_target_action=get_target_action,
                                            horizon=horizon,
                                            batch_size=batch_size,
                                            use_dynamics=True)

            #print(all_states.shape)
            
            #print(all_states)
            #print("----")
            #print(all_actions)
            all_rewards = self.reward_model(all_states, all_actions)
            #print(all_rewards.shape)
            
            discount_factors = torch.tensor([discount**i for i in range(horizon)]).to(self.device)
            all_rewards = torch.tensor(all_rewards).to(self.device) * discount_factors
            all_rewards = all_rewards.sum(dim=1)
            all_rewards = all_rewards.cpu().numpy()
            mean_model_rewards = np.mean(np.asarray(all_rewards))
            std_model_rewards = np.std(np.asarray(all_rewards))
            #print(mean_model_rewards)
            #exit()
            return mean_model_rewards * (1-discount), std_model_rewards * (1-discount)
            
    
    def save(self, save_path, filename="model_based2_torch_checkpoint.pth"):
        """
        Save the model's state dictionaries.
        
        Args:
        - save_path (str): The directory path where the model should be saved.
        - filename (str, optional): The name of the checkpoint file. Defaults to "model_based2_checkpoint.pth".
        """
        
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Define the checkpoint
        if self.use_reward_model:
            # throw not impleemneted error
            raise NotImplementedError
        else:
            self.dynamics_model.save(save_path, filename)


    def load(self, checkpoint_path, filename):
        """
        Load the model's state dictionaries from a checkpoint.
        
        Args:
        - checkpoint_path (str): The path to the saved checkpoint file.
        """

        # Load the checkpoint
        # checkpoint = torch.load(checkpoint_path)

        # Restore the state dictionaries
        self.dynamics_model.load(checkpoint_path, filename) #checkpoint["dynamics_model_state_dict"])
        """
        if self.use_reward_model:
            self.reward_model.load_state_dict(checkpoint["reward_model_state_dict"])
            self.done_model.load_state_dict(checkpoint["done_model_state_dict"])
            
        # Move models to the appropriate device
        self.dynamics_model.to(self.device)
        if self.use_reward_model:
            self.reward_model.to(self.device)
            self.done_model.to(self.device)
        """