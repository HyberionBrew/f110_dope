import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_raw_action(state, action,normalize_fn,unnormalize_fn):
    state = unnormalize_fn(state)
    prev_action = state[..., 7:9]
    raw_action = prev_action + torch.clip(action, -1, 1) * 0.05
    state[..., 7:9] = raw_action
    state = normalize_fn(state)
    return state[..., 7:9]

class XYModel(nn.Module):
    def __init__(self, dt, target_size = 2, state_size=5, action_size=2, hidden_size=[64], output_size=2):
        super().__init__()
        self.state_size = state_size
        self.dt = dt
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.target_size = target_size
        self.A_layers = nn.ModuleList()
        A_size = state_size * target_size
        B_size = action_size * target_size
        self.A_layers.append(nn.Linear(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.A_layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.A_layers.append(nn.Linear(hidden_size[-1], A_size))

        self.B_layers = nn.ModuleList()
        self.B_layers.append(nn.Linear(state_size + action_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.B_layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.B_layers.append(nn.Linear(hidden_size[-1], B_size))
        # optimizer and stuff


    def forward(self, target, state, action):
        xu = torch.cat((state, action), -1)
        for layer in self.A_layers:
            xu = F.relu(layer(xu))
        A = xu
        #shape to (N, 2, 5)
        #state has shape (N, 5)
        A = torch.reshape(A, (state.shape[0], self.target_size, self.state_size))
        xu = torch.cat((state, action), -1)
        for layer in self.B_layers:
            xu = F.relu(layer(xu))
        B = xu
        B = torch.reshape(B, (state.shape[0], self.target_size, self.action_size))
        dx = A @ state.unsqueeze(-1) + B @ action.unsqueeze(-1)
        x = target + dx.squeeze()*self.dt
        return x
    
    #def update(self, state, action, next_state):

class DeltaThetaModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initializes the DeltaThetaModel.

        Args:
        state_size (int): Size of the state input, including current theta.
        hidden_size (int): Size of the hidden layers.
        """
        super(DeltaThetaModel, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(state_size+ action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # Output layer: 2 units for Δsin(θ) and Δcos(θ)

    def forward(self, state, action):
        """
        Forward pass of the network.

        Args:
        state (torch.Tensor): The state tensor, including current theta.

        Returns:
        torch.Tensor: The network's output for Δsin(θ) and Δcos(θ).
        """
        # Feedforward
        x = torch.cat((state, action), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        # Optionally, you can normalize the output, but it's not strictly necessary
        # for Δsin(θ) and Δcos(θ) as they are changes, not absolute positions on the unit circle
        # output = F.normalize(output, p=2, dim=1)

        return output

class ProgressModel(nn.Module):
    def __init__(self, hidden_size=128):
        """
        Initializes the ProgressModel.

        Args:
        hidden_size (int): Size of the hidden layers.
        """
        super(ProgressModel, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(2, hidden_size)  # Input layer for x, y
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # Output layer: 2 units for sin(theta) and cos(theta)

    def forward(self, coordinates):
        """
        Forward pass of the network.

        Args:
        coordinates (torch.Tensor): The input coordinates (x, y), expected shape [batch_size, 2].

        Returns:
        torch.Tensor: The network's output for sin(theta) and cos(theta).
        """
        # Feedforward
        x = F.relu(self.fc1(coordinates))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        # Normalize the output to ensure it lies on the unit circle
        output = F.normalize(output, p=2, dim=1)

        return output
    
class NewDynamicsModel(object):
    def __init__(self, fn_normalize, fn_unnormalize,
                 learning_rate=1e-4, weight_decay=1e-4,target_reward=None, device='cpu'):
        self.device = device

        self.fn_normalize = fn_normalize
        self.fn_unnormalize = fn_unnormalize
        self.xy = XYModel(1/20).to(device)
        self.vel = XYModel(1/20).to(device)
        self.theta = DeltaThetaModel(5,2).to(device)
        self.progress = ProgressModel().to(device)
        # optimizer for XYModel
        self.optimizerXY = torch.optim.Adam(self.xy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizerTheta = torch.optim.Adam(self.theta.parameters(), lr=0.00001, weight_decay=weight_decay)
        self.optimizerProgress = torch.optim.Adam(self.progress.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizerVel = torch.optim.Adam(self.vel.parameters(), lr=learning_rate, weight_decay=weight_decay)
    def __call__(self, state, action, train=False):
        raw_action = compute_raw_action(state, action,self.fn_normalize, self.fn_unnormalize)
        # now put into XY network
        xy = state[...,:2]
        theta_vel = state[...,2:7]
        # send to torch
        #xy = torch.from_numpy(xy).float()
        #theta_vel = torch.from_numpy(theta_vel).float()
        #raw_action = torch.from_numpy(raw_action).float()
        #print(xy)
        # all = self.all(state, state, action)
        #print(raw_action.device)
        #print(xy.device)
        #print(theta_vel.device)
        xy_ = self.xy(xy, theta_vel, raw_action)

        theta_delta = self.theta(theta_vel, raw_action)
        theta_ = theta_delta + state[...,2:4]
        
        vel = state[...,4:6]
        #print(vel.shape)
        vel_ = self.vel(vel, theta_vel, raw_action)
        #print(vel_.shape)
        if train:
            progress = self.progress(xy)
        else:
            progress = self.progress(xy_)

        # build the next state tensor
        # zeros for one dimension
        zeros = torch.zeros((state.shape[0],1)).to(self.device)
        next_state = torch.cat((xy_, theta_, vel_,zeros,raw_action, progress), -1)
        return next_state,(xy_ , theta_, vel_, progress) #,all
    
    def delta_theta_loss(self,pred_delta, true_delta):
        """
        Computes the loss for predicted changes in theta.

        Args:
        pred_delta (torch.Tensor): Predicted changes, expected shape [batch_size, 2] for Δsin(θ) and Δcos(θ).
        true_delta (torch.Tensor): True changes, same shape as predictions.

        Returns:
        torch.Tensor: The computed loss.
        """

        # This step ensures that both predictions and targets are within a realistic range
        pred_delta_norm = F.normalize(pred_delta, p=2, dim=1)
        true_delta_norm = F.normalize(true_delta, p=2, dim=1)

        # Compute the mean squared error between the normalized predictions and targets
        loss = F.mse_loss(pred_delta_norm, true_delta_norm)

        return loss

    def update(self,state, action, next_state):
        # trainings loop for xy
        self.optimizerXY.zero_grad()
        self.optimizerTheta.zero_grad()
        self.optimizerProgress.zero_grad()
        self.optimizerVel.zero_grad()
         # use forward to get predicted xy
        #print("wdwd")
        #print(state.device)
        #print(action.device)
        next_state_tensor , train_vals = self(state, action, train=True)
        #print(next_state_tensor.shape)
        pred_xy, pred_theta, pred_vel, pred_progress = train_vals
        #print(pred_xy.shape)
        # extract real xy from next_state
        label_xy = next_state[...,:2]
        # compute loss
        loss_xy = F.mse_loss(pred_xy, label_xy)
        # backprop
        loss_xy.backward()
        # update
        self.optimizerXY.step()

        # trainings loop for theta
        
        label_theta = next_state[...,2:4]
        loss_theta = self.delta_theta_loss(pred_theta, label_theta)
        # print(loss_theta)
        loss_theta.backward()
        self.optimizerTheta.step()
        #print(loss_theta)
        #training vel losss
        label_vel = next_state[...,4:6]
        loss_vel = F.mse_loss(pred_vel, label_vel)
        loss_vel.backward()
        self.optimizerVel.step()


        # trainings loop for progress
        label_progress = state[...,9:11]

        loss_progress = F.mse_loss(pred_progress, label_progress)
        loss_progress.backward()
        self.optimizerProgress.step()

        return loss_xy.item(), loss_theta.item(), loss_vel.item(), loss_progress.item()
        