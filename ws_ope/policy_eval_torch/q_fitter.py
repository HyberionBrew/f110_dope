import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNet(nn.Module):
    """A critic network that estimates a dual Q-function."""
    
    def __init__(self, state_dim, action_dim):
        """Creates networks.
        
        Args:
          state_dim: State size.
          action_dim: Action size.
        """
        super(CriticNet, self).__init__()
        #self.fc1 = nn.Linear(state_dim + action_dim, 1)
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.squeeze(self.fc3(x), -1)
        #return torch.squeeze(self.fc1(x), -1)

class QFitter(nn.Module):
    """A critic network that estimates a dual Q-function."""

    def __init__(self, state_dim, action_dim, critic_lr, weight_decay, tau, discount, use_time=False, timestep_constant=0.01, log_frequency=500, writer=None):
        """Creates networks.
        
        Args:
          state_dim: State size.
          action_dim: Action size.
          critic_lr: Critic learning rate.
          weight_decay: Weight decay.
          tau: Soft update discount.
        """
        super(QFitter, self).__init__()
        if use_time:
            state_dim += 1
            self.use_time = True
            self.timestep_constant = timestep_constant

        self.critic = CriticNet(state_dim, action_dim)
        self.critic_target = CriticNet(state_dim, action_dim)

        self.tau = tau
        self.soft_update(self.critic, self.critic_target, tau=1.0)

        self.optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        self.log_frequency = log_frequency
        self.optimizer_iterations = 0
        self.writer = writer
        self.discount = discount
    # write device attibute
    def to(self, device):
        self.critic.to(device)
        self.critic_target.to(device)
        return self

    def soft_update(self, local_model, target_model, tau=0.005):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def forward(self, states, actions, timesteps=None, batch_size=1000):
        # batch this call to avoid OOM
        n_data = states.shape[0]
        results = []

        if self.use_time:
            if timesteps is None:
                timesteps = torch.zeros_like(states[:, -1]).unsqueeze(-1)
            states = torch.cat([states, timesteps], dim=1)

        for i in range(0, n_data, batch_size):
            batch_states = states[i: min(i + batch_size, n_data)]
            batch_actions = actions[i: min(i + batch_size, n_data)]
            batch_result = self.critic_target(batch_states, batch_actions)
            results.append(batch_result)

        final_result = torch.cat(results, dim=0)
        return final_result / (1 - self.discount)
    
    def update(self, states, actions, next_states, next_actions, rewards, masks, weights, discount, min_reward, max_reward, timesteps=None):
        """Updates critic parameters."""

        if self.use_time:
            states = torch.cat([states, timesteps], dim=1)
            next_states = torch.cat([next_states, timesteps + self.timestep_constant], dim=1)
        """
        print(states.shape)
        print(states[0])
        print(actions.shape)
        print(actions[0])
        print("------")
        print(next_states.shape)
        print(next_states[0])
        print(next_actions.shape)
        print(next_actions[0])
        """
        #exit()
        with torch.no_grad():
            next_q = self.critic_target(next_states, next_actions) / (1 - discount)
            #print("-------")
            #print(rewards.mean())
            #print(masks.mean())
            #print(next_q.mean())
            target_q = (rewards + discount * masks * next_q) #*(1-discount) #here
            #print("hello", target_q.mean())
            target_q = torch.clamp(target_q, min_reward,max_reward)
        
        
        self.optimizer.zero_grad()

        q = self.critic(states, actions) / (1 - discount) #here
        #print(target_q.mean())
        #print(q.mean())
        critic_loss = (torch.sum((target_q - q) ** 2 * weights) / torch.sum(weights))

        # Zero gradients before backward pass
        
        critic_loss.backward()
        self.optimizer.step()

        self.soft_update(self.critic, self.critic_target, self.tau)

        # Logging (this is a placeholder, adjust as per your logging setup)
        if self.optimizer_iterations % self.log_frequency == 0:
            self.writer.add_scalar('train/fqe/loss', critic_loss.item(), global_step=self.optimizer_iterations)
        self.optimizer_iterations += 1
        return critic_loss.item()
    """
    def estimate_returns(self, initial_states, initial_weights, get_action, action=None, timesteps=None):
        # Estimate returns with fitted q learning.
        with torch.no_grad():
            if action is not None:
                print("action not none")
                initial_actions = action
            else:
                initial_actions = get_action(initial_states)
            if timesteps is not None:
                print("timesteps not none")
                preds = self(initial_states, initial_actions, timesteps=timesteps)
            else:
                print("yo")
                print(initial_states.shape)
                print(initial_actions.shape)
                preds = self(initial_states, initial_actions)
            print("preds", preds.shape)
            print(preds)
            print(initial_weights)
            print("initial_weights", initial_weights.shape)
            print("inital_weights", torch.sum(initial_weights))
            print(torch.sum(preds * initial_weights))
            weighted_mean = torch.sum(preds * initial_weights) / torch.sum(initial_weights)
            print("weighted", weighted_mean)
            weighted_variance = torch.sum(initial_weights * (preds - weighted_mean) ** 2) / torch.sum(initial_weights)
            weighted_stddev = torch.sqrt(weighted_variance)
        return weighted_mean, weighted_stddev
    """

    def estimate_returns(self, initial_states, initial_weights, get_action, action_timesteps, action=None, timesteps=None, reduction=True):
        """Estimate returns with fitted q learning."""
        with torch.no_grad():
            weighted_stddev = 0.0
            if action is not None:
                initial_actions = action
            else:
                initial_actions = get_action(initial_states,action_timesteps)
            if timesteps is not None:
                preds = self(initial_states, initial_actions, timesteps=timesteps)
            else:

                preds = self(initial_states, initial_actions)
            if reduction==False:
                weighted_mean = preds
            else:
                weighted_mean = torch.sum(preds * initial_weights) / torch.sum(initial_weights)
                weighted_variance = torch.sum(initial_weights * (preds - weighted_mean) ** 2) / torch.sum(initial_weights)
                weighted_stddev = torch.sqrt(weighted_variance)
        return weighted_mean , weighted_stddev

    def estimate_returns_unweighted(self, initial_states, get_action):
        """Estimate returns unweighted."""
        with torch.no_grad():
            initial_actions = get_action(initial_states)
            preds = self(initial_states, initial_actions)
        return preds 
    
    def save(self, path, i=0):
        """Save the model."""
        torch.save(self.critic.state_dict(), path + f"/critic{i}.pth")
        torch.save(self.critic_target.state_dict(), path + f"/critic_target{i}.pth")
    
    def load(self, path, i=0):
        """Load the model."""
        self.critic.load_state_dict(torch.load(path + f"/critic{i}.pth"))
        self.critic_target.load_state_dict(torch.load(path + f"/critic_target{i}.pth"))


