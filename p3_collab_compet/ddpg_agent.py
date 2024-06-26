import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_Actor = 1e-4         # learning rate 
LR_CRITIC = 1e-4        # learning rate


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Policy: state -> max action"""
    
    def __init__(self, state_size, action_size, seed):
        super(Actor, self).__init__()
        
        """
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.uniform_(self.fc1.weight.data, *hidden_init(self.fc1))
        torch.nn.init.uniform_(self.fc2.weight.data, *hidden_init(self.fc2))
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic: state, action -> Q value"""
    
    def __init__(self, state_size, action_size, seed):
        super(Critic, self).__init__()
        
        """
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512 + action_size, 256)
        self.fc3 = nn.Linear(256, 1)
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.uniform_(self.fcs1.weight.data, *hidden_init(self.fcs1))
        torch.nn.init.uniform_(self.fc2.weight.data, *hidden_init(self.fc2))
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define ReplayBuffer to store data and OUNoise for action generation.

import copy
import random
import numpy as np
from collections import namedtuple, deque


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.state = copy.copy(self.mu)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, memory, state_size, action_size, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            action_limit (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor and critic
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
                
        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor_local.parameters(), lr=LR_Actor)
        self.optimizer_critic = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)
        
        self.noise = OUNoise(action_size, seed)

        # Replay memory
        self.memory = memory
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps):
        """Returns action for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        
        if random.random() < eps:
            action += self.noise.sample()

        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.actor_local.train()
        self.critic_local.train()
        
        states, actions, rewards, next_states, dones = experiences
        
        # ------------------- update critic ------------------- #
        state_action_values = self.critic_local(states, actions)
        
        with torch.no_grad():
            max_actions = self.actor_target(next_states)
            next_state_values = self.critic_target(next_states, max_actions)
        expected_state_action_values = (1 - dones) * (next_state_values * gamma) + rewards
        
        loss_critic = F.mse_loss(state_action_values, expected_state_action_values)
        
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()
        
        # ------------------- update actor ------------------- #
        self.critic_local.eval()
        for params in self.critic_local.parameters():
            params.requires_grad = False
        
        pred_actions = self.actor_local(states)
        loss_actor = -self.critic_local(states, pred_actions).mean()
        
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()
        
        for params in self.critic_local.parameters():
            params.requires_grad = True
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class Group():
    def __init__(self, num_agents, state_size, action_size, random_seed):
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.agents = [Agent(self.memory, state_size, action_size, random_seed) for _ in range(num_agents)]
        
    def reset(self):
        for agent in self.agents:
            agent.reset()
        
    def step(self, states, actions, rewards, next_states, dones):
        for agent, state, action, reward, next_state, done in \
        zip(self.agents, states, actions, rewards, next_states, dones):
            agent.step(state, action, reward, next_state, done)
        
    def act(self, states, eps):
        actions = list()
        for agent, state in zip(self.agents, states):
            actions.append(agent.act(state, eps))
        return actions
    
    def checkpoint(self):
        return [{
            'actor': agent.actor_local.state_dict(),
            'critic': agent.critic_local.state_dict()
        } for agent in self.agents]