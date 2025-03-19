import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class QNetwork(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, buffer_size=10000):
        self.memory = deque(maxlen=buffer_size)
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

class DDQNAgent:
    """Double Deep Q-Network agent implementation"""
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 batch_size=64, update_every=4):
        """
        Initialize the DDQN agent
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration rate decay factor
            batch_size: Mini-batch size
            update_every: How often to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_every = update_every
        self.t_step = 0
        
        # Q-Networks (online and target)
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayBuffer()
        
        # Initialize target network weights with online network weights
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network with online network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def step(self, state, action, reward, next_state, done):
        """
        Add experience to memory and learn if it's time
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.learn()
    
    def act(self, state, eval_mode=False):
        """
        Choose action according to epsilon-greedy policy
        
        Args:
            state: Current state
            eval_mode: If True, use greedy policy (no exploration)
        
        Returns:
            Selected action
        """
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Set network to eval mode
        self.q_network.eval()
        
        with torch.no_grad():
            action_values = self.q_network(state)
            
        # Set network back to train mode
        self.q_network.train()
        
        # Epsilon-greedy action selection
        if not eval_mode and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())
    
    def learn(self):
        """Update online network using sampled experiences"""
        # Sample a batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            # DDQN: Use online network to select action and target network to evaluate it
            # a' = argmax_a Q_online(s', a)
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            # Q(s', a') from target network
            Q_targets_next = self.target_network(next_states).gather(1, next_actions)
            # Compute Q targets for current states
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from online model
        Q_expected = self.q_network(states).gather(1, actions)
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        
        # Update target network
        self.update_target_network()
        
        return loss.item()
    
    def save(self, filename):
        """Save trained model"""
        torch.save(self.q_network.state_dict(), filename)
        
    def load(self, filename):
        """Load trained model"""
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(torch.load(filename))