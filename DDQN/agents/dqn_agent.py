import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from models.networks import DQNetwork

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

class DQNAgent:
    """
    Implementation of vanilla Deep Q-Network agent
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, update_every=4):
        """
        Initialize the DQN agent
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration rate decay factor
            buffer_size: Size of replay buffer
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
        self.q_network = DQNetwork(state_size, action_size)
        self.target_network = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize target network weights with online network weights
        self.update_target_network()
        
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        
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
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
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
        """Update online network parameters"""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors and send to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Get current Q values
        q_values = self.q_network(states).gather(1, actions)
        
        # Get max predicted Q values for next states from target model
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
            
        # Compute loss (MSE or Huber loss)
        loss = F.mse_loss(q_values, target_q_values)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        
        # Periodically update target network
        if self.t_step == 0:
            self.update_target_network()
        
        return loss.item()
    
    def save(self, filepath):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
    def load(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']