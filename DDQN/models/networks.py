import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture for UAV swarm control
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super(DQNetwork, self).__init__()
        
        # Create layers dynamically based on hidden_dims
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        # Combine all layers
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.layers(x)

class DuelingDQNetwork(nn.Module):
    """
    Dueling Deep Q-Network architecture with advantage and value streams
    This is an advanced architecture used in some papers to improve performance
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DuelingDQNetwork, self).__init__()
        
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
    
    def forward(self, x):
        """
        Forward pass with dueling architecture
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        """
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage streams
        # Subtract mean advantage to ensure identifiability
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for potential extension to policy gradient methods
    Not directly used in the paper but could be useful for extensions
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value