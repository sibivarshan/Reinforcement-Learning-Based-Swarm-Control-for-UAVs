import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Experience replay memory to store and sample transitions for reinforcement learning.
    
    This implementation includes:
    1. Standard uniform sampling
    2. Storage of (state, action, reward, next_state, done) tuples
    """
    def __init__(self, capacity):
        """
        Initialize replay buffer with specified capacity
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from memory
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            # If buffer doesn't have enough samples, return all available
            samples = list(self.buffer)
        else:
            # Random sampling without replacement
            samples = random.sample(self.buffer, batch_size)
        
        # Transpose the list of tuples into a tuple of lists
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Convert to numpy arrays for batch processing
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)
    
    def is_full(self):
        """Check if the buffer is full"""
        return len(self.buffer) == self.capacity


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer implementation.
    
    Implements prioritized experience replay based on TD-error priorities.
    This helps the agent learn more effectively from important transitions.
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent parameter (0 = uniform sampling, 1 = full prioritization)
            beta_start: Initial importance-sampling correction exponent
            beta_frames: Number of frames over which to anneal beta to 1.0
        """
        self.buffer = []
        self.position = 0
        self.capacity = capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # small constant to ensure non-zero priorities
    
    def beta_by_frame(self, frame_idx):
        """Calculate beta value based on current frame"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done, error=None):
        """
        Add a new experience to memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            error: TD error for prioritization (if None, max priority is used)
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Set priority for new experience
        if error is not None:
            # Priority based on TD error
            priority = (abs(error) + self.epsilon) ** self.alpha
        else:
            # New experiences get max priority to ensure they're sampled at least once
            priority = max_priority
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences based on priorities
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < self.capacity:
            prob = self.priorities[:len(self.buffer)]
        else:
            prob = self.priorities
        
        prob = prob / prob.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)
        
        # Calculate importance sampling weights
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        # Calculate weights
        weights = (len(self.buffer) * prob[indices]) ** (-beta)
        weights = weights / weights.max()
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                indices,
                np.array(weights, dtype=np.float32))
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions
        
        Args:
            indices: Indices of sampled transitions
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (abs(priority) + self.epsilon) ** self.alpha
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)
    
    def is_full(self):
        """Check if the buffer is full"""
        return len(self.buffer) == self.capacity