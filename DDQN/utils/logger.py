import os
import time
import json
import numpy as np
from datetime import datetime

class Logger:
    """Logger for training and testing results"""
    
    def __init__(self, log_dir="logs"):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate a unique ID for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        
        # Create run directory
        self.run_dir = os.path.join(log_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "episode_rewards": [],
            "avg_rewards": [],
            "collision_counts": [],
            "goal_reached_counts": [],
            "epsilon_values": [],
            "losses": []
        }
        
        # Initialize episode data
        self.episode_data = {}
        
        # Track time
        self.start_time = time.time()
        
    def log_episode(self, episode, reward, collision, goal_reached, epsilon, loss=None, additional_data=None):
        """
        Log data from a single episode
        
        Args:
            episode: Episode number
            reward: Total reward for the episode
            collision: Whether collision occurred
            goal_reached: Whether goal was reached
            epsilon: Current epsilon value for exploration
            loss: Loss value from training (if available)
            additional_data: Dict of additional data to log
        """
        # Update metrics
        self.metrics["episode_rewards"].append(reward)
        self.metrics["collision_counts"].append(1 if collision else 0)
        self.metrics["goal_reached_counts"].append(1 if goal_reached else 0)
        self.metrics["epsilon_values"].append(epsilon)
        
        if loss is not None:
            self.metrics["losses"].append(loss)
        
        # Calculate moving average reward
        window_size = min(10, len(self.metrics["episode_rewards"]))
        avg_reward = sum(self.metrics["episode_rewards"][-window_size:]) / window_size
        self.metrics["avg_rewards"].append(avg_reward)
        
        # Create episode data entry
        episode_entry = {
            "episode": episode,
            "reward": reward,
            "collision": collision,
            "goal_reached": goal_reached,
            "epsilon": epsilon,
            "avg_reward": avg_reward,
            "elapsed_time": time.time() - self.start_time
        }
        
        if loss is not None:
            episode_entry["loss"] = loss
            
        if additional_data:
            episode_entry.update(additional_data)
            
        self.episode_data[episode] = episode_entry
        
        # Save metrics periodically
        if episode % 10 == 0:
            self.save_metrics()
            
    def log_test_results(self, success_rate, collision_rate, avg_steps, avg_reward):
        """
        Log test results
        
        Args:
            success_rate: Rate of successful goal reaching
            collision_rate: Rate of collisions
            avg_steps: Average steps to complete episodes
            avg_reward: Average reward per episode
        """
        test_results = {
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save test results
        with open(os.path.join(self.run_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)
            
    def save_metrics(self):
        """Save metrics to files"""
        # Save metrics as JSON
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
            
        # Save episode data
        with open(os.path.join(self.run_dir, "episode_data.json"), "w") as f:
            json.dump(self.episode_data, f, indent=2)
            
        # Save metrics as NumPy arrays for easy loading
        for key, values in self.metrics.items():
            if values:  # Only save non-empty lists
                np.save(os.path.join(self.run_dir, f"{key}.npy"), np.array(values))
                
    def log_config(self, config):
        """
        Log configuration parameters
        
        Args:
            config: Configuration object or dictionary
        """
        if hasattr(config, "__dict__"):
            config_dict = vars(config)
        else:
            config_dict = config
            
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
            
    def log_model_summary(self, model):
        """
        Log model summary
        
        Args:
            model: PyTorch model
        """
        with open(os.path.join(self.run_dir, "model_summary.txt"), "w") as f:
            f.write(str(model))