"""
Default configuration parameters for UAV swarm control simulation
"""

class Config:
    """Configuration parameters for simulation"""
    
    def __init__(self):
        # Environment parameters
        self.num_uavs = 5
        self.num_obstacles = 5
        self.world_size = (500, 500)
        self.max_steps = 1000
        
        # UAV parameters
        self.uav_speed_min = 5.0
        self.uav_speed_max = 15.0
        self.uav_load_factor = 3.0
        
        # Flocking control parameters
        self.c1 = 0.5  # Neighbor position coefficient
        self.c2 = 0.5  # Neighbor velocity coefficient
        self.c3 = 0.8  # Leader position coefficient
        self.c4 = 0.5  # Leader velocity coefficient
        self.c5 = 1.0  # Obstacle avoidance coefficient
        self.d = 0.1   # Parameter for network potential
        self.perception_radius = 50.0  # Communication radius
        
        # Artificial Potential Field parameters
        self.eta = 100.0  # Repulsive potential gain
        self.rho0 = 50.0  # Influence radius
        self.c6 = 2.5     # Filter parameter for dynamic obstacles
        
        # DDQN parameters
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.buffer_size = 100000
        self.batch_size = 64
        self.target_update_freq = 10
        
        # Reward parameters
        self.collision_penalty = -100.0
        self.goal_reward = 100.0
        self.step_penalty = -0.1
        self.obstacle_penalty_factor = 10.0
        
        # Visualization parameters
        self.render_interval = 1  # How often to render (every n steps)
        self.save_interval = 50   # How often to save model (every n episodes)
        
    def get_flocking_params(self):
        """Return flocking parameters as a dictionary"""
        return {
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
            'c4': self.c4,
            'c5': self.c5,
            'd': self.d,
            'perception_radius': self.perception_radius
        }
        
    def get_apf_params(self):
        """Return APF parameters as a dictionary"""
        return {
            'eta': self.eta,
            'rho0': self.rho0,
            'c6': self.c6
        }