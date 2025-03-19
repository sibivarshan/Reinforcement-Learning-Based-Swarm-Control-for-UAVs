import numpy as np

class VirtualLeader:
    """
    Implementation of virtual leader for UAV swarm
    """
    def __init__(self, initial_position, initial_velocity, dt=0.1):
        """
        Initialize virtual leader
        
        Args:
            initial_position: Initial position [x, y]
            initial_velocity: Initial velocity [vx, vy]
            dt: Time step for integration
        """
        self.position = np.array(initial_position)
        self.velocity = np.array(initial_velocity)
        self.dt = dt
        
        # Position and velocity history (for visualization)
        self.position_history = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
    
    def step(self, action):
        """
        Update virtual leader state based on action
        
        Args:
            action: Control input changes [Δu_x, Δu_y]
        
        Returns:
            Updated position and velocity
        """
        # Update velocity based on action
        self.velocity += action
        
        # Update position
        self.position += self.velocity * self.dt
        
        # Store history
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        
        return self.position, self.velocity

class RLVirtualLeader(VirtualLeader):
    """
    RL-based virtual leader for UAV swarm
    """
    def __init__(self, initial_position, initial_velocity, agent, action_mapping, dt=0.1):
        """
        Initialize RL-based virtual leader
        
        Args:
            initial_position: Initial position [x, y]
            initial_velocity: Initial velocity [vx, vy]
            agent: RL agent for decision making
            action_mapping: Mapping from discrete actions to continuous control changes
            dt: Time step for integration
        """
        super().__init__(initial_position, initial_velocity, dt)
        self.agent = agent
        self.action_mapping = action_mapping
        self.last_state = None
        self.last_action = None
        
    def observe_environment(self, uav_swarm, obstacles):
        """
        Create state representation from environment
        
        Args:
            uav_swarm: List of UAVs in swarm
            obstacles: List of obstacles
        
        Returns:
            State vector for the RL agent
        """
        # Get leader position and velocity
        leader_pos = self.position
        leader_vel = self.velocity
        
        # Compute swarm centroid and average velocity
        if len(uav_swarm) > 0:
            swarm_positions = np.array([uav.position[:2] for uav in uav_swarm])
            swarm_velocities = np.array([uav.velocity_vector[:2] for uav in uav_swarm])
            
            swarm_centroid = np.mean(swarm_positions, axis=0)
            swarm_avg_vel = np.mean(swarm_velocities, axis=0)
        else:
            swarm_centroid = leader_pos
            swarm_avg_vel = leader_vel
        
        # Find closest obstacle
        min_dist = float('inf')
        closest_obs_pos = np.zeros(2)
        closest_obs_vel = np.zeros(2)
        
        for obs in obstacles:
            # Get filtered position for dynamic obstacles
            obs_pos = obs.get_filtered_position()[:2]
            
            # Calculate distance
            dist = np.linalg.norm(leader_pos - obs_pos) - obs.radius
            
            if dist < min_dist:
                min_dist = dist
                closest_obs_pos = obs_pos
                if hasattr(obs, 'velocity'):
                    closest_obs_vel = obs.velocity[:2]
                else:
                    closest_obs_vel = np.zeros(2)
        
        # Build state vector:
        # - Leader position relative to swarm centroid
        # - Leader velocity
        # - Closest obstacle position relative to leader
        # - Closest obstacle velocity
        # - Distance to closest obstacle
        
        state = np.concatenate([
            leader_pos - swarm_centroid,  # Relative position to swarm
            leader_vel,                   # Leader velocity
            leader_pos - closest_obs_pos, # Relative position to obstacle
            closest_obs_vel,              # Obstacle velocity
            [min_dist]                    # Distance to closest obstacle
        ])
        
        self.last_state = state
        return state
    
    def select_action(self, uav_swarm, obstacles):
        """
        Select action based on current environment state using the RL agent
        
        Args:
            uav_swarm: List of UAVs in swarm
            obstacles: List of obstacles
        
        Returns:
            Selected action as [Δu_x, Δu_y]
        """
        # Get state observation
        state = self.observe_environment(uav_swarm, obstacles)
        
        # Select action using agent
        action_idx = self.agent.select_action(state)
        self.last_action = action_idx
        
        # Convert discrete action to continuous control changes
        control_changes = self.action_mapping[action_idx]
        
        return control_changes
    
    def update_agent(self, reward, done):
        """
        Update the agent with the latest experience
        
        Args:
            reward: Reward value received
            done: Whether the episode is done
        """
        if self.last_state is not None and self.last_action is not None:
            # Get current state
            state = self.last_state
            action = self.last_action
            
            # Get next state (after action was applied)
            next_state = self.observe_environment([], [])  # Empty lists as placeholders
            
            # Store transition in agent's memory
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent (train)
            self.agent.update()
    
    def save_history(self, filename):
        """
        Save trajectory history to file
        
        Args:
            filename: Output filename
        """
        history = {
            'positions': np.array(self.position_history),
            'velocities': np.array(self.velocity_history)
        }
        np.save(filename, history)
    
    def load_history(self, filename):
        """
        Load trajectory history from file
        
        Args:
            filename: Input filename
        """
        history = np.load(filename, allow_pickle=True).item()
        self.position_history = list(history['positions'])
        self.velocity_history = list(history['velocities'])
        self.position = self.position_history[-1].copy()
        self.velocity = self.velocity_history[-1].copy()