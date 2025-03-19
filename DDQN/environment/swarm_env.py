import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from environment.uav_dynamics import UAV
from environment.obstacles import StaticObstacle, DynamicObstacle
from controllers.flocking import FlockingController
from controllers.virtual_leader import RLVirtualLeader

class UAVSwarmEnv(gym.Env):
    """
    Environment that simulates a UAV swarm navigating through obstacles
    """
    def __init__(self, 
                 num_uavs=5, 
                 num_static_obstacles=3,
                 num_dynamic_obstacles=2,
                 world_size=(500, 500),
                 max_steps=1000,
                 goal_position=None,
                 flocking_params=None,
                 action_mapping=None):
        """
        Initialize UAV swarm environment
        
        Args:
            num_uavs: Number of UAVs in swarm
            num_static_obstacles: Number of static obstacles
            num_dynamic_obstacles: Number of dynamic obstacles
            world_size: Size of the environment (width, height)
            max_steps: Maximum steps before episode ends
            goal_position: Target position for the swarm
            flocking_params: Parameters for the flocking controller
            action_mapping: Mapping from discrete actions to continuous control inputs
        """
        super(UAVSwarmEnv, self).__init__()
        
        self.num_uavs = num_uavs
        self.num_static_obstacles = num_static_obstacles
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.world_size = world_size
        self.max_steps = max_steps
        
        # Goal position (default to top right corner if not specified)
        self.goal_position = goal_position
        if self.goal_position is None:
            self.goal_position = np.array([0.8 * world_size[0], 0.8 * world_size[1]])
            
        # Set default flocking parameters
        if flocking_params is None:
            self.flocking_params = {
                'c1': 0.5,  # Neighbor position coefficient
                'c2': 0.5,  # Neighbor velocity coefficient
                'c3': 0.8,  # Leader position coefficient
                'c4': 0.5,  # Leader velocity coefficient
                'c5': 1.0,  # Obstacle avoidance coefficient
                'd': 0.1,   # Parameter for network potential
                'perception_radius': 50.0  # Communication radius
            }
        else:
            self.flocking_params = flocking_params
            
        # Set default action mapping for the virtual leader
        if action_mapping is None:
            self.action_mapping = {
                0: np.array([0, 0]),        # No change
                1: np.array([1, 0]),        # Increase x velocity
                2: np.array([-1, 0]),       # Decrease x velocity
                3: np.array([0, 1]),        # Increase y velocity
                4: np.array([0, -1]),       # Decrease y velocity
                5: np.array([0.7, 0.7]),    # Increase both velocities (NE)
                6: np.array([-0.7, 0.7]),   # SW
                7: np.array([-0.7, -0.7]),  # SE
                8: np.array([0.7, -0.7])    # NW
            }
        else:
            self.action_mapping = action_mapping
            
        # Define action and observation spaces
        # Action space: discrete actions for the virtual leader
        self.action_space = spaces.Discrete(len(self.action_mapping))
        
        # Observation space: state vector for the leader RL agent
        # [leader_pos(2), leader_vel(2), leader_to_swarm(2), leader_to_obstacle(2), obs_vel(2), dist_to_obs(1), dist_to_goal(1)]
        state_low = np.array([-np.inf] * 12)
        state_high = np.array([np.inf] * 12)
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        
        # Initialize environment components
        self.reset()
        
    def reset(self):
        """Reset environment to initial state and return observation"""
        # Initialize step counter
        self.steps = 0
        
        # Create UAV swarm
        self.uavs = []
        for i in range(self.num_uavs):
            # Start in a grid formation at the bottom left
            x = 50 + (i % 3) * 30
            y = 50 + (i // 3) * 30
            heading = np.random.uniform(0, 2*np.pi)
            
            self.uavs.append(UAV(
                init_pos=(x, y, 100),
                init_vel=10.0,
                init_heading=heading
            ))
        
        # Create static obstacles
        self.static_obstacles = []
        for _ in range(self.num_static_obstacles):
            # Place obstacles in middle region
            x = np.random.uniform(200, 300)
            y = np.random.uniform(200, 300)
            radius = np.random.uniform(15, 30)
            
            self.static_obstacles.append(StaticObstacle(
                position=(x, y, 100),
                radius=radius
            ))
        
        # Create dynamic obstacles
        self.dynamic_obstacles = []
        for _ in range(self.num_dynamic_obstacles):
            # Place obstacles with random velocities
            x = np.random.uniform(100, 400)
            y = np.random.uniform(100, 400)
            radius = np.random.uniform(10, 20)
            
            # Random velocity vector
            speed = np.random.uniform(1, 5)
            angle = np.random.uniform(0, 2*np.pi)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            
            self.dynamic_obstacles.append(DynamicObstacle(
                position=(x, y, 100),
                radius=radius,
                velocity=(vx, vy, 0)
            ))
        
        # Combine all obstacles
        self.obstacles = self.static_obstacles + self.dynamic_obstacles
        
        # Create flocking controller
        self.flocking_controller = FlockingController(**self.flocking_params)
        
        # Initialize virtual leader (positioned at swarm centroid initially)
        swarm_positions = np.array([uav.position[:2] for uav in self.uavs])
        initial_position = np.mean(swarm_positions, axis=0)
        initial_velocity = np.array([5.0, 5.0])  # Initial velocity towards goal
        
        self.virtual_leader = RLVirtualLeader(
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            agent=None,  # Will be set externally
            action_mapping=self.action_mapping
        )
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def step(self, action):
        """
        Take an action in the environment
        
        Args:
            action: Discrete action index for the virtual leader
        
        Returns:
            observation, reward, done, info
        """
        self.steps += 1
        
        # Convert discrete action to continuous control changes
        control_changes = self.action_mapping[action]
        
        # Update virtual leader based on action
        self.virtual_leader.step(control_changes)
        
        # Update dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            obstacle.update(dt=0.1)
            
            # Boundary handling for obstacles
            if obstacle.position[0] < 0 or obstacle.position[0] > self.world_size[0]:
                obstacle.velocity[0] *= -1
            if obstacle.position[1] < 0 or obstacle.position[1] > self.world_size[1]:
                obstacle.velocity[1] *= -1
        
        # Apply flocking control to each UAV
        for i, uav in enumerate(self.uavs):
            # Get control input from flocking controller
            control_input = self.flocking_controller.compute_control_input(
                uav=uav,
                neighbors=[u for j, u in enumerate(self.uavs) if j != i],
                obstacles=self.obstacles,
                leader_pos=self.virtual_leader.position,
                leader_vel=self.virtual_leader.velocity
            )
            
            # Convert control input to UAV commands
            V_cmd, psi_cmd = self.flocking_controller.convert_to_commands(uav, control_input)[:2]
            uav.set_commands(V_cmd=V_cmd, psi_cmd=psi_cmd)
            
            # Update UAV state
            uav.update()
        
        # Get new observation
        observation = self._get_observation()
        
        # Check for collisions
        collision = self._check_collisions()
        
        # Check if goal reached (any UAV close to goal)
        goal_reached = self._check_goal_reached()
        
        # Check if out of bounds (any UAV outside world boundaries)
        out_of_bounds = self._check_out_of_bounds()
        
        # Calculate reward
        reward = self._calculate_reward(collision, goal_reached)
        
        # Check if episode is done
        done = collision or goal_reached or out_of_bounds or (self.steps >= self.max_steps)
        
        # Additional info
        info = {
            'collision': collision,
            'goal_reached': goal_reached,
            'out_of_bounds': out_of_bounds,
            'steps': self.steps
        }
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """Create observation vector for the agent"""
        # Get leader position and velocity
        leader_pos = self.virtual_leader.position
        leader_vel = self.virtual_leader.velocity
        
        # Get swarm centroid
        swarm_positions = np.array([uav.position[:2] for uav in self.uavs])
        swarm_centroid = np.mean(swarm_positions, axis=0)
        
        # Get closest obstacle info
        closest_obs_dist = float('inf')
        closest_obs_pos = np.zeros(2)
        closest_obs_vel = np.zeros(2)
        
        for obs in self.obstacles:
            obs_pos = obs.position[:2]
            dist = np.linalg.norm(leader_pos - obs_pos) - obs.radius
            
            if dist < closest_obs_dist:
                closest_obs_dist = dist
                closest_obs_pos = obs_pos
                if isinstance(obs, DynamicObstacle):
                    closest_obs_vel = obs.velocity[:2]
        
        # Distance to goal
        dist_to_goal = np.linalg.norm(leader_pos - self.goal_position)
        
        # Combine into observation vector
        obs = np.concatenate([
            leader_pos,                     # Leader position [0, 1]
            leader_vel,                     # Leader velocity [2, 3]
            leader_pos - swarm_centroid,    # Relative position to swarm [4, 5]
            leader_pos - closest_obs_pos,   # Relative position to obstacle [6, 7]
            closest_obs_vel,                # Obstacle velocity [8, 9]
            [closest_obs_dist],             # Distance to obstacle [10]
            [dist_to_goal]                  # Distance to goal [11]
        ])
        
        return obs
    
    def _check_collisions(self):
        """Check if any UAV collides with obstacles"""
        for uav in self.uavs:
            for obstacle in self.obstacles:
                if obstacle.check_collision(uav.position):
                    return True
        return False
    
    def _check_goal_reached(self):
        """Check if the goal has been reached"""
        # Goal is reached when the swarm centroid is close to goal
        swarm_positions = np.array([uav.position[:2] for uav in self.uavs])
        swarm_centroid = np.mean(swarm_positions, axis=0)
        
        dist_to_goal = np.linalg.norm(swarm_centroid - self.goal_position)
        return dist_to_goal < 30.0  # Threshold for goal reaching
    
    def _check_out_of_bounds(self):
        """Check if any UAV is out of bounds"""
        for uav in self.uavs:
            x, y = uav.position[:2]
            if (x < 0 or x > self.world_size[0] or 
                y < 0 or y > self.world_size[1]):
                return True
        return False
    
    def _calculate_reward(self, collision, goal_reached):
        """Calculate reward based on current state"""
        # Base reward components from the paper
        w_col = 1.0  # Weight for collision penalty
        w_tm = 0.5   # Weight for target reaching reward
        
        # Initialize reward
        reward = 0.0
        
        # Collision penalty
        if collision:
            reward += w_col * (-100)
        
        # Goal reaching reward
        if goal_reached:
            reward += w_tm * 100
        
        # Add distance-based reward component
        swarm_positions = np.array([uav.position[:2] for uav in self.uavs])
        swarm_centroid = np.mean(swarm_positions, axis=0)
        dist_to_goal = np.linalg.norm(swarm_centroid - self.goal_position)
        
        # Reward for moving toward goal
        reward += 10.0 * (1.0 - min(1.0, dist_to_goal / 500.0))
        
        # Penalty for being close to obstacles
        min_obstacle_dist = float('inf')
        for uav in self.uavs:
            for obstacle in self.obstacles:
                dist = obstacle.distance_to(uav.position)
                min_obstacle_dist = min(min_obstacle_dist, dist)
        
        if min_obstacle_dist < 30:
            obstacle_penalty = 10 * (1 - min(1.0, min_obstacle_dist / 30.0))
            reward -= obstacle_penalty
            
        return reward
    
    def render(self, mode='human'):
        """Render the environment"""
        if not hasattr(self, 'fig'):
            # Create figure on first call
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.ion()
            
        self.ax.clear()
        
        # Set boundaries
        self.ax.set_xlim(0, self.world_size[0])
        self.ax.set_ylim(0, self.world_size[1])
        
        # Draw UAVs
        for uav in self.uavs:
            # UAV position
            x, y = uav.position[:2]
            
            # Draw UAV as a marker with direction
            self.ax.plot(x, y, 'bo', markersize=8)
            
            # Draw velocity vector
            vx = uav.V * np.cos(uav.psi)
            vy = uav.V * np.sin(uav.psi)
            self.ax.arrow(x, y, vx, vy, head_width=3, head_length=3, fc='blue', ec='blue')
        
        # Draw virtual leader
        x, y = self.virtual_leader.position
        self.ax.plot(x, y, 'ro', markersize=10)
        self.ax.arrow(x, y, self.virtual_leader.velocity[0], self.virtual_leader.velocity[1], 
                     head_width=3, head_length=3, fc='red', ec='red')
        
        # Draw obstacles
        for obstacle in self.static_obstacles:
            x, y = obstacle.position[:2]
            radius = obstacle.radius
            circle = Circle((x, y), radius, fill=True, color='gray', alpha=0.5)
            self.ax.add_patch(circle)
            
        for obstacle in self.dynamic_obstacles:
            x, y = obstacle.position[:2]
            radius = obstacle.radius
            circle = Circle((x, y), radius, fill=True, color='orange', alpha=0.5)
            self.ax.add_patch(circle)
            
            # Draw velocity vector
            vx, vy = obstacle.velocity[:2]
            self.ax.arrow(x, y, vx*3, vy*3, head_width=3, head_length=3, fc='orange', ec='orange')
        
        # Draw goal
        self.ax.plot(self.goal_position[0], self.goal_position[1], 'gx', markersize=15, markeredgewidth=3)
        circle = Circle((self.goal_position[0], self.goal_position[1]), 30, fill=False, color='green', linestyle='--')
        self.ax.add_patch(circle)
        
        # Add legend and title
        self.ax.set_title(f'UAV Swarm Simulation - Step: {self.steps}')
        self.ax.set_xlabel('X position')
        self.ax.set_ylabel('Y position')
        
        plt.draw()
        plt.pause(0.01)
        
    def close(self):
        """Close environment resources"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)