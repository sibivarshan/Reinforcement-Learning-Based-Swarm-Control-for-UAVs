import numpy as np

class FlockingController:
    """
    Implementation of flocking control as described in the paper
    """
    def __init__(self, c1=1.0, c2=1.0, c3=1.0, c4=1.0, c5=1.0, 
                 d=0.5, perception_radius=50.0, obstacle_radius=20.0):
        """
        Initialize flocking controller parameters
        
        Args:
            c1-c5: Control gain parameters
            d: Parameter for potential function
            perception_radius: Maximum distance to consider neighbors
            obstacle_radius: Safety radius around obstacles
        """
        self.c1 = c1  # Velocity consensus gain
        self.c2 = c2  # Position consensus gain
        self.c3 = c3  # Virtual leader position gain
        self.c4 = c4  # Virtual leader velocity gain
        self.c5 = c5  # Obstacle avoidance gain
        self.d = d    # Parameter for potential function
        self.perception_radius = perception_radius
        self.obstacle_radius = obstacle_radius
        
    def net_interaction_force(self, uav, neighbors):
        """
        Calculate network interaction force for flocking
        
        Args:
            uav: Current UAV instance
            neighbors: List of neighboring UAVs
        
        Returns:
            Network interaction force [fx, fy]
        """
        pos_i = uav.position[:2]
        vel_i = uav.velocity_vector[:2]
        
        f_net = np.zeros(2)
        
        # Calculate network interaction force based on neighbors
        for neighbor in neighbors:
            # Skip if outside perception radius
            pos_j = neighbor.position[:2]
            vel_j = neighbor.velocity_vector[:2]
            
            # Calculate distance
            distance = np.linalg.norm(pos_i - pos_j)
            if distance > self.perception_radius or distance < 1e-6:
                continue
                
            # Implement equation for f_i_net
            # f_i_net = c_1 * sum_j(p_i - p_j) - c_2 * sum_j(grad(U_net(||q_i - q_j||)))
            # Where U_net(||x||) = d*||x||^2 + ln(||x||^2)
            
            # Velocity consensus term
            velocity_term = self.c1 * (vel_i - vel_j)
            
            # Gradient of potential function
            # U_net(||x||) = d*||x||^2 + ln(||x||^2)
            # grad(U_net) = 2*d*(q_i - q_j) + 2*(q_i - q_j)/||q_i - q_j||^2
            direction = (pos_i - pos_j) / distance
            grad_potential = 2 * self.d * (pos_i - pos_j) + 2 * direction / distance
            position_term = -self.c2 * grad_potential
            
            f_net += velocity_term + position_term
            
        return f_net
    
    def obstacle_avoidance_force(self, uav, obstacles):
        """
        Calculate obstacle avoidance force using Artificial Potential Field
        
        Args:
            uav: Current UAV instance
            obstacles: List of obstacle instances
        
        Returns:
            Obstacle avoidance force [fx, fy]
        """
        pos_i = uav.position[:2]
        f_obs = np.zeros(2)
        
        # Calculate repulsive force for each obstacle
        for obstacle in obstacles:
            # Use filtered position for dynamic obstacles
            obs_pos = obstacle.get_filtered_position()[:2]
            
            # Calculate distance to obstacle
            distance = np.linalg.norm(pos_i - obs_pos) - obstacle.radius
            
            # Apply repulsive force if within safety radius
            if distance < self.obstacle_radius:
                # Repulsive force based on artificial potential field
                # U_rep(q_i) = 1/2 * η * (1/ρ - 1/ρ_0)^2 if ρ ≤ ρ_0
                # f_obs = -grad(U_rep)
                if distance < 1e-6:  # Avoid division by zero
                    distance = 1e-6
                
                eta = 1.0  # Repulsive gain
                direction = (pos_i - obs_pos) / (distance + obstacle.radius)
                
                # Calculate gradient of potential
                # grad(U_rep) = eta * (1/ρ - 1/ρ_0) * (1/ρ^2) * (q_i - q_k_obs)/||q_i - q_k_obs||
                grad_term = eta * (1/distance - 1/self.obstacle_radius) * (1/distance**2) * direction
                
                # Add repulsive force
                f_obs -= self.c5 * grad_term
                
        return f_obs
    
    def regulation_force(self, uav, leader_pos, leader_vel):
        """
        Calculate regulation force for following virtual leader
        
        Args:
            uav: Current UAV instance
            leader_pos: Virtual leader position [x, y]
            leader_vel: Virtual leader velocity [vx, vy]
        
        Returns:
            Regulation force [fx, fy]
        """
        pos_i = uav.position[:2]
        vel_i = uav.velocity_vector[:2]
        
        # f_i_reg = -c_3 * (p_i - p_L) - c_4 * (q_i - q_L)
        position_term = -self.c3 * (pos_i - leader_pos)
        velocity_term = -self.c4 * (vel_i - leader_vel)
        
        f_reg = position_term + velocity_term
        return f_reg
    
    def compute_control_input(self, uav, neighbors, obstacles, leader_pos, leader_vel):
        """
        Calculate total control input for UAV
        
        Args:
            uav: Current UAV instance
            neighbors: List of neighboring UAVs
            obstacles: List of obstacle instances
            leader_pos: Virtual leader position
            leader_vel: Virtual leader velocity
        
        Returns:
            Control input [ux, uy]
        """
        # Calculate individual forces
        f_net = self.net_interaction_force(uav, neighbors)
        f_obs = self.obstacle_avoidance_force(uav, obstacles)
        f_reg = self.regulation_force(uav, leader_pos, leader_vel)
        
        # Combine forces: u_i = f_i_net + f_i_obs + f_i_reg
        u_i = f_net + f_obs + f_reg
        
        return u_i
    
    def convert_to_commands(self, uav, control_input):
        """
        Convert control force to UAV command inputs
        
        Args:
            uav: UAV object
            control_input: Control force [u_x, u_y, u_z]
            
        Returns:
            Command values for velocity and heading
        """
        # Extract current state from UAV
        current_speed = uav.V
        current_heading = uav.psi
        
        # Calculate desired heading from control vector
        desired_heading = np.arctan2(control_input[1], control_input[0])
        
        # Calculate desired speed adjustment
        control_magnitude = np.linalg.norm(control_input[:2])
        speed_adjustment = np.clip(control_magnitude * 0.1, -2.0, 2.0)  # Limit speed changes
        
        # Calculate desired speed
        desired_speed = current_speed + speed_adjustment
        
        # Limit heading change rate to avoid abrupt turns
        heading_diff = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
        max_heading_change = 0.2  # Maximum heading change per step (radians)
        limited_heading_change = np.clip(heading_diff, -max_heading_change, max_heading_change)
        new_heading = (current_heading + limited_heading_change) % (2 * np.pi)
        
        # For altitude, maintain current altitude (optional: you could add altitude control)
        desired_altitude = uav.position[2]
        
        return desired_speed, new_heading, desired_altitude