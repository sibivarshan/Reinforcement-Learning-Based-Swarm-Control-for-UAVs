import numpy as np

class ArtificialPotentialField:
    """
    Implementation of Artificial Potential Field for obstacle avoidance
    """
    def __init__(self, eta=1.0, rho0=10.0):
        """
        Initialize APF
        
        Args:
            eta: Scaling parameter for repulsive potential
            rho0: Influence radius for repulsive potential
        """
        self.eta = eta
        self.rho0 = rho0
        
    def repulsive_potential(self, position, obstacle):
        """
        Calculate repulsive potential for an obstacle
        
        Args:
            position: Current position
            obstacle: Obstacle instance
        
        Returns:
            Repulsive potential value
        """
        # Get obstacle position (filtered for dynamic obstacles)
        obs_position = obstacle.get_filtered_position()
        
        # Calculate distance to obstacle
        rho = np.linalg.norm(position[:2] - obs_position[:2]) - obstacle.radius
        
        # Apply repulsive potential
        if rho <= self.rho0:
            # U_rep(q) = 1/2 * η * (1/ρ - 1/ρ_0)^2 if ρ ≤ ρ_0
            if rho < 1e-6:  # Avoid division by zero
                rho = 1e-6
            return 0.5 * self.eta * ((1.0/rho) - (1.0/self.rho0))**2
        else:
            return 0.0
    
    def repulsive_force(self, position, obstacle):
        """
        Calculate repulsive force for an obstacle
        
        Args:
            position: Current position
            obstacle: Obstacle instance
        
        Returns:
            Repulsive force vector
        """
        # Get obstacle position (filtered for dynamic obstacles)
        obs_position = obstacle.get_filtered_position()
        
        # Calculate distance to obstacle
        distance = np.linalg.norm(position[:2] - obs_position[:2]) - obstacle.radius
        
        # Apply repulsive force
        if distance <= self.rho0:
            # f_rep = -grad(U_rep)
            if distance < 1e-6:  # Avoid division by zero
                distance = 1e-6
                
            direction = (position[:2] - obs_position[:2]) / (distance + obstacle.radius)
            
            # grad(U_rep) = η * (1/ρ - 1/ρ_0) * (1/ρ^2) * (q - q_obs)/||q - q_obs||
            force_magnitude = self.eta * ((1.0/distance) - (1.0/self.rho0)) * (1.0/distance**2)
            return force_magnitude * direction
        else:
            return np.zeros(2)
    
    def total_repulsive_force(self, position, obstacles):
        """
        Calculate total repulsive force from all obstacles
        
        Args:
            position: Current position
            obstacles: List of obstacle instances
        
        Returns:
            Total repulsive force vector
        """
        total_force = np.zeros(2)
        
        for obstacle in obstacles:
            total_force += self.repulsive_force(position, obstacle)
            
        return total_force