import numpy as np

class Obstacle:
    """Base class for obstacles"""
    def __init__(self, position, radius):
        self.position = np.array(position)
        self.radius = radius
        
    def check_collision(self, position):
        """Check if there's a collision with the given position"""
        distance = np.linalg.norm(self.position[:2] - position[:2])
        return distance < self.radius
    
    def distance_to(self, position):
        """Calculate distance to the obstacle"""
        return np.linalg.norm(self.position[:2] - position[:2]) - self.radius

class StaticObstacle(Obstacle):
    """Static obstacle implementation"""
    def __init__(self, position, radius):
        super().__init__(position, radius)
        
    def get_position(self):
        return self.position
    
    def get_filtered_position(self, c6=1.0):
        """For compatibility with dynamic obstacles"""
        return self.position

class DynamicObstacle(Obstacle):
    """Dynamic obstacle implementation with velocity"""
    def __init__(self, position, radius, velocity):
        super().__init__(position, radius)
        self.velocity = np.array(velocity)
        self.initial_position = np.array(position)
        self.time = 0
        
    def update(self, dt):
        """Update obstacle position based on velocity"""
        self.position = self.position + self.velocity * dt
        self.time += dt
        return self.position
    
    def get_filtered_position(self, c6=1.0):
        """Get filtered position for obstacle avoidance"""
        # Implementation of equation: ξ_k_filtered = q_k_obs + (1/c_6) * v_o
        return self.position + (1 / c6) * self.velocity