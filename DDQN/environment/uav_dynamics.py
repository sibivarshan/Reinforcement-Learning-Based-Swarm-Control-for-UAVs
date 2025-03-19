import numpy as np

class UAV:
    """
    Implementation of UAV dynamics based on the kinematic model from the paper:
    
    ẋᵢ = Vᵢ cos(ψᵢ)
    ẏᵢ = Vᵢ sin(ψᵢ)
    V̇ᵢ = (Vᵢᶜ - Vᵢ) / τᵥ
    ψ̇ᵢ = (ψᵢᶜ - ψᵢ) / τᵩ
    ḣᵢ = ḣᵢ / τₕ + (hᵢᶜ - hᵢ) / τₕ
    """
    def __init__(self, 
                 init_pos=(0, 0, 100),  # (x, y, h) initial position
                 init_vel=10.0,         # Initial velocity
                 init_heading=0.0,      # Initial heading
                 tau_v=1.0,             # Velocity time constant
                 tau_psi=0.5,           # Heading time constant
                 tau_h=1.0,             # Altitude time constant
                 v_min=5.0,             # Minimum velocity
                 v_max=15.0,            # Maximum velocity
                 n_max=3.0,             # Maximum load factor
                 dt=0.1):               # Simulation time step
        
        # State variables
        self.position = np.array(init_pos)  # [x, y, h]
        self.V = init_vel                   # Speed
        self.psi = init_heading             # Heading angle
        
        # Command variables (control inputs)
        self.V_c = init_vel
        self.psi_c = init_heading
        self.h_c = init_pos[2]
        
        # Time constants
        self.tau_v = tau_v
        self.tau_psi = tau_psi
        self.tau_h = tau_h
        
        # Constraints
        self.v_min = v_min
        self.v_max = v_max
        self.n_max = n_max
        self.g = 9.81  # gravity constant
        
        # Time step for integration
        self.dt = dt
        
        # Position and state history (for visualization)
        self.position_history = [self.position.copy()]
        self.velocity_vector = np.array([self.V * np.cos(self.psi), 
                                         self.V * np.sin(self.psi), 
                                         0])
        self.velocity_history = [self.velocity_vector.copy()]
        
    def update(self):
        """
        Update UAV state for one time step based on current commands
        """
        # Calculate derivatives based on the kinematic model
        x_dot = self.V * np.cos(self.psi)
        y_dot = self.V * np.sin(self.psi)
        V_dot = (self.V_c - self.V) / self.tau_v
        
        # Apply heading rate constraint based on load factor
        max_psi_dot = self.n_max * self.g / max(self.V, 0.1)  # Avoid division by zero
        psi_cmd_delta = (self.psi_c - self.psi + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
        psi_dot_cmd = psi_cmd_delta / self.tau_psi
        psi_dot = np.clip(psi_dot_cmd, -max_psi_dot, max_psi_dot)
        
        # Altitude dynamics
        h_dot = 0  # Simplified: height doesn't change for 2D scenario
        h_dot = h_dot / self.tau_h + (self.h_c - self.position[2]) / self.tau_h
        
        # Update state using Euler integration
        self.position[0] += x_dot * self.dt
        self.position[1] += y_dot * self.dt
        self.position[2] += h_dot * self.dt
        self.V += V_dot * self.dt
        self.psi += psi_dot * self.dt
        
        # Apply velocity constraints
        self.V = np.clip(self.V, self.v_min, self.v_max)
        
        # Normalize heading angle to [0, 2π]
        self.psi = self.psi % (2 * np.pi)
        
        # Update velocity vector
        self.velocity_vector = np.array([self.V * np.cos(self.psi),
                                         self.V * np.sin(self.psi),
                                         h_dot])
        
        # Store history
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity_vector.copy())
        
    def set_commands(self, V_cmd=None, psi_cmd=None, h_cmd=None):
        """
        Set command inputs for the UAV
        
        Args:
            V_cmd: Velocity command (m/s)
            psi_cmd: Heading command (rad)
            h_cmd: Altitude command (m)
        """
        if V_cmd is not None:
            self.V_c = np.clip(V_cmd, self.v_min, self.v_max)
        
        if psi_cmd is not None:
            self.psi_c = psi_cmd % (2 * np.pi)  # Normalize to [0, 2π]
        
        if h_cmd is not None:
            self.h_c = h_cmd

    def get_state(self):
        """
        Get UAV state as a vector
        
        Returns:
            State vector [x, y, h, V, psi]
        """
        return np.array([self.position[0], self.position[1], self.position[2], 
                         self.V, self.psi])