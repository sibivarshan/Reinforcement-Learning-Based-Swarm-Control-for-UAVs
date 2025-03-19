import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.animation import FuncAnimation
import os

def plot_trajectory(trajectory_data, world_size, goal_position, save_path=None):
    """
    Plot UAV trajectories from recorded data
    
    Args:
        trajectory_data: Dict containing 'uav_positions', 'leader_positions', 'obstacle_positions'
        world_size: Tuple of (width, height) for world boundaries
        goal_position: Tuple of (x, y) for goal position
        save_path: Path to save figure (if None, figure is displayed)
    """
    uav_positions = trajectory_data['uav_positions']
    leader_positions = trajectory_data['leader_positions']
    obstacle_positions = trajectory_data['obstacle_positions']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Draw world boundaries
    plt.plot([0, world_size[0], world_size[0], 0, 0],
             [0, 0, world_size[1], world_size[1], 0], 'k-')
    
    # Draw goal position
    plt.plot(goal_position[0], goal_position[1], 'g*', markersize=15)
    circle = plt.Circle((goal_position[0], goal_position[1]), 30, 
                        color='g', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    
    # Draw UAV trajectories
    num_uavs = len(uav_positions[0])
    for i in range(num_uavs):
        x = [pos[i][0] for pos in uav_positions]
        y = [pos[i][1] for pos in uav_positions]
        plt.plot(x, y, linewidth=1.5, alpha=0.7)
        
        # Mark start and end positions
        plt.plot(x[0], y[0], 'bo', markersize=6)
        plt.plot(x[-1], y[-1], 'bx', markersize=8)
    
    # Draw virtual leader trajectory
    leader_x = [pos[0] for pos in leader_positions]
    leader_y = [pos[1] for pos in leader_positions]
    plt.plot(leader_x, leader_y, 'r-', linewidth=2)
    plt.plot(leader_x[0], leader_y[0], 'ro', markersize=8)  # Start
    plt.plot(leader_x[-1], leader_y[-1], 'rx', markersize=10)  # End
    
    # Draw obstacles (initial position)
    for obs_pos in obstacle_positions[0]:
        circle = plt.Circle((obs_pos[0], obs_pos[1]), radius=15,
                           color='gray', fill=True, alpha=0.5)
        plt.gca().add_patch(circle)
    
    # Set labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('UAV Swarm Trajectory')
    
    # Set plot limits with some padding
    plt.xlim(-50, world_size[0] + 50)
    plt.ylim(-50, world_size[1] + 50)
    
    # Add legend
    plt.legend(['World Boundary', 'Goal', 'UAV Path', 'Leader Path', 'Obstacle'])
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_animation(trajectory_data, world_size, goal_position, save_path=None, fps=10):
    """
    Create an animation of UAV swarm trajectory
    
    Args:
        trajectory_data: Dict containing 'uav_positions', 'leader_positions', 'obstacle_positions'
        world_size: Tuple of (width, height) for world boundaries
        goal_position: Tuple of (x, y) for goal position
        save_path: Path to save animation (if None, animation is displayed)
        fps: Frames per second for animation
    """
    uav_positions = trajectory_data['uav_positions']
    leader_positions = trajectory_data['leader_positions']
    obstacle_positions = trajectory_data['obstacle_positions']
    
    num_frames = len(uav_positions)
    num_uavs = len(uav_positions[0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def init():
        # Draw world boundaries
        ax.plot([0, world_size[0], world_size[0], 0, 0],
               [0, 0, world_size[1], world_size[1], 0], 'k-')
        
        # Draw goal position
        ax.plot(goal_position[0], goal_position[1], 'g*', markersize=15)
        circle = plt.Circle((goal_position[0], goal_position[1]), 30, 
                           color='g', fill=False, linestyle='--')
        ax.add_patch(circle)
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('UAV Swarm Animation')
        
        # Set plot limits with some padding
        ax.set_xlim(-50, world_size[0] + 50)
        ax.set_ylim(-50, world_size[1] + 50)
        
        return []
    
    def update(frame):
        ax.clear()
        
        # Draw world boundaries
        ax.plot([0, world_size[0], world_size[0], 0, 0],
               [0, 0, world_size[1], world_size[1], 0], 'k-')
        
        # Draw goal position
        ax.plot(goal_position[0], goal_position[1], 'g*', markersize=15)
        circle = plt.Circle((goal_position[0], goal_position[1]), 30, 
                           color='g', fill=False, linestyle='--')
        ax.add_patch(circle)
        
        # Draw UAVs
        for i in range(num_uavs):
            # Draw UAV position
            ax.plot(uav_positions[frame][i][0], uav_positions[frame][i][1], 'bo', markersize=8)
            
            # Draw UAV trajectory
            if frame > 0:
                x = [uav_positions[f][i][0] for f in range(frame+1)]
                y = [uav_positions[f][i][1] for f in range(frame+1)]
                ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.7)
        
        # Draw virtual leader
        ax.plot(leader_positions[frame][0], leader_positions[frame][1], 'ro', markersize=10)
        
        # Draw leader trajectory
        if frame > 0:
            x = [leader_positions[f][0] for f in range(frame+1)]
            y = [leader_positions[f][1] for f in range(frame+1)]
            ax.plot(x, y, 'r-', linewidth=2)
        
        # Draw obstacles
        for obs_pos in obstacle_positions[frame]:
            circle = plt.Circle((obs_pos[0], obs_pos[1]), radius=15,
                              color='gray', fill=True, alpha=0.5)
            ax.add_patch(circle)
        
        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'UAV Swarm Animation - Frame {frame}/{num_frames-1}')
        
        # Set plot limits with some padding
        ax.set_xlim(-50, world_size[0] + 50)
        ax.set_ylim(-50, world_size[1] + 50)
        
        return []
    
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    
    if save_path:
        anim.save(save_path, fps=fps, writer='ffmpeg')
        plt.close()
    else:
        plt.show()

def plot_training_metrics(metrics, save_path=None):
    """
    Plot training metrics
    
    Args:
        metrics: Dict containing training metrics
        save_path: Path to save figure (if None, figure is displayed)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot moving average rewards
    plt.subplot(2, 2, 2)
    plt.plot(metrics['avg_rewards'])
    plt.title('Moving Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    # Plot collision rate
    plt.subplot(2, 2, 3)
    plt.plot(metrics['collision_counts'])
    plt.title('Collision Rate')
    plt.xlabel('Episode')
    plt.ylabel('Collision (1=yes, 0=no)')
    
    # Plot goal reached rate
    plt.subplot(2, 2, 4)
    plt.plot(metrics['goal_reached_counts'])
    plt.title('Goal Reaching Rate')
    plt.xlabel('Episode')
    plt.ylabel('Goal Reached (1=yes, 0=no)')
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()