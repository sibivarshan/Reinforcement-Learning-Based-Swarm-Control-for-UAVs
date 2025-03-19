import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import time

from environment.swarm_env import UAVSwarmEnv
from agents.ddqn_agent import DDQNAgent
from controllers.virtual_leader import RLVirtualLeader

def parse_args():
    parser = argparse.ArgumentParser(description="UAV Swarm Control with DDQN")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "visualize"],
                        help="Mode to run the simulation")
    parser.add_argument("--num_uavs", type=int, default=5,
                        help="Number of UAVs in the swarm")
    parser.add_argument("--static_obstacles", type=int, default=3,
                        help="Number of static obstacles")
    parser.add_argument("--dynamic_obstacles", type=int, default=2,
                        help="Number of dynamic obstacles")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes for training")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during training/testing")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load pretrained model")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
                        
    return parser.parse_args()

def create_environment(args):
    """Create the UAV swarm environment"""
    env = UAVSwarmEnv(
        num_uavs=args.num_uavs,
        num_static_obstacles=args.static_obstacles,
        num_dynamic_obstacles=args.dynamic_obstacles,
        max_steps=args.max_steps
    )
    return env

def create_agent(env):
    """Create DDQN agent"""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DDQNAgent(
        state_size=state_size, 
        action_size=action_size, 
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    return agent

def train(env, agent, args):
    """Train the DDQN agent"""
    print("Starting training...")
    
    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"training_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load pretrained model if specified
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        print(f"Loaded pretrained model from {args.load_model}")
    
    # Assign agent to virtual leader
    env.virtual_leader.agent = agent
    
    # Initialize metrics
    episode_rewards = []
    avg_rewards = []
    collision_counts = []
    goal_reached_counts = []
    
    # Training loop
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        collision = False
        goal_reached = False
        
        # Episode loop
        for step in range(args.max_steps):
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Track metrics
            if info['collision']:
                collision = True
            if info['goal_reached']:
                goal_reached = True
                
            # Render if enabled
            if args.render:
                env.render()
                
            if done:
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        collision_counts.append(1 if collision else 0)
        goal_reached_counts.append(1 if goal_reached else 0)
        
        # Calculate moving average
        window_size = 10
        if len(episode_rewards) >= window_size:
            avg_reward = np.mean(episode_rewards[-window_size:])
            avg_rewards.append(avg_reward)
        else:
            avg_rewards.append(episode_reward)
        
        # Print progress
        print(f"Episode {episode+1}/{args.episodes}, Reward: {episode_reward:.2f}, "
              f"Avg Reward: {avg_rewards[-1]:.2f}, Epsilon: {agent.epsilon:.3f}, "
              f"{'Goal reached' if goal_reached else 'Goal not reached'}, "
              f"{'Collision' if collision else 'No collision'}")
        
        # Save model periodically
        if (episode + 1) % 50 == 0 or (episode + 1) == args.episodes:
            model_path = os.path.join(save_dir, f"ddqn_model_episode_{episode+1}.pth")
            agent.save(model_path)
            print(f"Saved model to {model_path}")
    
    # Plot and save training metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(avg_rewards)
    plt.title('Moving Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.subplot(2, 2, 3)
    plt.plot(collision_counts)
    plt.title('Collision Rate')
    plt.xlabel('Episode')
    plt.ylabel('Collision (1=yes, 0=no)')
    
    plt.subplot(2, 2, 4)
    plt.plot(goal_reached_counts)
    plt.title('Goal Reaching Rate')
    plt.xlabel('Episode')
    plt.ylabel('Goal Reached (1=yes, 0=no)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    
    # Save metrics as numpy arrays
    np.save(os.path.join(save_dir, 'episode_rewards.npy'), np.array(episode_rewards))
    np.save(os.path.join(save_dir, 'avg_rewards.npy'), np.array(avg_rewards))
    np.save(os.path.join(save_dir, 'collision_counts.npy'), np.array(collision_counts))
    np.save(os.path.join(save_dir, 'goal_reached_counts.npy'), np.array(goal_reached_counts))
    
    print(f"Training completed. Results saved to {save_dir}")
    return agent

def test(env, agent, args):
    """Test the trained agent"""
    print("Starting testing...")
    
    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"testing_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        print(f"Loaded model from {args.load_model}")
    else:
        print("No model specified for testing. Using random policy.")
    
    # Assign agent to virtual leader
    env.virtual_leader.agent = agent
    
    # Disable exploration
    agent.epsilon = 0.0
    
    # Testing metrics
    num_episodes = 10
    episode_rewards = []
    collision_counts = []
    goal_reached_counts = []
    steps_to_complete = []
    
    # Testing loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        collision = False
        goal_reached = False
        step_count = 0
        
        # Track trajectory
        trajectory = {
            'uav_positions': [],
            'leader_positions': [],
            'obstacle_positions': []
        }
        
        # Episode loop
        for step in range(args.max_steps):
            # Select action (no exploration)
            action = agent.act(state, eval_mode=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Track metrics
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if info['collision']:
                collision = True
            if info['goal_reached']:
                goal_reached = True
                
            # Store trajectory data
            trajectory['uav_positions'].append([uav.position.copy() for uav in env.uavs])
            trajectory['leader_positions'].append(env.virtual_leader.position.copy())
            trajectory['obstacle_positions'].append(
                [obs.position.copy() for obs in env.obstacles]
            )
                
            # Render if enabled
            if args.render:
                env.render()
                time.sleep(0.05)  # Slow down for better visualization
                
            if done:
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        collision_counts.append(1 if collision else 0)
        goal_reached_counts.append(1 if goal_reached else 0)
        steps_to_complete.append(step_count)
        
        # Save trajectory
        trajectory_path = os.path.join(save_dir, f"trajectory_episode_{episode+1}.npy")
        np.save(trajectory_path, trajectory)
        
        # Print results
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, "
              f"Steps: {step_count}, "
              f"{'Goal reached' if goal_reached else 'Goal not reached'}, "
              f"{'Collision' if collision else 'No collision'}")
    
    # Print summary
    success_rate = sum(goal_reached_counts) / num_episodes
    collision_rate = sum(collision_counts) / num_episodes
    avg_steps = sum(steps_to_complete) / num_episodes if steps_to_complete else 0
    
    print("\nTest Results Summary:")
    print(f"Success Rate: {success_rate*100:.1f}%")
    print(f"Collision Rate: {collision_rate*100:.1f}%")
    print(f"Average Steps to Complete: {avg_steps:.1f}")
    print(f"Average Reward: {sum(episode_rewards)/num_episodes:.2f}")
    
    # Save summary
    summary = {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_steps': avg_steps,
        'avg_reward': sum(episode_rewards)/num_episodes,
        'episode_rewards': episode_rewards,
        'collision_counts': collision_counts,
        'goal_reached_counts': goal_reached_counts,
        'steps_to_complete': steps_to_complete
    }
    
    summary_path = os.path.join(save_dir, "test_summary.npy")
    np.save(summary_path, summary)
    
    # Create visualization of the last episode trajectory
    visualize_trajectory(trajectory, env.world_size, env.goal_position, 
                         os.path.join(save_dir, "trajectory_visualization.png"))
    
    print(f"Testing completed. Results saved to {save_dir}")

def visualize_trajectory(trajectory, world_size, goal_position, save_path):
    """Create visualization of a trajectory"""
    # Extract data
    uav_positions = trajectory['uav_positions']
    leader_positions = trajectory['leader_positions']
    obstacle_positions = trajectory['obstacle_positions']
    
    # Number of steps
    num_steps = len(uav_positions)
    num_uavs = len(uav_positions[0])
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw world boundaries
    plt.plot([0, world_size[0], world_size[0], 0, 0], 
             [0, 0, world_size[1], world_size[1], 0], 'k-')
    
    # Draw goal
    plt.plot(goal_position[0], goal_position[1], 'g*', markersize=15)
    circle = plt.Circle((goal_position[0], goal_position[1]), 30, color='g', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
    
    # Draw UAV trajectories
    for i in range(num_uavs):
        x = [uav_positions[step][i][0] for step in range(num_steps)]
        y = [uav_positions[step][i][1] for step in range(num_steps)]
        plt.plot(x, y, 'b-', linewidth=1, alpha=0.5)
        
        # Mark start position
        plt.plot(x[0], y[0], 'bo', markersize=8)
        
        # Mark end position
        plt.plot(x[-1], y[-1], 'bx', markersize=8)
    
    # Draw leader trajectory
    leader_x = [leader_positions[step][0] for step in range(num_steps)]
    leader_y = [leader_positions[step][1] for step in range(num_steps)]
    plt.plot(leader_x, leader_y, 'r-', linewidth=2)
    plt.plot(leader_x[0], leader_y[0], 'ro', markersize=10)
    plt.plot(leader_x[-1], leader_y[-1], 'rx', markersize=10)
    
    # Draw obstacles (initial and final positions)
    for step_idx in [0, -1]:
        for obs_idx, obs_pos in enumerate(obstacle_positions[step_idx]):
            if step_idx == 0:  # Initial position
                circle = plt.Circle((obs_pos[0], obs_pos[1]), 10, 
                                   color='k', fill=True, alpha=0.3)
            else:  # Final position
                circle = plt.Circle((obs_pos[0], obs_pos[1]), 10,
                                   color='k', fill=True, alpha=0.7)
            plt.gca().add_patch(circle)
    
    # Set plot limits with some padding
    plt.xlim(-50, world_size[0] + 50)
    plt.ylim(-50, world_size[1] + 50)
    
    # Add labels and title
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('UAV Swarm Trajectory')
    
    # Add legend
    plt.plot([], [], 'b-', label='UAV paths')
    plt.plot([], [], 'r-', label='Virtual leader path')
    plt.plot([], [], 'bo', label='Start positions')
    plt.plot([], [], 'bx', label='End positions')
    plt.plot([], [], 'g*', label='Goal')
    plt.plot([], [], 'k-', label='Obstacles')
    plt.legend(loc='upper right')
    
    # Save figure
    plt.savefig(save_path)
    plt.close()

def visualize(env, agent, args):
    """Interactive visualization mode"""
    print("Starting visualization mode...")
    
    # Create directory for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"visualization_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        print(f"Loaded model from {args.load_model}")
    else:
        print("No model specified for visualization. Using random policy.")
    
    # Assign agent to virtual leader
    env.virtual_leader.agent = agent
    
    # Disable exploration
    agent.epsilon = 0.0
    
    # Run a single episode with visualization
    state = env.reset()
    episode_reward = 0
    collision = False
    goal_reached = False
    step_count = 0
    
    # Track trajectory
    trajectory = {
        'uav_positions': [],
        'leader_positions': [],
        'obstacle_positions': []
    }
    
    # Episode loop
    while True:
        # Select action (no exploration)
        action = agent.act(state, eval_mode=True)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Track metrics
        state = next_state
        episode_reward += reward
        step_count += 1
        
        if info['collision']:
            collision = True
        if info['goal_reached']:
            goal_reached = True
            
        # Store trajectory data
        trajectory['uav_positions'].append([uav.position.copy() for uav in env.uavs])
        trajectory['leader_positions'].append(env.virtual_leader.position.copy())
        trajectory['obstacle_positions'].append(
            [obs.position.copy() for obs in env.obstacles]
        )
            
        # Always render in visualization mode
        env.render()
        time.sleep(0.1)  # Slower for better visualization
            
        if done:
            break
    
    # Save trajectory
    trajectory_path = os.path.join(save_dir, "trajectory.npy")
    np.save(trajectory_path, trajectory)
    
    # Print results
    print(f"Episode completed. Reward: {episode_reward:.2f}, Steps: {step_count}")
    print(f"{'Goal reached' if goal_reached else 'Goal not reached'}, "
          f"{'Collision' if collision else 'No collision'}")
    
    # Create visualization of the trajectory
    visualize_trajectory(trajectory, env.world_size, env.goal_position, 
                       os.path.join(save_dir, "trajectory_visualization.png"))
    
    print(f"Visualization completed. Results saved to {save_dir}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create directory for results if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create environment
    env = create_environment(args)
    
    # Create agent
    agent = create_agent(env)
    
    # Run specified mode
    if args.mode == "train":
        train(env, agent, args)
    elif args.mode == "test":
        test(env, agent, args)
    elif args.mode == "visualize":
        visualize(env, agent, args)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()