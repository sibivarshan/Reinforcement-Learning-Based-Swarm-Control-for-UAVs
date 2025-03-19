# UAV Swarm Control with Reinforcement Learning

This project implements a Reinforcement Learning-based approach for UAV (Unmanned Aerial Vehicle) swarm control in environments with static and dynamic obstacles. The implementation is based on the paper "Reinforcement Learning-Based Swarm Control for UAVs in Static and Dynamic Multi-Obstacle Environments".

## Overview

The system uses a Double Deep Q-Network (DDQN) to train a virtual leader that guides a swarm of UAVs through an environment with obstacles to reach a goal position. The UAVs follow the virtual leader using flocking control rules while avoiding collisions through artificial potential fields.

## Key Components

- **UAV Dynamics**: Implementation of fixed-wing UAV kinematics
- **Flocking Control**: Rules for UAV swarm behavior
- **Artificial Potential Field**: Obstacle avoidance mechanism
- **Double DQN**: Reinforcement learning algorithm for the virtual leader
- **Simulation Environment**: 2D world with static and dynamic obstacles

## Installation

1. Clone this repository
2. Install dependencies: