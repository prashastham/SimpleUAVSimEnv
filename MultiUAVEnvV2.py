import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class MultiUAVPathPlanningEnv(gym.Env):
    def __init__(self, space_size=(100, 100), num_uavs=2, num_targets=2, num_obstacles=3):
        super(MultiUAVPathPlanningEnv, self).__init__()
        
        self.space_size = space_size
        self.num_uavs = num_uavs
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        
        # Define action space (move in x and y direction)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_uavs, 2), dtype=np.float32)
        
        # Define observation space: UAV, target, and obstacle positions
        obs_dim = num_uavs * 2 + num_targets * 2 + num_obstacles * 2
        self.observation_space = spaces.Box(low=0, high=max(space_size), shape=(obs_dim,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        # Initialize UAVs, targets, and obstacles at random positions
        self.uav_positions = [np.random.uniform(0, self.space_size[i % 2]) for i in range(self.num_uavs * 2)]
        self.target_positions = [np.random.uniform(0, self.space_size[i % 2]) for i in range(self.num_targets * 2)]
        self.obstacle_positions = [np.random.uniform(0, self.space_size[i % 2]) for i in range(self.num_obstacles * 2)]
        
        return self._get_observation()
    
    def step(self, actions):
        # Move UAVs based on actions
        for i in range(self.num_uavs):
            new_x = np.clip(self.uav_positions[i * 2] + actions[i][0], 0, self.space_size[0])
            new_y = np.clip(self.uav_positions[i * 2 + 1] + actions[i][1], 0, self.space_size[1])
            self.uav_positions[i * 2], self.uav_positions[i * 2 + 1] = new_x, new_y
        
        # Move obstacles randomly
        for i in range(self.num_obstacles):
            self.obstacle_positions[i * 2] = np.clip(self.obstacle_positions[i * 2] + np.random.uniform(-1, 1), 0, self.space_size[0])
            self.obstacle_positions[i * 2 + 1] = np.clip(self.obstacle_positions[i * 2 + 1] + np.random.uniform(-1, 1), 0, self.space_size[1])
        
        # Compute rewards
        reward = 0
        for i in range(self.num_uavs):
            uav_pos = (self.uav_positions[i * 2], self.uav_positions[i * 2 + 1])
            for j in range(self.num_targets):
                target_pos = (self.target_positions[j * 2], self.target_positions[j * 2 + 1])
                if np.linalg.norm(np.array(uav_pos) - np.array(target_pos)) < 2.0:
                    reward += 10  # Reward for reaching a target
        
        # Check for collisions with obstacles
        for i in range(self.num_uavs):
            uav_pos = (self.uav_positions[i * 2], self.uav_positions[i * 2 + 1])
            for j in range(self.num_obstacles):
                obstacle_pos = (self.obstacle_positions[j * 2], self.obstacle_positions[j * 2 + 1])
                if np.linalg.norm(np.array(uav_pos) - np.array(obstacle_pos)) < 2.0:
                    reward -= 5  # Penalty for collision
        
        done = all(np.linalg.norm(np.array(self.uav_positions[i * 2:i * 2 + 2]) - np.array(self.target_positions[j * 2:j * 2 + 2])) < 2.0 for i in range(self.num_uavs) for j in range(self.num_targets))
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        return np.array(self.uav_positions + self.target_positions + self.obstacle_positions, dtype=np.float32)
    
    def render(self, mode='human'):
        plt.figure(figsize=(6, 6))
        plt.xlim(0, self.space_size[0])
        plt.ylim(0, self.space_size[1])
        
        # Plot UAVs
        for i in range(self.num_uavs):
            plt.scatter(self.uav_positions[i * 2], self.uav_positions[i * 2 + 1], color='blue', label='UAV' if i == 0 else "")
        
        # Plot Targets
        for i in range(self.num_targets):
            plt.scatter(self.target_positions[i * 2], self.target_positions[i * 2 + 1], color='green', marker='*', s=200, label='Target' if i == 0 else "")
        
        # Plot Obstacles
        for i in range(self.num_obstacles):
            plt.scatter(self.obstacle_positions[i * 2], self.obstacle_positions[i * 2 + 1], color='red', marker='x', s=100, label='Obstacle' if i == 0 else "")
        
        plt.legend()
        plt.grid(True)
        plt.title("Multi-UAV Path Planning Environment")
        plt.show()
    
    def close(self):
        pass
