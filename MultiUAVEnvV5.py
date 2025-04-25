import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MultiUAVPathPlanningEnv(gym.Env):
    def __init__(self, space_size=(100, 100, 100), num_uavs=2, num_targets=2, num_obstacles=3, num_buildings=5):
        super(MultiUAVPathPlanningEnv, self).__init__()
        
        self.space_size = space_size
        self.num_uavs = num_uavs
        self.num_targets = num_targets
        self.num_obstacles = num_obstacles
        self.num_buildings = num_buildings
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.uav_paths = [[] for _ in range(self.num_uavs)]  # Store UAV paths
        
        # Define action space (move in x, y, and z direction)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_uavs, 3), dtype=np.float32)
        
        # Define observation space: UAV, target, obstacle, and building positions
        obs_dim = num_uavs * 3 + num_targets * 3 + num_obstacles * 3 + num_buildings * 3
        self.observation_space = spaces.Box(low=0, high=max(space_size), shape=(obs_dim,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        # Initialize UAVs, targets, obstacles, and buildings at random positions
        self.uav_positions = [np.random.uniform(0, self.space_size[i % 3]) for i in range(self.num_uavs * 3)]
        self.target_positions = [np.random.uniform(0, self.space_size[i % 3]) for i in range(self.num_targets * 3)]
        self.obstacle_positions = [np.random.uniform(0, self.space_size[i % 3]) for i in range(self.num_obstacles * 3)]
        self.building_positions = [np.random.uniform(0, self.space_size[i % 3]) for i in range(self.num_buildings * 3)]
        
        self.current_episode_reward = 0
        self.uav_paths = [[] for _ in range(self.num_uavs)]  # Reset UAV paths
        
        # Store initial positions
        for i in range(self.num_uavs):
            self.uav_paths[i].append(self.uav_positions[i * 3:i * 3 + 3])
        
        return self._get_observation(), {}
    
    def step(self, actions):
        # Move UAVs based on actions
        for i in range(self.num_uavs):
            new_x = np.clip(self.uav_positions[i * 3] + actions[i][0], 0, self.space_size[0])
            new_y = np.clip(self.uav_positions[i * 3 + 1] + actions[i][1], 0, self.space_size[1])
            new_z = np.clip(self.uav_positions[i * 3 + 2] + actions[i][2], 0, self.space_size[2])
            self.uav_positions[i * 3], self.uav_positions[i * 3 + 1], self.uav_positions[i * 3 + 2] = new_x, new_y, new_z
            
            # Store the new position in the path history
            self.uav_paths[i].append([new_x, new_y, new_z])
        
        # Compute rewards
        reward = 0
        for i in range(self.num_uavs):
            uav_pos = np.array(self.uav_positions[i * 3:i * 3 + 3])
            for j in range(self.num_targets):
                target_pos = np.array(self.target_positions[j * 3:j * 3 + 3])
                if np.linalg.norm(uav_pos - target_pos) < 2.0:
                    reward += 10  # Reward for reaching a target
        
        # Check for collisions with obstacles or buildings
        for i in range(self.num_uavs):
            uav_pos = np.array(self.uav_positions[i * 3:i * 3 + 3])
            for j in range(self.num_obstacles):
                obstacle_pos = np.array(self.obstacle_positions[j * 3:j * 3 + 3])
                if np.linalg.norm(uav_pos - obstacle_pos) < 2.0:
                    reward -= 5  # Penalty for collision
            for j in range(self.num_buildings):
                building_pos = np.array(self.building_positions[j * 3:j * 3 + 3])
                if np.linalg.norm(uav_pos - building_pos) < 5.0:
                    reward -= 10  # Larger penalty for hitting buildings
        
        self.current_episode_reward += reward
        done = all(np.linalg.norm(np.array(self.uav_positions[i * 3:i * 3 + 3]) - np.array(self.target_positions[j * 3:j * 3 + 3])) < 2.0 for i in range(self.num_uavs) for j in range(self.num_targets))
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            print(f"Episode finished. Total Reward: {self.current_episode_reward}")
        
        return self._get_observation(), reward, done, False, {}
    
    def render(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, self.space_size[0])
        ax.set_ylim(0, self.space_size[1])
        ax.set_zlim(0, self.space_size[2])
        
        # Plot UAV paths
        for i in range(self.num_uavs):
            path = np.array(self.uav_paths[i])
            ax.plot(path[:, 0], path[:, 1], path[:, 2], linestyle='dashed', label=f'UAV {i+1} Path')
        
        # Plot UAVs
        uav_positions = np.array(self.uav_positions).reshape(-1, 3)
        ax.scatter(uav_positions[:, 0], uav_positions[:, 1], uav_positions[:, 2], c='blue', marker='o', label='UAVs')
        
        # Plot targets
        target_positions = np.array(self.target_positions).reshape(-1, 3)
        ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], c='red', marker='x', label='Targets')
        
        # Plot obstacles
        obstacle_positions = np.array(self.obstacle_positions).reshape(-1, 3)
        ax.scatter(obstacle_positions[:, 0], obstacle_positions[:, 1], obstacle_positions[:, 2], c='black', marker='s', label='Obstacles')
        
        # Plot buildings as cubes
        for i in range(self.num_buildings):
            x, y, z = self.building_positions[i * 3:i * 3 + 3]
            ax.bar3d(x, y, 0, 5, 5, z, color='gray', alpha=0.5)
        
        ax.legend()
        ax.set_title("Multi-UAV Path Planning Environment with Buildings and Paths")
        plt.show()
    
    def close(self):
        pass
