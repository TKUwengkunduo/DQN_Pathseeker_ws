import gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class GridWorldEnv(gym.Env):
    def __init__(self, size=5, max_obstacles_num=10, track_length=4):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.max_obstacles_num = max_obstacles_num
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.observation_space = gym.spaces.Box(low=-1, high=2, shape=(size, size), dtype=int)
        self.track_length = track_length
        self.move_track = deque(maxlen=track_length)  # 儲存最近的幾個位置
        self.reset()

    def reset(self):
        self.move_track.clear()  # 重置移動追踪
        self.agent_pos = self._random_position()
        self.goal_pos = self._random_position(exclude=self.agent_pos)

        # 隨機放置障礙物
        num_obstacles = random.randint(0, self.max_obstacles_num)  # 定義最大障礙物數量
        self.obstacles = []
        while len(self.obstacles) < num_obstacles:
            obstacle_pos = self._random_position(exclude=[self.agent_pos, self.goal_pos] + self.obstacles)
            self.obstacles.append(obstacle_pos)

        self.update_map()  # 初始化地圖

        self.visited = set()
        self.visited.add(tuple(self.agent_pos))
        return self.map.copy()
    
    def update_map(self):
        self.map = np.zeros((self.size, self.size), dtype=int)
        for obs in self.obstacles:
            self.map[obs[0], obs[1]] = -1
        self.map[self.agent_pos[0], self.agent_pos[1]] = 1
        self.map[self.goal_pos[0], self.goal_pos[1]] = 2


    def _random_position(self, exclude=[]):
        while True:
            position = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
            if all(position != e for e in exclude):
                return position

    def step(self, action):
        next_pos = self.agent_pos.copy()
        if action == 0 and self.agent_pos[0] > 0:  # up
            next_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:  # down
            next_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            next_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:  # right
            next_pos[1] += 1

        reward = self._calculate_reward(next_pos)
        
        self.move_track.append(tuple(self.agent_pos))
        # 計算重複來回的附加懲罰
        if len(self.move_track) == self.track_length and len(set(self.move_track)) < self.track_length:
            additional_penalty = -2  # 根據需要調整懲罰數值
            reward += additional_penalty

        # 檢查是否達到終點或碰到障礙物
        hit_obstacle = self.map[next_pos[0], next_pos[1]] == -1
        reached_goal = self.agent_pos == self.goal_pos
        hit_wall = next_pos == self.agent_pos   # 檢查是否撞牆（嘗試移動出地圖邊界）
        done = hit_obstacle or reached_goal or hit_wall

        # 如果未達到終點且未碰到障礙物，則更新機器人位置
        if not done:
            self.agent_pos = next_pos
            self.update_map()  # 更新地圖


        return self.map.copy(), reward, done, {}

    

    def _calculate_reward(self, next_pos):
        reward = 0

        if tuple(next_pos) in self.visited:
            reward -= 0.1  # 重訪懲罰
        else:
            reward += 0.1  # 探索新位置獎勵
            self.visited.add(tuple(next_pos))

        if next_pos == self.agent_pos:
            reward -= 20  # 撞牆懲罰
        elif self.map[next_pos[0], next_pos[1]] == -1:  # 撞障礙物懲罰
            reward -= 20
        elif self.map[next_pos[0], next_pos[1]] == 2:  # 到達目標
            reward += 100
        else:
            # 計算移動前後的距離，根據距離變化計算獎勵或懲罰
            distance_before = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
            distance_after = np.linalg.norm(np.array(next_pos) - np.array(self.goal_pos))
            if distance_after < distance_before:
                reward += 1  # 移動後距離減少，給予獎勵
            else:
                reward -= 1  # 移動後距離增加或不變，給予懲罰

        reward -= 0.1  # 步長懲罰

        return reward


    def render(self, mode='human', close=False, episode=None, total_reward=None):
        if close:
            plt.close()
            return

        if not hasattr(self, 'figure'):
            self.figure, self.ax = plt.subplots(figsize=(5, 5))
            plt.ion()

        self.ax.clear()
        self.ax.imshow(self.map, cmap='hot', interpolation='nearest')
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        self.ax.grid(False)

        # 標記機器人、終點和障礙物
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i, j] == 1:
                    self.ax.text(j, i, 'R', ha='center', va='center', color='blue')
                elif self.map[i, j] == 2:
                    self.ax.text(j, i, 'G', ha='center', va='center', color='green')
                elif self.map[i, j] == -1:
                    self.ax.text(j, i, 'X', ha='center', va='center', color='red')

        # 顯示當前局數和累積獎勵
        if episode is not None and total_reward is not None:
            info_text = f"Episode: {episode}, Total Reward: {total_reward}"
            self.ax.set_title(info_text)

        plt.draw()
        plt.pause(0.00001)
    

    