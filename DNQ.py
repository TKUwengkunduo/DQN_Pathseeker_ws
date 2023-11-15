import os
import numpy as np
import random
import gym
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

# 停用互動式日誌記錄時，Keras 會將日誌傳送到absl.logging。當以非互動方式使用 Keras（例如在伺服器上執行訓練或推理作業）時，這是最佳選擇。
# https://stackoverflow.com/questions/73189476/disable-command-prompt-to-show-every-training-step
tf.keras.utils.disable_interactive_logging()

class GridEnvironment(gym.Env):
    """
    Custom Grid Environment for path planning.
    """
    def __init__(self, size=5):
        super(GridEnvironment, self).__init__()
        self.size = size
        self.state_size = 2
        self.action_space = gym.spaces.Discrete(4)  # Up, down, left, right
        self.goal_position = (size - 1, size - 1)

    def reset(self):
        self.position = (0, 0)
        return self.position

    def step(self, action):
        x, y = self.position
        if action == 0:   # Up
            x = max(0, x - 1)
        elif action == 1: # Down
            x = min(self.size - 1, x + 1)
        elif action == 2: # Left
            y = max(0, y - 1)
        elif action == 3: # Right
            y = min(self.size - 1, y + 1)

        self.position = (x, y)

        done = self.position == self.goal_position
        reward = 1 if done else -0.1  # Reward for reaching the goal, small negative reward otherwise

        return self.position, reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((self.size, self.size))
        x, y = self.position
        gx, gy = self.goal_position
        grid[gx, gy] = 0.5  # Goal position
        grid[x, y] = 1  # Agent position

        if mode == 'human':
            plt.imshow(grid, cmap='gray')
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()
    
    def get_max_steps(self):
        # 例如，设定 max_steps 为网格面积的两倍
        return self.size * self.size * 2

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    # def _build_model(self):
    #     model = Sequential()
    #     model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    #     model.add(Dense(24, activation='relu'))
    #     model.add(Dense(24, activation='relu'))  # 添加額外的隱藏層
    #     model.add(Dense(self.action_size, activation='linear'))
    #     model.compile(loss='mse', optimizer=Adam(lr=0.0005))  # 調整學習率
    #     return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main loop
if __name__ == "__main__":

    """
    選擇使用GPU
    """
    # amd64
    tf.config.list_physical_devices('GPU')
    # jetson
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print(tf.test.is_gpu_available())

    # 動態分配顯卡記憶體
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    env = GridEnvironment(size=5)
    state_size = env.state_size
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    max_steps = env.get_max_steps()     # 獲取步長
    episodes = 1000
    print("max_steps = ", max_steps)
    batch_size = max_steps
    scores = []  # 初始化得分列表

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [state_size])
        total_reward = 0  # 初始化总奖励为 0
        

        for time in tqdm(range(max_steps), desc=f"Episode {e+1}/{episodes}"):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            env.render()  # 绘制代理位置

            next_state = np.reshape(next_state, [state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        scores.append(total_reward)

        # 每 10 个情节绘制一次得分图
        if e % 10 == 0 and e != 0:
            plt.plot(scores, label='Scores')
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward')
            plt.title('Agent Training Performance')
            plt.legend()
            plt.show(block=False)
            plt.pause(2)
