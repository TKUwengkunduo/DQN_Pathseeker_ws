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
import threading
import queue

# 停用互動式日誌記錄時，Keras 會將日誌傳送到absl.logging。當以非互動方式使用 Keras（例如在伺服器上執行訓練或推理作業）時，這是最佳選擇。
# https://stackoverflow.com/questions/73189476/disable-command-prompt-to-show-every-training-step
tf.keras.utils.disable_interactive_logging()

# 全局标志，控制线程何时停止
training_active = True


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

    # def _build_model(self):
    #     model = Sequential()
    #     model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    #     model.add(Dense(24, activation='relu'))
    #     model.add(Dense(self.action_size, activation='linear'))
    #     model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    #     return model


    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))  # 添加額外的隱藏層
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.0005))  # 調整學習率
        return model


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


def environment_interaction(env, data_queue, num_episodes, max_steps):
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0  # 初始化奖励为0
        for _ in range(max_steps):
            if not training_active:
                return  # 终止线程
            action = random.randrange(env.action_space.n)  # 或使用策略选择动作
            next_state, reward, done, _ = env.step(action)
            total_reward += reward  # 更新总奖励
            data_queue.put((state, action, reward, next_state, done, total_reward))

            state = next_state
            if done:
                break



def model_training(agent, data_queue, batch_size):
    while True:
        if not training_active:
            return  # 终止线程
        if not data_queue.empty():
            minibatch = [data_queue.get() for _ in range(batch_size)]
            for state, action, reward, next_state, done, _ in minibatch:
                target = reward
                if not done:
                    target = reward + agent.gamma * np.amax(agent.model.predict(np.array([next_state]))[0])
                target_f = agent.model.predict(np.array([state]))
                target_f[0][action] = target
                agent.model.fit(np.array([state]), target_f, epochs=1, verbose=0)


def save_model(agent, episode, model_dir='models'):
    """
    保存模型到指定目录
    """
    print('Model saving')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'model.keras')
    agent.model.save(model_path)
    print(f'Model saved to {model_path}')


if __name__ == "__main__":
    # 选择使用GPU
    tf.config.list_physical_devices('GPU')
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print(tf.test.is_gpu_available())

    env = GridEnvironment(size=5)
    state_size = env.state_size
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    max_steps = env.get_max_steps()
    episodes = 1000
    print("max_steps = ", max_steps)
    batch_size = max_steps
    scores = []

    data_queue = queue.Queue(maxsize=100000)

    # 创建和启动环境交互线程
    num_interaction_threads = 4
    interaction_threads = [threading.Thread(target=environment_interaction, args=(GridEnvironment(5), data_queue, episodes, max_steps)) for _ in range(num_interaction_threads)]
    for t in interaction_threads:
        t.start()

    # 创建和启动模型训练线程
    num_training_threads = 4
    training_threads = [threading.Thread(target=model_training, args=(agent, data_queue, batch_size)) for _ in range(num_training_threads)]
    for t in training_threads:
        t.start()

    # 主训练循环
    for e in range(episodes):
        total_reward = 0
        for _ in tqdm(range(max_steps), desc=f"Episode {e+1}/{episodes}"):
            # 从 data_queue 获取数据
            if not data_queue.empty():
                state, action, reward, next_state, done, episode_reward = data_queue.get()
                agent.remember(state, action, reward, next_state, done)
                total_reward = episode_reward  # 更新总奖励

                if done:
                    break

        scores.append(total_reward)  # 更新得分
        print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # 每 10 个情节绘制一次得分图
        if e % 10 == 0 and e != 0:
            plt.plot(scores, color='deepskyblue')
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward')
            plt.title('Agent Training Performance')
            plt.legend()
            plt.show(block=False)
            plt.pause(0.1)

    training_active = False

    # 等待所有线程完成
    for t in interaction_threads + training_threads:
        t.join()

    save_model(agent, 'DQN')

"""
强化学习中的奖励（reward）取值和趋势通常依赖于特定的任务和环境。在DQN（Deep Q-Network）或其他强化学习方法中，并没有一个“正确”的奖励值，但是有一些指标可以帮助判断学习过程是否成功：

与目标的一致性：理想的奖励值应该反映代理（agent）达成任务目标的程度。例如，在一个导航任务中，如果代理成功到达目的地，应该获得正奖励；如果它撞到障碍物或者用时过长，则可能获得负奖励。

趋势和稳定性：在训练过程中，总体奖励值应该呈现出逐渐提升的趋势，显示代理正在学习如何更好地完成任务。随着训练的进行，奖励值应该越来越稳定，波动减少。

与环境的关联：奖励值应该与环境中的状态和动作直接相关，确保代理可以从其动作中学习到正确的反馈。

例如，如果您的任务是让代理在迷宫中找到出口，一个不断增加的平均奖励可能表示代理正在学习更有效的路径。最终，如果代理能够一致地快速且有效地找到出口，可能会达到一个稳定且较高的平均奖励值。

最重要的是要注意奖励值的趋势和代理行为的改进，而不是单一的数值。每个强化学习问题都是独特的，奖励的最佳值和趋势取决于特定任务的目标和环境的复杂性。
"""