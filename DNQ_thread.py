import os
import numpy as np
import random
import gym
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
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
    """==============================
    " 初始化環境
    =============================="""
    def __init__(self, size=5, obstacle_range=(1, 5)):
        super(GridEnvironment, self).__init__()
        self.size = size
        self.state_size = 2
        self.action_space = gym.spaces.Discrete(4)  # Up, down, left, right
        self.goal_position = (size - 1, size - 1)
        self.obstacle_range = obstacle_range  # 障碍物数量范围
        self.obstacles = []
        self.explored_positions = set() # 儲存以探索位置

    def reset(self):
        self.position = (0, 0)
        self.obstacle_count = random.randint(*self.obstacle_range)  # 随机选择障碍物数量
        self._place_obstacles()
        return self.position

    def _place_obstacles(self):
        self.obstacles = []
        for _ in range(self.obstacle_count):
            while True:
                obstacle = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if obstacle != self.goal_position and obstacle not in self.obstacles and obstacle != (0, 0):
                    self.obstacles.append(obstacle)
                    break

    def step(self, action, episode_number):
        x, y = self.position
        hit_wall = False  # 墙壁撞击标志
        hit_obstacle = False  # 障碍物撞击标志

        # 计算新位置
        if action == 0:   # Up
            x_new = max(0, x - 1)
            y_new = y
        elif action == 1: # Down
            x_new = min(self.size - 1, x + 1)
            y_new = y
        elif action == 2: # Left
            x_new = x
            y_new = max(0, y - 1)
        elif action == 3: # Right
            x_new = x
            y_new = min(self.size - 1, y + 1)

        # 检查是否撞墙
        hit_wall = (x_new, y_new) == (x, y)

        # 检查是否碰到障碍物
        hit_obstacle = (x_new, y_new) in self.obstacles

        # 檢查是否為新位置
        is_new_position = self.position not in self.explored_positions
        if is_new_position:
            self.explored_positions.add(self.position)

        # 更新位置
        if not hit_wall and not hit_obstacle:
            self.position = (x_new, y_new)

        # 计算最大可能距离
        max_distance = np.sqrt((self.size - 1)**2 + (self.size - 1)**2)

        # 计算当前距离并正规化
        current_distance = np.sqrt((self.goal_position[0] - x_new)**2 + (self.goal_position[1] - y_new)**2)
        nor_distance_neg = (current_distance / max_distance) / -1
        nor_distance_pos = 1/(current_distance / max_distance)

        # 根据情节编号调整奖励策略
        if episode_number < 400:
            reward = self.calculate_reward_v1(hit_wall, hit_obstacle, nor_distance_pos, nor_distance_neg, is_new_position)
        elif episode_number < 600:
            reward = self.calculate_reward_v2(hit_wall, hit_obstacle, nor_distance_pos, nor_distance_neg, is_new_position)
        else:
            reward = self.calculate_reward_v3(hit_wall, hit_obstacle, nor_distance_pos, nor_distance_neg, is_new_position)

        done = self.position == self.goal_position or hit_wall or hit_obstacle
        return self.position, reward, done, {}


    def calculate_reward_v1(self, hit_wall, hit_obstacle, nor_distance_pos, nor_distance_neg, is_new_position):
        exploration_reward = 10  # 探索新位置的獎勵
        if self.position == self.goal_position:
            return 100.0  # 成功獎勵
        elif hit_wall or hit_obstacle:
            return -100.0  # 碰撞懲罰
        else:
            reward = 0
            if nor_distance_pos > 0.5:      reward = nor_distance_neg*10
            elif nor_distance_neg == 0:     reward = 100
            else:                           reward = nor_distance_pos*10

            if is_new_position:             reward += exploration_reward    # 如果是新位置，增加探索獎勵
            
            return reward

    def calculate_reward_v2(self, hit_wall, hit_obstacle, nor_distance_pos, nor_distance_neg, is_new_position):
        exploration_reward = 5  # 探索新位置的獎勵
        if self.position == self.goal_position:
            return 100.0  # 成功獎勵
        elif hit_wall or hit_obstacle:
            return -100.0  # 碰撞懲罰
        else:
            reward = 0
            if nor_distance_pos > 0.5:      reward = nor_distance_neg*10
            elif nor_distance_neg == 0:     reward = 100
            else:                           reward = nor_distance_pos*10

            if is_new_position:             reward += exploration_reward    # 如果是新位置，增加探索獎勵
            
            return reward
    
    def calculate_reward_v3(self, hit_wall, hit_obstacle, nor_distance_pos, nor_distance_neg, is_new_position):
        exploration_reward = 0  # 探索新位置的獎勵
        if self.position == self.goal_position:
            return 100.0  # 成功獎勵
        elif hit_wall or hit_obstacle:
            return -100.0  # 碰撞懲罰
        else:
            reward = 0
            if nor_distance_pos > 0.5:      reward = nor_distance_neg*10
            elif nor_distance_neg == 0:     reward = 100
            else:                           reward = nor_distance_pos*10

            if is_new_position:             reward += exploration_reward    # 如果是新位置，增加探索獎勵
            
            return reward


    def render(self, mode='human'):
        grid = np.zeros((self.size, self.size))
        x, y = self.position
        gx, gy = self.goal_position
        grid[gx, gy] = 0.5  # Goal position
        grid[x, y] = 1  # Agent position
        for obs in self.obstacles:
            grid[obs] = 0.5  # 障碍物的表示

        if mode == 'human':
            plt.imshow(grid, cmap='gray')
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()
        return self

    def get_max_steps(self):
        return self.size * self.size * 2
    
    def get_grid_state(self):
        # 機器人位置(2), 終點(1), 障礙物(-1)

        grid = np.zeros((self.size, self.size)) # 创建一个全零的二维数组
        grid[self.position] = 2             # 标记智能体位置
        grid[self.goal_position] = 1        # 标记目标位置 # 用1表示目标位置

        # 标记障碍物位置 # 用-1表示障碍物
        for obs in self.obstacles:
            grid[obs] = -1  

        return grid


class DQNAgent:
    def __init__(self, state_size, action_size, size, max_memory_size=2000, history_length=10):
        self.state_size = state_size
        self.action_size = action_size
        self.size = size  # 添加 size 属性
        self.memory = deque(maxlen=max_memory_size)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.8  # exploration rate (1.0)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.learning_rate_decay = 0.999
        self.learning_rate_min = 0.0001
        self.model = self._build_model()
        self.history = deque(maxlen=history_length)  # 存储最近的状态转换
        self.episode_count = 0  # 新增情节计数器
    
    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(self.size, self.size, 1)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_parameters(self):
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # 更新学习率
        self.learning_rate = max(self.learning_rate_min, self.learning_rate * self.learning_rate_decay)
        # 更新模型的学习率
        self.model.optimizer.learning_rate = self.learning_rate


    def remember(self, state, action, reward, next_state, done):
        if self.is_valid_transition(state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))
    
    def is_valid_transition(self, state, action, reward, next_state, done):
        # 确保状态确实因为行动而发生了变化
        if state == next_state:
            return False
        # 避免循环行为
        elif self.is_cyclic_transition(state, next_state):
            return False
        else:
            return True

    def is_cyclic_transition(self, state, next_state):
        # 检查当前转换是否与历史中的转换重复
        current_transition = (state, next_state)
        if current_transition in self.history:
            return True  # 如果当前转换已在历史中，认为是循环
        self.history.append(current_transition)  # 更新历史
        return False

    def act(self, state):
        # 将状态转换为适合 CNN 输入的格式
        grid_state = state.get_grid_state()[np.newaxis, :, :, np.newaxis]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(grid_state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_grid = state.get_grid_state()[np.newaxis, :, :, np.newaxis]
            next_state_grid = next_state.get_grid_state()[np.newaxis, :, :, np.newaxis]

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state_grid)[0])
            target_f = self.model.predict(state_grid)
            target_f[0][action] = target

            self.model.fit(state_grid, target_f, epochs=1, verbose=0)


def environment_interaction(env, data_queue, num_episodes, max_steps):
    for episode_number in range(num_episodes):
        state = env.reset()
        total_reward = 0  # 初始化奖励为0
        for _ in range(max_steps):
            if not training_active:
                return  # 终止线程
            action = random.randrange(env.action_space.n)  # 或使用策略选择动作
            next_state, reward, done, _ = env.step(action, episode_number)
            total_reward += reward  # 更新总奖励
            data_queue.put((state, action, reward, next_state, done, total_reward))

            state = next_state
            if done:
                break



def model_training(agent, data_queue, batch_size):
    while training_active:
        if not data_queue.empty():
            minibatch = [data_queue.get() for _ in range(min(batch_size, data_queue.qsize()))]
            for state, action, reward, next_state, done, _ in minibatch:
                print("remember")
                agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                print("train")




def save_model(agent, episode, model_dir='models'):
    """
    保存模型到指定目录
    """
    try:
        print('Model saving')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f'model_{episode}.keras')  # 加入情节编号
        agent.model.save(model_path)
        print(f'Model saved to {model_path}')
    except Exception as e:
        print(f"Error saving model: {e}")


if __name__ == "__main__":
    # 选择使用GPU
    tf.config.list_physical_devices('GPU')
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print(tf.test.is_gpu_available())

    env = GridEnvironment(size=5)
    state_size = env.state_size
    action_size = env.action_space.n
    grid_size = env.size  # 获取环境的 size 属性
    agent = DQNAgent(state_size, action_size, grid_size)  # 传递 size 给 DQNAgent

    max_steps = env.get_max_steps()
    episodes = 2000
    print("max_steps = ", max_steps)
    batch_size = 128  # 一般情况下的标准批量大小

    scores = []
    average_rewards = []  # 平均奖励
    exploration_rates = []  # 探索率
    loss_values = []  # 损失值
    success_rates = []  # 成功率
    steps_per_episode = []  # 每情节步骤数

    # 创建新窗口和子图
    plt.figure(figsize=(12, 8))

    data_queue = queue.Queue(maxsize=1000000)

    # 创建和启动环境交互线程
    num_interaction_threads = 4
    interaction_threads = [threading.Thread(target=environment_interaction, args=(GridEnvironment(5), data_queue, episodes, max_steps)) for _ in range(num_interaction_threads)]
    for t in interaction_threads:
        t.start()

    # 创建和启动模型训练线程
    num_training_threads = 2  # 通常情况下，训练线程少于交互线程
    training_threads = [threading.Thread(target=model_training, args=(agent, data_queue, batch_size)) for _ in range(num_training_threads)]
    for t in training_threads:
        t.start()

    # 主训练循环
    for e in range(episodes):
        total_reward = 0
        step_count = 0
        successful_episode = False  # 用于记录情节是否成功

        # agent.update_parameters()  # 更新 DQNAgent 参数

        # 储存模型
        if e % 100 == 0 and e != 0:
            save_model(agent, e)

        # 更新经验回放缓冲区大小
        if e % 200 == 0 and e != 0:
            agent.memory = deque(maxlen=min(10000, agent.memory.maxlen + 1000))

        # for _ in tqdm(range(max_steps), desc=f"Episode {e+1}/{episodes}"):
        #     if not data_queue.empty():
        #         state, action, reward, next_state, done, episode_reward = data_queue.get()
        #         # agent.remember(state, action, reward, next_state, done)
        #         total_reward = episode_reward  # 更新总奖励
        #         step_count += 1

        #         if done:
        #             successful_episode = reward > 0  # 根据奖励判断是否成功
        #             break
        # while not data_queue.empty():
        #     state, action, reward, next_state, done, episode_reward = data_queue.get()
        #     print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}, Episode Reward: {episode_reward}")

        scores.append(total_reward)  # 更新得分
        steps_per_episode.append(step_count)  # 更新步骤数
        average_rewards.append(np.mean(scores[-100:]))  # 更新平均奖励
        exploration_rates.append(agent.epsilon)  # 更新探索率
        success_rates.append(int(successful_episode))  # 更新成功率

        # 绘制指标图表
        if e % 10 == 0 and e != 0:
            plt.subplot(241)
            plt.plot(scores, color='deepskyblue', linewidth=1)
            plt.title("Total Reward")

            plt.subplot(242)
            plt.plot(average_rewards, color='green', linewidth=1)
            plt.title("Average Reward")

            plt.subplot(243)
            plt.plot(exploration_rates, color='red', linewidth=1)
            plt.title("Exploration Rate")

            plt.subplot(244)
            plt.plot(loss_values, color='purple', linewidth=1)
            plt.title("Loss Value")

            plt.subplot(245)
            plt.plot([np.mean(success_rates[-100:])] * len(success_rates), color='orange', linewidth=1)
            plt.title("Success Rate")

            plt.subplot(246)
            plt.plot(steps_per_episode, color='pink', linewidth=1)
            plt.title("Steps per Episode")

            # 可以继续添加其他指标的绘制

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)

    training_active = False

    # 等待所有线程完成
    for t in interaction_threads + training_threads:
        t.join()

    save_model(agent, 'final')

        


