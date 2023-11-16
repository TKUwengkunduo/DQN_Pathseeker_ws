import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from collections import deque
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

tf.keras.utils.disable_interactive_logging()
"""
    選擇使用GPU
"""
# amd64
tf.config.list_physical_devices('GPU')
# jetson
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print(tf.test.is_gpu_available())

class DQN:
    def __init__(self, env):
        self.env = env
        # self.memory = deque(maxlen=2000)
        self.memory = PrioritizedMemory(2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(self.env.size, self.env.size, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(state.reshape(1, self.env.size, self.env.size, 1))
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done, td_error):
        # self.memory.append((state, action, reward, next_state, done))
        self.memory.add((state, action, reward, next_state, done), td_error)

    def replay(self, batch_size):
        print("Training model...")
        minibatch, indices = self.memory.sample(batch_size)  # 解包為經驗和索引
        new_priorities = []

        for experience in minibatch:
            state, action, reward, next_state, done = experience  # 正確解包每個經驗
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state.reshape(1, self.env.size, self.env.size, 1))[0])
            target_f = self.model.predict(state.reshape(1, self.env.size, self.env.size, 1))[0]
            target_f[action] = target
            self.model.fit(state.reshape(1, self.env.size, self.env.size, 1), target_f.reshape(-1, self.env.action_space.n), epochs=1, verbose=0)

            # 計算新的TD誤差並加入新優先級列表
            new_priorities.append(abs(target - target_f[action]))

        # 更新記憶體中的優先級
        self.memory.update_priorities(indices, new_priorities)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PrioritizedMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, experience, error):
        self.memory.append(experience)
        self.priorities.append(error)

    def sample(self, batch_size):
        probabilities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(range(len(self.memory)), batch_size, p=probabilities)
        return [self.memory[i] for i in indices], indices

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            self.priorities[i] = error
