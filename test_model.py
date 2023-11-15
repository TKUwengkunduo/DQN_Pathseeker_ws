import os
import numpy as np
import gym
from tensorflow.keras.models import load_model

from DNQ_thread import GridEnvironment


def test_model(model_path, num_episodes=100):
    env = GridEnvironment(size=5)
    model = load_model(model_path)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = np.argmax(model.predict(np.array([state])))
            next_state, reward, done, _ = env.step(action, episode)
            total_reward += reward
            steps += 1
            state = next_state

            # 可选：如果要显示环境，取消注释下面的行
            env.render()

        print(f"Episode: {episode + 1}, Total reward: {total_reward}, Steps: {steps}")

if __name__ == "__main__":
    model_path = './models/model_400.keras'  # 更改为您的模型路径
    test_model(model_path)
