import numpy as np
from tensorflow.keras.models import load_model
from GridWorldEnv import GridWorldEnv
import os

def test_dqn(model_path, env, episodes=100):
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    model = load_model(model_path)

    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.size, env.size, 1])
        total_reward = 0

        while True:
            action = np.argmax(model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.size, env.size, 1])

            total_reward += reward
            state = next_state
            env.render()  # 繪製當前狀態

            if done:
                break
        
        total_rewards.append(total_reward)
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    env = GridWorldEnv(size=10)
    model_path = 'models/model_160.keras'  # Replace with your model path
    test_dqn(model_path, env)
