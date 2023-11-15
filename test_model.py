import numpy as np
import gym
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

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
        reward = 1 if done else -0.1
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

def test_model(model_path, num_episodes=5, size=5):
    env = GridEnvironment(size)
    model = load_model(model_path)

    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        total_reward = 0

        while True:
            action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])

            total_reward += reward
            state = next_state

            env.render()  # Render the environment

            if done:
                print(f"Episode: {episode+1}, Total reward: {total_reward}")
                break

if __name__ == "__main__":
    model_path = './models/model.keras'  # Update this to your model's path
    test_model(model_path)
