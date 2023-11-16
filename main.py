import numpy as np
from GridWorldEnv import GridWorldEnv
from dqn import DQN
import os
from tqdm import tqdm
import matplotlib.pyplot as plt 
plt.ion()  # 啟用交互模式


def train_dqn(env, dqn_agent, episodes=1000, save_interval=20):
    total_rewards = []

    # 創建用於繪製獎勵的圖表
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim([0, episodes])
    ax.set_ylim([-200, 200])
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward per Episode')


    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.size, env.size, 1])
        total_reward = 0
        max_steps = 100  # maximum steps per episode

        for step in tqdm(range(max_steps), desc="step"):
            
            action = dqn_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            next_state = np.reshape(next_state, [1, env.size, env.size, 1])
            total_reward += reward
            
            env.render(mode='human', close=False, episode=e, total_reward=total_reward)  # 繪製當前狀態
            # if a: 
            #     a=0
            #     time.sleep(5)
            #     env.render(mode='human', close=False, episode=e, total_reward=total_reward)
            #     time.sleep(1)
                
            # dqn_agent.remember(state, action, reward, next_state, done)

            # Compute TD error for the current experience (needed for PER)
            target = reward
            if not done:
                target += dqn_agent.gamma * np.amax(dqn_agent.model.predict(next_state)[0])
            current = dqn_agent.model.predict(state)[0][action]
            td_error = abs(target - current)

            # Store experience with TD error in memory
            dqn_agent.remember(state, action, reward, next_state, done, td_error)

            if done: break

            state = next_state

        total_rewards.append(total_reward)
        # 更新獎勵圖表
        ax.plot(total_rewards, label='Total Reward per Episode' if e == 0 else "", color='pink', linewidth=1)
        if e == 0:  ax.legend()
        plt.pause(0.00001)  # 短暫暫停以更新圖表
        

        # Replay experiences from memory when enough memories are stored
        if len(dqn_agent.memory.memory) > 128:
            dqn_agent.replay(128)

        if e%1 == 0:
            print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}\n")

        if e % save_interval == 0 and e!=0:
            dqn_agent.model.save(f'models/model_{e}.keras')
            print(f"Saved model at episode {e}")
    
    plt.ioff()  # 關閉交互模式
    plt.show()  # 顯示最終圖表


if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    
    env = GridWorldEnv(size=10, max_obstacles_num=10)
    dqn_agent = DQN(env)
    train_dqn(env, dqn_agent)
