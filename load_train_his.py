import json
import matplotlib.pyplot as plt
import numpy as np

def load_json_training_history(filename):
    """載入JSON格式的訓練歷史數據"""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def plot_training_progress(training_history, save_path="training_progress.png"):
    """繪製訓練進度圖"""
    fig, axs = plt.subplots(6, 1, figsize=(12, 20))
    
    episodes = range(1, len(training_history['episode_rewards']) + 1)
    
    # 繪製總獎勵
    axs[0].plot(episodes, training_history['episode_rewards'], color='#2E8B57')  # 深綠色表示總體
    axs[0].set_title('total reward')
    axs[0].set_xlabel('episode')
    axs[0].set_ylabel('reward')
    axs[0].grid(True)
    
    # 繪製各機器人獎勵
    axs[1].plot(episodes, training_history['robot1_rewards'], 
                color='#8A2BE2', label='Robot1')  # 紫色
    axs[1].plot(episodes, training_history['robot2_rewards'], 
                color='#FFA500', label='Robot2')  # 橘色
    axs[1].set_title('reward per robot')
    axs[1].set_xlabel('episode')
    axs[1].set_ylabel('reward')
    axs[1].legend()
    axs[1].grid(True)
    
    # 繪製步數
    axs[2].plot(episodes, training_history['episode_lengths'], color='#4169E1')  # 藍色
    axs[2].set_title('step per episode')
    axs[2].set_xlabel('episode')
    axs[2].set_ylabel('step')
    axs[2].grid(True)
    
    # 繪製探索率
    axs[3].plot(episodes, training_history['exploration_rates'], color='#DC143C')  # 深紅色
    axs[3].set_title('epsilon rate')
    axs[3].set_xlabel('episode')
    axs[3].set_ylabel('Epsilon')
    axs[3].grid(True)
    
    # 繪製損失
    axs[4].plot(episodes, training_history['losses'], color='#2F4F4F')  # 深灰色
    axs[4].set_title('training loss')
    axs[4].set_xlabel('episode')
    axs[4].set_ylabel('loss')
    axs[4].grid(True)
    
    # 繪製探索進度
    axs[5].plot(episodes, training_history['exploration_progress'], color='#228B22')  # 森林綠
    axs[5].set_title('exploration progress')
    axs[5].set_xlabel('episode')
    axs[5].set_ylabel('exploration rate')
    axs[5].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # 另外繪製一個單獨的兩機器人獎勵對比圖
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, training_history['robot1_rewards'], 
            color='#8A2BE2', label='Robot1', alpha=0.7)  # 紫色
    plt.plot(episodes, training_history['robot2_rewards'], 
            color='#FFA500', label='Robot2', alpha=0.7)  # 橘色
    plt.fill_between(episodes, training_history['robot1_rewards'], 
                    alpha=0.3, color='#9370DB')  # 淺紫色填充
    plt.fill_between(episodes, training_history['robot2_rewards'], 
                    alpha=0.3, color='#FFB84D')  # 淺橘色填充
    plt.title('機器人獎勵對比')
    plt.xlabel('輪數')
    plt.ylabel('獎勵')
    plt.legend()
    plt.grid(True)
    plt.savefig('robots_rewards_comparison.png')
    plt.close()

def main():
    # 載入JSON數據
    filename = "multi_robot_training_history_ep001780.json"
    training_history = load_json_training_history(filename)
    
    # 繪製訓練進度圖
    plot_training_progress(training_history)
    
    print("圖表已成功生成!")

if __name__ == "__main__":
    main()