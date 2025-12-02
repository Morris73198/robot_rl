import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_training_history(filename='multi_robot_training_history_ac_ep000002.json'):
    """
    讀取訓練歷史數據
    
    Args:
        filename: 包含訓練歷史的JSON文件
    
    Returns:
        包含訓練數據的字典
    """
    print(f"正在讀取檔案: {filename}")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {filename}")
        return None
    except json.JSONDecodeError:
        print(f"錯誤: 無法解析JSON檔案 {filename}")
        return None

def plot_a2c_training_progress(history, output_dir='./a2c_visualization_results', max_episodes=None):
    """
    繪製Actor-Critic訓練進度圖表
    
    Args:
        history: 包含訓練數據的字典
        output_dir: 輸出目錄
        max_episodes: 最多繪製前多少個episode的數據，None表示全部
    """
    # 檢查歷史數據是否為空
    if not history:
        print("錯誤: 歷史數據為空")
        return
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 檢查數據是否存在
    required_keys = ['episode_rewards', 'robot1_rewards', 'robot2_rewards', 'episode_lengths', 
                     'actor_losses', 'critic_losses', 'exploration_progress']
    for key in required_keys:
        if key not in history:
            print(f"警告: 缺少 {key} 數據")
            return
    
    # 檢查數據長度
    data_length = len(history['episode_rewards'])
    
    # 如果指定了最大episode數，則限制數據長度
    if max_episodes is not None and max_episodes > 0 and max_episodes < data_length:
        data_length = max_episodes
        print(f"僅顯示前 {data_length} 個episode的數據")
    else:
        print(f"顯示全部 {data_length} 個episode的數據")
    
    # 如果數據點太少，可能無法進行有意義的視覺化
    if data_length < 1:
        print("警告: 數據點太少，無法進行有意義的視覺化")
        return
    
    # 截取需要的數據
    episodes = range(1, data_length + 1)
    episode_rewards = history['episode_rewards'][:data_length]
    robot1_rewards = history['robot1_rewards'][:data_length]
    robot2_rewards = history['robot2_rewards'][:data_length]
    episode_lengths = history['episode_lengths'][:data_length]
    actor_losses = history['actor_losses'][:data_length]
    critic_losses = history['critic_losses'][:data_length]
    exploration_progress = history['exploration_progress'][:data_length]
    
    # 設置美學風格
    plt.style.use('seaborn-whitegrid')    
    # 1. 繪製總獎勵圖
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, episode_rewards, '-', color='#2E8B57', linewidth=2)
    plt.title('Total Rewards per Episode', fontsize=14, pad=10)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加最大值和最小值標記
    plt.axhline(y=max(episode_rewards), color='green', linestyle='--', alpha=0.5, 
               label=f'Max: {max(episode_rewards):.2f}')
    plt.axhline(y=min(episode_rewards), color='red', linestyle='--', alpha=0.5,
               label=f'Min: {min(episode_rewards):.2f}')
    plt.legend()
    
    total_rewards_path = os.path.join(output_dir, 'a2c_total_rewards.png')
    plt.savefig(total_rewards_path, dpi=300, bbox_inches='tight')
    print(f"保存總獎勵圖表到: {total_rewards_path}")
    plt.close()
    
    # 2. 繪製機器人獎勵圖
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, robot1_rewards, '-', color='#8A2BE2', linewidth=2, label='Robot 1')
    plt.plot(episodes, robot2_rewards, 's-', color='#FFA500', linewidth=2, label='Robot 2')
    plt.title('Robot Rewards per Episode', fontsize=14, pad=10)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    robots_rewards_path = os.path.join(output_dir, 'a2c_robot_rewards.png')
    plt.savefig(robots_rewards_path, dpi=300, bbox_inches='tight')
    print(f"保存機器人獎勵圖表到: {robots_rewards_path}")
    plt.close()
    
    # 3. 繪製步數圖
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, episode_lengths, '-', color='#4169E1', linewidth=2)
    plt.title('Steps per Episode', fontsize=14, pad=10)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    steps_path = os.path.join(output_dir, 'a2c_episode_lengths.png')
    plt.savefig(steps_path, dpi=300, bbox_inches='tight')
    print(f"保存步數圖表到: {steps_path}")
    plt.close()
    
    # 4. 繪製Actor Loss圖
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, actor_losses, '-', color='#DC143C', linewidth=2)
    plt.title('Actor Loss', fontsize=14, pad=10)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    actor_loss_path = os.path.join(output_dir, 'a2c_actor_loss.png')
    plt.savefig(actor_loss_path, dpi=300, bbox_inches='tight')
    print(f"保存Actor Loss圖表到: {actor_loss_path}")
    plt.close()
    
    # 5. 繪製Critic Loss圖
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, critic_losses, '-', color='#2F4F4F', linewidth=2)
    plt.title('Critic Loss', fontsize=14, pad=10)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    critic_loss_path = os.path.join(output_dir, 'a2c_critic_loss.png')
    plt.savefig(critic_loss_path, dpi=300, bbox_inches='tight')
    print(f"保存Critic Loss圖表到: {critic_loss_path}")
    plt.close()
    
    # 6. 繪製探索進度圖
    plt.figure(figsize=(12, 7))
    exploration = [x * 100 for x in exploration_progress]  # 轉換為百分比
    plt.plot(episodes, exploration, '-', color='#228B22', linewidth=2)
    plt.title('Exploration Progress', fontsize=14, pad=10)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Exploration Rate (%)', fontsize=12)
    plt.ylim([min(exploration) - 1, 101])  # 設置百分比的範圍
    plt.grid(True, linestyle='--', alpha=0.7)
    
    exploration_path = os.path.join(output_dir, 'a2c_exploration_progress.png')
    plt.savefig(exploration_path, dpi=300, bbox_inches='tight')
    print(f"保存探索進度圖表到: {exploration_path}")
    plt.close()
    
    # 7. 機器人獎勵對比圖 (條形圖)
    if data_length <= 10:  # 只在數據點較少時使用條形圖
        plt.figure(figsize=(12, 7))
        width = 0.35  # 條形寬度
        x = np.arange(len(episodes))  # x軸位置
        
        robot1_bar = plt.bar(x - width/2, robot1_rewards, width, label='Robot 1', color='#8A2BE2', alpha=0.7)
        robot2_bar = plt.bar(x + width/2, robot2_rewards, width, label='Robot 2', color='#FFA500', alpha=0.7)
        
        # 添加數據標籤
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3點垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9)
        
        add_labels(robot1_bar)
        add_labels(robot2_bar)
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Robot Rewards Comparison')
        plt.xticks(x, episodes)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        robots_bar_path = os.path.join(output_dir, 'a2c_robots_rewards_bar.png')
        plt.savefig(robots_bar_path, dpi=300, bbox_inches='tight')
        print(f"保存機器人獎勵條形圖到: {robots_bar_path}")
        plt.close()
    
    # 8. Actor-Critic 損失對比圖 (雙Y軸)
    plt.figure(figsize=(12, 7))
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Actor Loss', color=color)
    ax1.plot(episodes, actor_losses, '-', color=color, linewidth=2, label='Actor Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # 創建第二個y軸
    
    color = 'tab:blue'
    ax2.set_ylabel('Critic Loss', color=color)
    ax2.plot(episodes, critic_losses, 's-', color=color, linewidth=2, label='Critic Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Actor-Critic Loss Comparison')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 添加兩個軸的圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    fig.tight_layout()
    loss_comparison_path = os.path.join(output_dir, 'a2c_loss_comparison.png')
    plt.savefig(loss_comparison_path, dpi=300, bbox_inches='tight')
    print(f"保存Actor-Critic損失對比圖到: {loss_comparison_path}")
    plt.close()
    
    # 9. 綜合總覽圖（適合用於演示和概述）- 單列布局
    fig, axs = plt.subplots(6, 1, figsize=(12, 24))
    
    # 調整子圖間距
    plt.subplots_adjust(hspace=0.4)
    
    # 總獎勵圖
    axs[0].plot(episodes, episode_rewards, '-', color='#2E8B57', linewidth=2)
    axs[0].set_title('Total Rewards', fontsize=14)
    axs[0].set_xlabel('Episode', fontsize=12)
    axs[0].set_ylabel('Reward', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # 機器人獎勵圖
    axs[1].plot(episodes, robot1_rewards, '-', color='#8A2BE2', linewidth=2, label='Robot 1')
    axs[1].plot(episodes, robot2_rewards, '-', color='#FFA500', linewidth=2, label='Robot 2')
    axs[1].set_title('Robot Rewards', fontsize=14)
    axs[1].set_xlabel('Episode', fontsize=12)
    axs[1].set_ylabel('Reward', fontsize=12)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # 步數圖
    axs[2].plot(episodes, episode_lengths, '-', color='#4169E1', linewidth=2)
    axs[2].set_title('Steps per Episode', fontsize=14)
    axs[2].set_xlabel('Episode', fontsize=12)
    axs[2].set_ylabel('Steps', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    # Actor Loss圖
    axs[3].plot(episodes, actor_losses, '-', color='#DC143C', linewidth=2)
    axs[3].set_title('Actor Loss', fontsize=14)
    axs[3].set_xlabel('Episode', fontsize=12)
    axs[3].set_ylabel('Loss', fontsize=12)
    axs[3].grid(True, linestyle='--', alpha=0.7)
    
    # Critic Loss圖
    axs[4].plot(episodes, critic_losses, '-', color='#2F4F4F', linewidth=2)
    axs[4].set_title('Critic Loss', fontsize=14)
    axs[4].set_xlabel('Episode', fontsize=12)
    axs[4].set_ylabel('Loss', fontsize=12)
    axs[4].grid(True, linestyle='--', alpha=0.7)
    
    # 探索進度圖
    axs[5].plot(episodes, exploration, '-', color='#228B22', linewidth=2)
    axs[5].set_title('Exploration Progress', fontsize=14)
    axs[5].set_xlabel('Episode', fontsize=12)
    axs[5].set_ylabel('Exploration Rate (%)', fontsize=12)
    axs[5].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    overview_path = os.path.join(output_dir, 'a2c_training_overview.png')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    print(f"保存訓練總覽圖到: {overview_path}")
    plt.close()
    
    print(f"完成所有圖表繪製，結果保存在目錄: {output_dir}")

def main():
    # 硬編碼檔名，無需命令行參數
    filename = 'multi_robot_training_history_ac_ep009640.json'
    
    # 設置每個繪圖顯示的最大episode數，None表示顯示全部
    max_episodes = None  # 可以修改此值以限制顯示前N個episode
    
    # 加載歷史數據
    history = load_training_history(filename)
    
    # 繪製并保存圖表
    if history:
        plot_a2c_training_progress(history, max_episodes=max_episodes)
    else:
        print("無法繪製圖表，請檢查檔案是否存在且格式正確")

if __name__ == "__main__":
    main()