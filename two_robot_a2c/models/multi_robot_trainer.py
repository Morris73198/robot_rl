import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from two_robot_a2c.config import MODEL_DIR, ROBOT_CONFIG

class MultiRobotACTrainer:
    def __init__(self, model, robot1, robot2, gamma=0.99, gae_lambda=0.95):
        self.model = model
        self.robot1 = robot1
        self.robot2 = robot2
        self.gamma = gamma
        self.gae_lambda = gae_lambda  # GAE(Generalized Advantage Estimation)參數
        
        self.map_size = self.robot1.map_size
        
        # 訓練歷史記錄
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'robot1_rewards': [],
            'robot2_rewards': [],
            'exploration_progress': []
        }
        
        # 經驗緩衝區用於儲存當前episode的軌跡
        self.reset_episode_buffer()

    def reset_episode_buffer(self):
        """重置episode緩衝區"""
        self.current_episode = {
            'states': [],
            'frontiers': [],
            'robot1_pos': [],
            'robot2_pos': [],
            'robot1_target': [],
            'robot2_target': [],
            'robot1_actions': [],
            'robot2_actions': [],
            'robot1_rewards': [],
            'robot2_rewards': [],
            'robot1_values': [],
            'robot2_values': [],
            'robot1_dones': [],
            'robot2_dones': []
        }
    
    def pad_frontiers(self, frontiers):
        """填充frontier點到固定長度並進行標準化"""
        padded = np.zeros((self.model.max_frontiers, 2))
        
        if len(frontiers) > 0:
            frontiers = np.array(frontiers)
            
            # 標準化座標
            normalized_frontiers = frontiers.copy()
            normalized_frontiers[:, 0] = frontiers[:, 0] / float(self.map_size[1])
            normalized_frontiers[:, 1] = frontiers[:, 1] / float(self.map_size[0])
            
            n_frontiers = min(len(frontiers), self.model.max_frontiers)
            padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
        
        return padded

    def get_normalized_target(self, target):
        """標準化目標位置"""
        if target is None:
            return np.array([0.0, 0.0])
        normalized = np.array([
            target[0] / float(self.map_size[1]),
            target[1] / float(self.map_size[0])
        ])
        return normalized

    def choose_actions(self, state, frontiers, robot1_pos, robot2_pos,
                      robot1_target, robot2_target):
        """根據當前策略選擇動作"""
        if len(frontiers) == 0:
            return 0, 0

        # 準備輸入數據
        state_batch = np.expand_dims(state, 0)
        frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0)
        robot1_pos_batch = np.expand_dims(robot1_pos, 0)
        robot2_pos_batch = np.expand_dims(robot2_pos, 0)
        robot1_target_batch = np.expand_dims(robot1_target, 0)
        robot2_target_batch = np.expand_dims(robot2_target, 0)

        # 獲取動作概率分布
        policy_dict = self.model.predict_policy(
            state_batch, frontiers_batch,
            robot1_pos_batch, robot2_pos_batch,
            robot1_target_batch, robot2_target_batch
        )

        valid_frontiers = min(self.model.max_frontiers, len(frontiers))
        
        # 從概率分布中採樣動作
        robot1_probs = policy_dict['robot1_policy'][0, :valid_frontiers]
        robot2_probs = policy_dict['robot2_policy'][0, :valid_frontiers]
        
        robot1_action = np.random.choice(valid_frontiers, p=robot1_probs/np.sum(robot1_probs))
        robot2_action = np.random.choice(valid_frontiers, p=robot2_probs/np.sum(robot2_probs))

        return robot1_action, robot2_action

    def compute_advantages(self, rewards, values, dones):
        """計算GAE優勢值"""
        advantages = np.zeros_like(rewards)
        gae = 0
        
        # 反向計算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # 計算回報值
        returns = advantages + values
        
        # 標準化優勢值
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def train_on_episode(self):
        """在收集的episode數據上進行訓練"""
        # 將經驗數據轉換為numpy數組
        states = np.array(self.current_episode['states'])
        frontiers = np.array(self.current_episode['frontiers'])
        robot1_pos = np.array(self.current_episode['robot1_pos'])
        robot2_pos = np.array(self.current_episode['robot2_pos'])
        robot1_target = np.array(self.current_episode['robot1_target'])
        robot2_target = np.array(self.current_episode['robot2_target'])
        
        robot1_values = np.array(self.current_episode['robot1_values'])
        robot2_values = np.array(self.current_episode['robot2_values'])
        robot1_rewards = np.array(self.current_episode['robot1_rewards'])
        robot2_rewards = np.array(self.current_episode['robot2_rewards'])
        robot1_dones = np.array(self.current_episode['robot1_dones'])
        robot2_dones = np.array(self.current_episode['robot2_dones'])
        
        # 計算優勢值和回報值
        robot1_advantages, robot1_returns = self.compute_advantages(
            robot1_rewards, robot1_values, robot1_dones)
        robot2_advantages, robot2_returns = self.compute_advantages(
            robot2_rewards, robot2_values, robot2_dones)
        
        # 訓練Critic
        critic_loss = self.model.train_critic(
            states, frontiers,
            robot1_pos, robot2_pos,
            robot1_target, robot2_target,
            {
                'robot1': robot1_returns,
                'robot2': robot2_returns
            }
        )
        
        # 訓練Actor
        actor_loss = self.model.train_actor(
            states, frontiers,
            robot1_pos, robot2_pos,
            robot1_target, robot2_target,
            {
                'robot1': self.current_episode['robot1_actions'],
                'robot2': self.current_episode['robot2_actions']
            },
            {
                'robot1': robot1_advantages,
                'robot2': robot2_advantages
            }
        )
        
        return actor_loss, critic_loss

    def train(self, episodes=1000000, save_freq=10):
        """執行多機器人協同訓練"""
        try:
            for episode in range(episodes):
                # 初始化環境
                state = self.robot1.begin()
                self.robot2.begin()
                
                # 重置episode緩衝區
                self.reset_episode_buffer()
                
                # 初始化episode統計
                total_reward = 0
                robot1_total_reward = 0
                robot2_total_reward = 0
                steps = 0
                
                while not (self.robot1.check_done() or self.robot2.check_done()):
                    frontiers = self.robot1.get_frontiers()
                    if len(frontiers) == 0:
                        break
                        
                    # 獲取當前狀態
                    robot1_pos = self.robot1.get_normalized_position()
                    robot2_pos = self.robot2.get_normalized_position()
                    
                    robot1_target = self.get_normalized_target(
                        self.robot1.current_target_frontier)
                    robot2_target = self.get_normalized_target(
                        self.robot2.current_target_frontier)
                    
                    # 選擇動作
                    robot1_action, robot2_action = self.choose_actions(
                        state, frontiers, robot1_pos, robot2_pos,
                        robot1_target, robot2_target
                    )
                    
                    # 獲取當前狀態的價值估計
                    values = self.model.predict_value(
                        np.expand_dims(state, 0),
                        np.expand_dims(self.pad_frontiers(frontiers), 0),
                        np.expand_dims(robot1_pos, 0),
                        np.expand_dims(robot2_pos, 0),
                        np.expand_dims(robot1_target, 0),
                        np.expand_dims(robot2_target, 0)
                    )
                    robot1_value = values['robot1_value'][0, 0]
                    robot2_value = values['robot2_value'][0, 0]
                    
                    # 移動機器人
                    robot1_target_pos = frontiers[robot1_action]
                    robot2_target_pos = frontiers[robot2_action]
                    
                    next_state1, r1, d1 = self.robot1.move_to_frontier(robot1_target_pos)
                    self.robot2.op_map = self.robot1.op_map.copy()
                    
                    next_state2, r2, d2 = self.robot2.move_to_frontier(robot2_target_pos)
                    self.robot1.op_map = self.robot2.op_map.copy()
                    
                    # 更新機器人位置
                    self.robot1.other_robot_position = self.robot2.robot_position.copy()
                    self.robot2.other_robot_position = self.robot1.robot_position.copy()
                    
                    # 保存經驗到當前episode緩衝區
                    self.current_episode['states'].append(state)
                    self.current_episode['frontiers'].append(self.pad_frontiers(frontiers))
                    self.current_episode['robot1_pos'].append(robot1_pos)
                    self.current_episode['robot2_pos'].append(robot2_pos)
                    self.current_episode['robot1_target'].append(robot1_target)
                    self.current_episode['robot2_target'].append(robot2_target)
                    self.current_episode['robot1_actions'].append(robot1_action)
                    self.current_episode['robot2_actions'].append(robot2_action)
                    self.current_episode['robot1_rewards'].append(r1)
                    self.current_episode['robot2_rewards'].append(r2)
                    self.current_episode['robot1_values'].append(robot1_value)
                    self.current_episode['robot2_values'].append(robot2_value)
                    self.current_episode['robot1_dones'].append(d1)
                    self.current_episode['robot2_dones'].append(d2)
                    
                    # 更新狀態和累積獎勵
                    state = next_state1
                    total_reward += (r1 + r2)
                    robot1_total_reward += r1
                    robot2_total_reward += r2
                    steps += 1
                    
                    # 更新可視化
                    if steps % ROBOT_CONFIG['plot_interval'] == 0:
                        if self.robot1.plot:
                            self.robot1.plot_env()
                        if self.robot2.plot:
                            self.robot2.plot_env()
                
                # Episode結束，進行訓練
                actor_loss, critic_loss = self.train_on_episode()
                
                # 更新訓練歷史
                exploration_progress = self.robot1.get_exploration_progress()
                self.training_history['episode_rewards'].append(total_reward)
                self.training_history['robot1_rewards'].append(robot1_total_reward)
                self.training_history['robot2_rewards'].append(robot2_total_reward)
                self.training_history['episode_lengths'].append(steps)
                self.training_history['actor_losses'].append(actor_loss)
                self.training_history['critic_losses'].append(critic_loss)
                self.training_history['exploration_progress'].append(exploration_progress)
                
                # 定期保存模型
                if (episode + 1) % save_freq == 0:
                    self.save_checkpoint(episode + 1)
                    self.plot_training_progress()
                
                # 列印訓練信息
                print(f"\nEpisode {episode + 1}/{episodes} (Map {self.robot1.li_map})")
                print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
                print(f"Robot1 Reward: {robot1_total_reward:.2f}")
                print(f"Robot2 Reward: {robot2_total_reward:.2f}")
                print(f"Actor Loss: {actor_loss:.6f}")
                print(f"Critic Loss: {critic_loss:.6f}")
                print(f"Exploration Progress: {exploration_progress:.1%}")
                
                if exploration_progress >= self.robot1.finish_percent:
                    print("Map Exploration Complete!")
                else:
                    print("Map Exploration Incomplete")
                print("-" * 50)
                
                # 重置環境
                state = self.robot1.reset()
                self.robot2.reset()
            
            # 訓練結束後保存最終模型
            self.save_checkpoint(episodes)
            
        except Exception as e:
            print(f"Training Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 確保清理資源
            if hasattr(self.robot1, 'cleanup_visualization'):
                self.robot1.cleanup_visualization()
            if hasattr(self.robot2, 'cleanup_visualization'):
                self.robot2.cleanup_visualization()

    def plot_training_progress(self):
        """繪製訓練進度圖"""
        fig, axs = plt.subplots(7, 1, figsize=(12, 24))
        
        episodes = range(1, len(self.training_history['episode_rewards']) + 1)
        
        # 繪製總獎勵
        axs[0].plot(episodes, self.training_history['episode_rewards'], color='#2E8B57')
        axs[0].set_title('Total Reward')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        axs[0].grid(True)
        
        # 繪製各機器人獎勵
        axs[1].plot(episodes, self.training_history['robot1_rewards'], 
                    color='#8A2BE2', label='Robot1')
        axs[1].plot(episodes, self.training_history['robot2_rewards'], 
                    color='#FFA500', label='Robot2')
        axs[1].set_title('Reward per Robot')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Reward')
        axs[1].legend()
        axs[1].grid(True)
        
        # 繪製步數
        axs[2].plot(episodes, self.training_history['episode_lengths'], color='#4169E1')
        axs[2].set_title('Steps per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Steps')
        axs[2].grid(True)
        
        # 繪製Actor損失
        axs[3].plot(episodes, self.training_history['actor_losses'], color='#DC143C')
        axs[3].set_title('Actor Loss')
        axs[3].set_xlabel('Episode')
        axs[3].set_ylabel('Loss')
        axs[3].grid(True)
        
        # 繪製Critic損失
        axs[4].plot(episodes, self.training_history['critic_losses'], color='#2F4F4F')
        axs[4].set_title('Critic Loss')
        axs[4].set_xlabel('Episode')
        axs[4].set_ylabel('Loss')
        axs[4].grid(True)
        
        # 繪製探索進度
        axs[5].plot(episodes, self.training_history['exploration_progress'], color='#228B22')
        axs[5].set_title('Exploration Progress')
        axs[5].set_xlabel('Episode')
        axs[5].set_ylabel('Completion Rate')
        axs[5].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
        
        # 另外繪製一個單獨的兩機器人獎勵對比圖
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, self.training_history['robot1_rewards'], 
                color='#8A2BE2', label='Robot1', alpha=0.7)
        plt.plot(episodes, self.training_history['robot2_rewards'], 
                color='#FFA500', label='Robot2', alpha=0.7)
        plt.fill_between(episodes, self.training_history['robot1_rewards'], 
                        alpha=0.3, color='#9370DB')
        plt.fill_between(episodes, self.training_history['robot2_rewards'], 
                        alpha=0.3, color='#FFB84D')
        plt.title('Robot Rewards Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig('robots_rewards_comparison.png')
        plt.close()
    
    def save_checkpoint(self, episode):
        """保存檢查點
        
        Args:
            episode: 當前訓練輪數
        """
        # 用零填充確保文件名排序正確
        ep_str = str(episode).zfill(6)
        
        # 保存模型
        model_path = os.path.join(MODEL_DIR, f'multi_robot_model_ac_ep{ep_str}')
        self.model.save(model_path)
        
        # 保存訓練歷史
        history_path = os.path.join(MODEL_DIR, f'multi_robot_training_history_ac_ep{ep_str}.json')
        history_to_save = {
            'episode_rewards': [float(x) for x in self.training_history['episode_rewards']],
            'robot1_rewards': [float(x) for x in self.training_history['robot1_rewards']],
            'robot2_rewards': [float(x) for x in self.training_history['robot2_rewards']],
            'episode_lengths': [int(x) for x in self.training_history['episode_lengths']],
            'actor_losses': [float(x) for x in self.training_history['actor_losses']],
            'critic_losses': [float(x) for x in self.training_history['critic_losses']],
            'exploration_progress': [float(x) for x in self.training_history['exploration_progress']]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=4)
        
        print(f"Checkpoint saved at episode {episode}")
