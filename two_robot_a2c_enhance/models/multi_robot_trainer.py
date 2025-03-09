import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from two_robot_a2c_enhance.config import MODEL_DIR, ROBOT_CONFIG

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
        
        # 添加收斂檢查相關的參數
        self.convergence_window = 100  # 檢查最近100輪的數據
        self.reward_threshold = 0.01   # 獎勵收斂閾值
        self.loss_threshold = 0.001    # 損失收斂閾值
        self.target_exploration_rate = 0.95  # 目標探索完成率
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
        """根據當前策略選擇動作 - 修復NaN問題"""
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
        
        # 從概率分布中採樣動作 - 使用安全版本處理NaN
        robot1_probs = policy_dict['robot1_policy'][0, :valid_frontiers]
        robot2_probs = policy_dict['robot2_policy'][0, :valid_frontiers]
        
        # 檢查並修復NaN值
        if np.any(np.isnan(robot1_probs)):
            print("Warning: NaN detected in robot1 probabilities, using uniform distribution")
            robot1_probs = np.ones(valid_frontiers) / valid_frontiers
        else:
            # 確保總和為1
            robot1_sum = np.sum(robot1_probs)
            if robot1_sum <= 0 or np.isnan(robot1_sum):
                robot1_probs = np.ones(valid_frontiers) / valid_frontiers
            else:
                robot1_probs = robot1_probs / robot1_sum
        
        if np.any(np.isnan(robot2_probs)):
            print("Warning: NaN detected in robot2 probabilities, using uniform distribution")
            robot2_probs = np.ones(valid_frontiers) / valid_frontiers
        else:
            # 確保總和為1
            robot2_sum = np.sum(robot2_probs)
            if robot2_sum <= 0 or np.isnan(robot2_sum):
                robot2_probs = np.ones(valid_frontiers) / valid_frontiers
            else:
                robot2_probs = robot2_probs / robot2_sum
        
        # 安全採樣
        robot1_action = np.random.choice(valid_frontiers, p=robot1_probs)
        robot2_action = np.random.choice(valid_frontiers, p=robot2_probs)

        return robot1_action, robot2_action

    def compute_advantages(self, rewards, values, dones):
        # 不需要額外的next_value參數
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0
        next_value = 0  # 對於episode結束時的處理
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # 對於序列的最後一步
                delta = rewards[t] - values[t]  # 因為是episode結束，所以不需要下一個value
            else:
                # 對於序列中間的步驟
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        # 標準化優勢值
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages, returns


    def train_on_episode(self):
        # 獲取經驗數據
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
    
    
    
    def check_reward_convergence(self):
        """檢查獎勵是否收斂"""
        if len(self.training_history['episode_rewards']) < self.convergence_window * 2:
            return False
            
        recent_rewards = self.training_history['episode_rewards'][-self.convergence_window:]
        previous_rewards = self.training_history['episode_rewards'][-2*self.convergence_window:-self.convergence_window]
        
        recent_mean = np.mean(recent_rewards)
        previous_mean = np.mean(previous_rewards)
        
        is_converged = abs(recent_mean - previous_mean) < self.reward_threshold
        
        print(f"\nReward Convergence Check:")
        print(f"Recent mean reward: {recent_mean:.3f}")
        print(f"Previous mean reward: {previous_mean:.3f}")
        print(f"Difference: {abs(recent_mean - previous_mean):.3f}")
        print(f"Is converged: {is_converged}")
        
        return is_converged

    def check_loss_convergence(self):
        """檢查Actor和Critic損失是否收斂"""
        if len(self.training_history['actor_losses']) < self.convergence_window:
            return False
            
        recent_actor_losses = self.training_history['actor_losses'][-self.convergence_window:]
        recent_critic_losses = self.training_history['critic_losses'][-self.convergence_window:]
        
        mean_actor_loss = np.mean(recent_actor_losses)
        mean_critic_loss = np.mean(recent_critic_losses)
        
        is_converged = (mean_actor_loss < self.loss_threshold and 
                       mean_critic_loss < self.loss_threshold)
        
        print(f"\nLoss Convergence Check:")
        print(f"Mean actor loss: {mean_actor_loss:.6f}")
        print(f"Mean critic loss: {mean_critic_loss:.6f}")
        print(f"Is converged: {is_converged}")
        
        return is_converged

    def check_exploration_performance(self):
        """檢查探索性能是否達標"""
        if len(self.training_history['exploration_progress']) < self.convergence_window:
            return False
            
        recent_progress = self.training_history['exploration_progress'][-self.convergence_window:]
        mean_progress = np.mean(recent_progress)
        
        is_achieved = mean_progress >= self.target_exploration_rate
        
        print(f"\nExploration Performance Check:")
        print(f"Mean exploration progress: {mean_progress:.1%}")
        print(f"Target progress: {self.target_exploration_rate:.1%}")
        print(f"Is achieved: {is_achieved}")
        
        return is_achieved

    def should_stop_training(self):
        """綜合判斷是否應該停止訓練"""
        reward_converged = self.check_reward_convergence()
        loss_converged = self.check_loss_convergence()
        exploration_achieved = self.check_exploration_performance()
        
        # 至少滿足兩個條件才停止訓練
        conditions_met = sum([reward_converged, loss_converged, exploration_achieved])
        should_stop = conditions_met >= 2
        
        print("\nTraining Stop Criteria:")
        print(f"Conditions met: {conditions_met}/3")
        print(f"Should stop: {should_stop}")
        
        return should_stop
    
    

    def train(self, episodes=1000000, save_freq=10, accumulation_steps=4):
        """執行多機器人協同訓練 - 記憶體優化版
        
        Args:
            episodes (int): 訓練的總輪數
            save_freq (int): 保存模型的頻率（每多少輪保存一次）
            accumulation_steps (int): 梯度累積步數，用於降低記憶體使用
        """
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
                
                # 主訓練循環
                while not (self.robot1.check_done() or self.robot2.check_done()):
                    # 將長序列分成小批次處理，減少記憶體使用
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
                    
                    # 獲取當前狀態的價值估計 - 批次大小1，減少記憶體使用
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
                    
                    # 主動釋放不需要的大型變量的記憶體
                    if hasattr(tf, 'keras'):
                        tf.keras.backend.clear_session()
                    
                    # 定期清理 GPU 快取 (每100步)
                    if steps % 100 == 0 and hasattr(tf, 'config'):
                        try:
                            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                                tf.config.experimental.reset_memory_stats(gpu)
                        except:
                            pass
                    
                    # 更新可視化 (減少更新頻率)
                    if steps % (ROBOT_CONFIG['plot_interval'] * 2) == 0:
                        if self.robot1.plot:
                            self.robot1.plot_env()
                        if self.robot2.plot:
                            self.robot2.plot_env()
                
                # Episode結束，進行訓練 - 使用梯度累積優化記憶體使用
                if len(self.current_episode['states']) > 0:
                    self.train_on_episode_with_accumulation(accumulation_steps)
                
                # 更新訓練歷史
                exploration_progress = self.robot1.get_exploration_progress()
                self.training_history['episode_rewards'].append(total_reward)
                self.training_history['robot1_rewards'].append(robot1_total_reward)
                self.training_history['robot2_rewards'].append(robot2_total_reward)
                self.training_history['episode_lengths'].append(steps)
                self.training_history['exploration_progress'].append(exploration_progress)
                
                # 定期保存模型和檢查訓練狀態 (減少檢查頻率)
                if (episode + 1) % save_freq == 0:
                    self.save_checkpoint(episode + 1)
                    
                    # 每5個保存周期才做一次繪圖，減少IO操作
                    if (episode + 1) % (save_freq * 5) == 0:
                        self.plot_training_progress()
                    
                    # 每10個保存周期才檢查一次收斂性，減少計算開銷
                    if episode > self.convergence_window * 2 and (episode + 1) % (save_freq * 10) == 0:
                        print("\n" + "="*50)
                        print(f"Checking training status at episode {episode + 1}")
                        if self.should_stop_training():
                            print("Training converged. Stopping.")
                            break
                        print("="*50)
                
                # 列印基本訓練信息 - 簡化輸出
                print(f"\nEpisode {episode + 1}/{episodes} (Map {self.robot1.li_map})")
                print(f"Steps: {steps}, Total Reward: {total_reward:.2f}, Exploration: {exploration_progress:.1%}")
                
                # 主動進行垃圾回收
                try:
                    import gc
                    gc.collect()
                except:
                    pass
                
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






    def train_on_episode_with_accumulation(self, accumulation_steps=4):
        """使用梯度累積訓練，減少記憶體使用"""
        # 數據分批處理，確保每批數據量較小
        n_data = len(self.current_episode['states'])
        batch_size = max(1, n_data // accumulation_steps)
        
        # 初始化累積梯度
        actor_loss_total = 0
        critic_loss_total = 0
        
        for i in range(0, n_data, batch_size):
            # 計算此批次的結束索引
            end_idx = min(i + batch_size, n_data)
            batch_indices = slice(i, end_idx)
            
            # 準備批次數據
            states = np.array(self.current_episode['states'][batch_indices])
            frontiers = np.array(self.current_episode['frontiers'][batch_indices])
            robot1_pos = np.array(self.current_episode['robot1_pos'][batch_indices])
            robot2_pos = np.array(self.current_episode['robot2_pos'][batch_indices])
            robot1_target = np.array(self.current_episode['robot1_target'][batch_indices])
            robot2_target = np.array(self.current_episode['robot2_target'][batch_indices])
            
            robot1_values = np.array(self.current_episode['robot1_values'][batch_indices])
            robot2_values = np.array(self.current_episode['robot2_values'][batch_indices])
            robot1_rewards = np.array(self.current_episode['robot1_rewards'][batch_indices])
            robot2_rewards = np.array(self.current_episode['robot2_rewards'][batch_indices])
            robot1_dones = np.array(self.current_episode['robot1_dones'][batch_indices])
            robot2_dones = np.array(self.current_episode['robot2_dones'][batch_indices])
            robot1_actions = np.array(self.current_episode['robot1_actions'][batch_indices])
            robot2_actions = np.array(self.current_episode['robot2_actions'][batch_indices])
            
            # 計算優勢和回報
            robot1_advantages, robot1_returns = self.compute_advantages(
                robot1_rewards, robot1_values, robot1_dones)
            robot2_advantages, robot2_returns = self.compute_advantages(
                robot2_rewards, robot2_values, robot2_dones)
            
            # 訓練 Critic
            critic_loss = self.model.train_critic(
                states, frontiers,
                robot1_pos, robot2_pos,
                robot1_target, robot2_target,
                {
                    'robot1': robot1_returns,
                    'robot2': robot2_returns
                }
            )
            
            # 訓練 Actor
            actor_loss = self.model.train_actor(
                states, frontiers,
                robot1_pos, robot2_pos,
                robot1_target, robot2_target,
                {
                    'robot1': robot1_actions,
                    'robot2': robot2_actions
                },
                {
                    'robot1': robot1_advantages,
                    'robot2': robot2_advantages
                }
            )
            
            # 累積損失
            actor_loss_total += actor_loss
            critic_loss_total += critic_loss
            
            # 清理記憶體
            del states, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target
            del robot1_values, robot2_values, robot1_rewards, robot2_rewards
            del robot1_dones, robot2_dones, robot1_actions, robot2_actions
            del robot1_advantages, robot1_returns, robot2_advantages, robot2_returns
            
            # 主動釋放 TensorFlow 的內部緩存
            if hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()
        
        # 計算平均損失
        actor_loss_avg = actor_loss_total / accumulation_steps
        critic_loss_avg = critic_loss_total / accumulation_steps
        
        # 更新損失歷史
        self.training_history['actor_losses'].append(float(actor_loss_avg))
        self.training_history['critic_losses'].append(float(critic_loss_avg))
        
        return actor_loss_avg, critic_loss_avg
            
            
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
