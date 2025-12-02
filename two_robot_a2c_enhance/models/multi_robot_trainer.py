import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import gc
from two_robot_a2c_enhance.config import MODEL_DIR, ROBOT_CONFIG

# 在文件開頭設置GPU記憶體配置
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"GPU memory growth enabled for {physical_devices[0]}")
except Exception as e:
    print(f"GPU setup warning: {e}")

# 設置環境變數
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class MultiRobotACTrainer:
    def __init__(self, model, robot1, robot2, gamma=0.99, gae_lambda=0.95):
        self.model = model
        self.robot1 = robot1
        self.robot2 = robot2
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.map_size = self.robot1.map_size
        
        # 添加記憶體管理參數
        self.memory_cleanup_frequency = 5  # 每5個episode清理一次記憶體
        self.steps_since_cleanup = 0
        
        # 訓練歷史記錄
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'robot1_rewards': [],
            'robot2_rewards': [],
            'exploration_progress': [],
            'overlap_ratios': []
        }
        
        # 收斂檢查參數
        self.convergence_window = 100
        self.reward_threshold = 0.01
        self.loss_threshold = 0.001
        self.target_exploration_rate = 0.95
        
        # 經驗緩衝區
        self.reset_episode_buffer()
        
        # 地圖追蹤器
        self.map_tracker = None

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
    
    def clear_memory(self):
        """清理記憶體"""
        try:
            # 清理 TensorFlow 記憶體
            tf.keras.backend.clear_session()
            
            # 手動垃圾回收
            gc.collect()
            
            # 如果使用 GPU，嘗試清理 GPU 記憶體
            try:
                if tf.config.list_physical_devices('GPU'):
                    # 重新設置記憶體增長
                    physical_devices = tf.config.experimental.list_physical_devices('GPU')
                    if len(physical_devices) > 0:
                        tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except Exception as e:
                print(f"GPU memory cleanup warning: {e}")
                
            print("Memory cleared successfully")
            
        except Exception as e:
            print(f"Memory cleanup error: {e}")
    
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
        """根據當前策略選擇動作，專注於修復網路輸出而非替換策略"""
        if len(frontiers) == 0:
            return 0, 0

        # 準備輸入數據
        state_batch = np.expand_dims(state, 0)
        frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0)
        robot1_pos_batch = np.expand_dims(robot1_pos, 0)
        robot2_pos_batch = np.expand_dims(robot2_pos, 0)
        robot1_target_batch = np.expand_dims(robot1_target, 0)
        robot2_target_batch = np.expand_dims(robot2_target, 0)

        try:
            # 獲取動作概率分布
            policy_dict = self.model.predict_policy(
                state_batch, frontiers_batch,
                robot1_pos_batch, robot2_pos_batch,
                robot1_target_batch, robot2_target_batch
            )

            valid_frontiers = min(self.model.max_frontiers, len(frontiers))
            
            # 從概率分布中採樣動作 - 專注於修復網路輸出
            robot1_probs = policy_dict['robot1_policy'][0, :valid_frontiers]
            robot2_probs = policy_dict['robot2_policy'][0, :valid_frontiers]
            
            # 修復網路輸出的概率分布
            robot1_probs = self._repair_probability_distribution(robot1_probs, "Robot1")
            robot2_probs = self._repair_probability_distribution(robot2_probs, "Robot2")
            
            # 安全地選擇動作
            robot1_action = np.random.choice(valid_frontiers, p=robot1_probs)
            robot2_action = np.random.choice(valid_frontiers, p=robot2_probs)
            
        except Exception as e:
            # 如果選擇動作失敗，使用隨機動作
            print(f"警告：選擇動作時出錯: {str(e)}")
            valid_frontiers = min(self.model.max_frontiers, len(frontiers))
            robot1_action = np.random.randint(0, max(1, valid_frontiers))
            robot2_action = np.random.randint(0, max(1, valid_frontiers))
        
        return robot1_action, robot2_action

    def _repair_probability_distribution(self, probs, robot_name):
        """修復概率分布，使其有效且可用"""
        original_probs = probs.copy()
        
        # 檢查無效值
        invalid_mask = ~np.isfinite(probs) | (probs < 0)
        if np.any(invalid_mask):
            print(f"警告：{robot_name}存在 {np.sum(invalid_mask)} 個無效概率值")
            min_valid_prob = np.min(probs[~invalid_mask]) if np.any(~invalid_mask) else 1e-10
            probs[invalid_mask] = min_valid_prob * 0.1
        
        # 檢查和修復概率總和
        prob_sum = np.sum(probs)
        if prob_sum <= 1e-10 or not np.isfinite(prob_sum):
            print(f"警告：{robot_name}概率總和異常 ({prob_sum})，進行數值修復")
            probs = np.ones_like(probs) * 1e-7
            if np.any(np.isfinite(original_probs)):
                top_k = min(3, len(probs))
                top_indices = np.argsort(original_probs)[-top_k:]
                for idx in top_indices:
                    if idx < len(probs) and np.isfinite(original_probs[idx]):
                        probs[idx] = max(probs[idx], 1e-5)
        
        # 正規化概率，確保總和為1
        probs = probs / np.sum(probs)
        
        # 最後的安全檢查
        if not np.all(np.isfinite(probs)):
            print(f"警告：{robot_name}最終概率仍有無限值，進行最終修復")
            probs = np.ones(len(probs)) / len(probs)
        
        return probs

    def compute_advantages(self, rewards, values, dones):
        """計算優勢值"""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        # 標準化優勢值
        if np.std(advantages) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages * 0.0  # 如果標準差太小，設為0
            
        return advantages, returns

    def train_on_episode(self):
        """在episode結束後進行訓練"""
        if not self.current_episode['states']:
            return 0.0, 0.0
            
        try:
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
            
        except Exception as e:
            print(f"Training error in train_on_episode: {str(e)}")
            return 0.0, 0.0

    def train(self, episodes=1000000, save_freq=10):
        """執行多機器人協同訓練"""
        try:
            for episode in range(episodes):
                # 定期清理記憶體
                if episode % self.memory_cleanup_frequency == 0 and episode > 0:
                    self.clear_memory()
                
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
                
                # 初始化或重置地圖追蹤器
                if self.map_tracker is not None:
                    self.map_tracker.start_tracking()
                
                # Episode主循環
                max_steps = 1500  # 設置最大步數避免無限循環
                while (not (self.robot1.check_done() or self.robot2.check_done()) and 
                       steps < max_steps):
                    
                    try:
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
                        try:
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
                        except Exception as e:
                            print(f"Value prediction error: {e}")
                            robot1_value = 0.0
                            robot2_value = 0.0
                        
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
                        
                        # 更新地圖追蹤器
                        if self.map_tracker is not None:
                            self.map_tracker.update()
                        
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
                                
                    except Exception as e:
                        print(f"Step error at step {steps}: {e}")
                        break
                
                # 計算地圖重疊比例
                overlap_ratio = 0.0
                if self.map_tracker is not None:
                    try:
                        overlap_ratio = self.map_tracker.calculate_overlap()
                        self.training_history['overlap_ratios'].append(float(overlap_ratio))
                        self.map_tracker.save_current_maps(episode)
                    except Exception as e:
                        print(f"Map tracker error: {e}")
                        self.training_history['overlap_ratios'].append(0.0)
                
                # Episode結束，進行訓練
                try:
                    actor_loss, critic_loss = self.train_on_episode()
                except Exception as e:
                    print(f"Training error: {e}")
                    actor_loss, critic_loss = 0.0, 0.0
                
                # 更新訓練歷史
                exploration_progress = self.robot1.get_exploration_progress()
                self.training_history['episode_rewards'].append(total_reward)
                self.training_history['robot1_rewards'].append(robot1_total_reward)
                self.training_history['robot2_rewards'].append(robot2_total_reward)
                self.training_history['episode_lengths'].append(steps)
                self.training_history['actor_losses'].append(float(actor_loss))
                self.training_history['critic_losses'].append(float(critic_loss))
                self.training_history['exploration_progress'].append(exploration_progress)
                
                # 定期保存模型和檢查訓練狀態
                if (episode + 1) % save_freq == 0:
                    try:
                        self.save_checkpoint(episode + 1)
                        self.plot_training_progress()
                    except Exception as e:
                        print(f"Save/plot error: {e}")
                
                # 列印訓練信息
                print(f"\nEpisode {episode + 1}/{episodes} (Map {self.robot1.li_map})")
                print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
                print(f"Robot1 Reward: {robot1_total_reward:.2f}")
                print(f"Robot2 Reward: {robot2_total_reward:.2f}")
                print(f"Actor Loss: {float(actor_loss):.6f}")
                print(f"Critic Loss: {float(critic_loss):.6f}")
                print(f"Exploration Progress: {exploration_progress:.1%}")
                print(f"Map Overlap Ratio: {overlap_ratio:.2%}")
                
                if exploration_progress >= self.robot1.finish_percent:
                    print("Map Exploration Complete!")
                else:
                    print("Map Exploration Incomplete")
                
                # 監控記憶體狀態
                if episode % 10 == 0:
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
                    except ImportError:
                        pass
                
                # 重置環境
                try:
                    state = self.robot1.reset()
                    self.robot2.reset()
                except Exception as e:
                    print(f"Reset error: {e}")
                    break
            
            # 訓練結束後保存最終模型
            try:
                self.save_checkpoint(episodes)
                if self.map_tracker is not None:
                    self.map_tracker.stop_tracking()
                    self.map_tracker.plot_coverage_over_time()
            except Exception as e:
                print(f"Final save error: {e}")
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Training Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 確保清理資源
            try:
                self.clear_memory()
                if hasattr(self.robot1, 'cleanup_visualization'):
                    self.robot1.cleanup_visualization()
                if hasattr(self.robot2, 'cleanup_visualization'):
                    self.robot2.cleanup_visualization()
            except Exception as e:
                print(f"Cleanup error: {e}")

    def plot_training_progress(self):
        """繪製訓練進度圖"""
        try:
            # 決定圖形數量: 如果有重疊率數據, 則需要7個子圖
            n_plots = 7 if ('overlap_ratios' in self.training_history and 
                           self.training_history['overlap_ratios']) else 6
            
            fig, axs = plt.subplots(n_plots, 1, figsize=(12, n_plots * 3.5))
            
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
            
            # 如果有重疊率數據，則繪製重疊率圖
            if n_plots > 6 and 'overlap_ratios' in self.training_history:
                overlap_data = self.training_history['overlap_ratios']
                if len(overlap_data) < len(episodes):
                    padded_data = overlap_data + [0.0] * (len(episodes) - len(overlap_data))
                    overlap_data = padded_data[:len(episodes)]
                elif len(overlap_data) > len(episodes):
                    overlap_data = overlap_data[:len(episodes)]
                    
                axs[6].plot(episodes, overlap_data, color='#8B008B')
                axs[6].set_title('Map Overlap Ratio')
                axs[6].set_xlabel('Episode')
                axs[6].set_ylabel('Overlap Ratio')
                axs[6].grid(True)
                axs[6].set_ylim(0, 1.0)
            
            plt.tight_layout()
            plt.savefig('training_progress.png', dpi=150)
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
            plt.savefig('robots_rewards_comparison.png', dpi=150)
            plt.close()
            
            # 如果有重疊率數據，另外繪製一個單獨的重疊率圖
            if 'overlap_ratios' in self.training_history and self.training_history['overlap_ratios']:
                overlap_data = self.training_history['overlap_ratios']
                if len(overlap_data) < len(episodes):
                    padded_data = overlap_data + [0.0] * (len(episodes) - len(overlap_data))
                    overlap_data = padded_data[:len(episodes)]
                elif len(overlap_data) > len(episodes):
                    overlap_data = overlap_data[:len(episodes)]
                
                plt.figure(figsize=(10, 6))
                plt.plot(episodes, overlap_data, color='#8B008B', linewidth=2)
                plt.fill_between(episodes, overlap_data, alpha=0.3, color='#9370DB')
                plt.title('Map Overlap Ratio Over Episodes')
                plt.xlabel('Episode')
                plt.ylabel('Overlap Ratio')
                plt.ylim(0, 1.0)
                plt.grid(True)
                plt.savefig('map_overlap_ratio.png', dpi=150)
                plt.close()
                
        except Exception as e:
            print(f"Plot error: {e}")
    
    def save_checkpoint(self, episode):
        """保存檢查點 - 修改為.h5格式"""
        try:
            # 用零填充確保文件名排序正確
            ep_str = str(episode).zfill(6)
            
            # 保存模型為.h5格式
            model_path = os.path.join(MODEL_DIR, f'multi_robot_model_ac_ep{ep_str}')
            print(f"\n正在保存檢查點 #{episode} 到: {model_path}")
            
            # 保存模型(.h5格式)
            save_result = self.model.save(model_path)
            
            if not save_result:
                print("警告: 模型保存失敗，可能無法正確載入")
            
            # 保存訓練歷史
            history_path = os.path.join(MODEL_DIR, f'multi_robot_history_ac_ep{ep_str}.json')
            print(f"保存訓練歷史到: {history_path}")
            
            history_to_save = {
                'episode_rewards': [float(x) for x in self.training_history['episode_rewards']],
                'robot1_rewards': [float(x) for x in self.training_history['robot1_rewards']],
                'robot2_rewards': [float(x) for x in self.training_history['robot2_rewards']],
                'episode_lengths': [int(x) for x in self.training_history['episode_lengths']],
                'actor_losses': [float(x) for x in self.training_history['actor_losses']],
                'critic_losses': [float(x) for x in self.training_history['critic_losses']],
                'exploration_progress': [float(x) for x in self.training_history['exploration_progress']]
            }
            
            # 添加重疊率數據（如果有）
            if 'overlap_ratios' in self.training_history and self.training_history['overlap_ratios']:
                history_to_save['overlap_ratios'] = [float(x) for x in self.training_history['overlap_ratios']]
            
            with open(history_path, 'w') as f:
                json.dump(history_to_save, f, indent=4)
            
            print(f"檢查點 #{episode} 保存完成(.h5格式)")
            
        except Exception as e:
            print(f"保存檢查點時出錯: {str(e)}")

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