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
            'exploration_progress': [],
            'overlap_ratios': []  # 新增: 記錄機器人探索區域的交集比例
        }
        
        # 添加收斂檢查相關的參數
        self.convergence_window = 100  # 檢查最近100輪的數據
        self.reward_threshold = 0.01   # 獎勵收斂閾值
        self.loss_threshold = 0.001    # 損失收斂閾值
        self.target_exploration_rate = 0.95  # 目標探索完成率
        # 經驗緩衝區用於儲存當前episode的軌跡
        self.reset_episode_buffer()
        
        # 地圖追蹤器（將由 train.py 設置）
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
        """根據當前策略選擇動作，結合兩種方法的穩定性處理"""
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
        
        # 從概率分布中採樣動作 - 整合兩種方法的安全處理
        robot1_probs = policy_dict['robot1_policy'][0, :valid_frontiers]
        robot2_probs = policy_dict['robot2_policy'][0, :valid_frontiers]
        
        # 替換NaN、正無窮和負無窮值
        robot1_probs = np.nan_to_num(robot1_probs, nan=1.0/valid_frontiers, posinf=1.0, neginf=0.0)
        robot2_probs = np.nan_to_num(robot2_probs, nan=1.0/valid_frontiers, posinf=1.0, neginf=0.0)
        
        # 確保所有概率值都為正且有最小值
        min_prob = 1e-10
        robot1_probs = np.maximum(robot1_probs, min_prob)
        robot2_probs = np.maximum(robot2_probs, min_prob)
        
        # 安全的歸一化處理
        robot1_sum = np.sum(robot1_probs)
        robot2_sum = np.sum(robot2_probs)
        
        # 檢測異常情況並使用均勻分布
        if robot1_sum < 1e-8 or not np.isfinite(robot1_sum):
            print("警告：Robot1概率總和異常，使用均勻分布")
            robot1_probs = np.ones(valid_frontiers) / valid_frontiers
        else:
            robot1_probs = robot1_probs / robot1_sum
            
        if robot2_sum < 1e-8 or not np.isfinite(robot2_sum):
            print("警告：Robot2概率總和異常，使用均勻分布")
            robot2_probs = np.ones(valid_frontiers) / valid_frontiers
        else:
            robot2_probs = robot2_probs / robot2_sum
        
        # 添加小的擾動以確保概率分布多樣性
        epsilon = 1e-5
        robot1_probs = (1 - epsilon) * robot1_probs + epsilon / valid_frontiers
        robot2_probs = (1 - epsilon) * robot2_probs + epsilon / valid_frontiers
        
        # 再次正規化
        robot1_probs = robot1_probs / np.sum(robot1_probs)
        robot2_probs = robot2_probs / np.sum(robot2_probs)
        
        # 最後確保沒有任何NaN和概率總和為1
        robot1_probs = np.nan_to_num(robot1_probs, nan=1.0/valid_frontiers)
        robot2_probs = np.nan_to_num(robot2_probs, nan=1.0/valid_frontiers)
        
        # 確保總和精確地為1
        if abs(np.sum(robot1_probs) - 1.0) > 1e-10:
            robot1_probs = robot1_probs / np.sum(robot1_probs)
        if abs(np.sum(robot2_probs) - 1.0) > 1e-10:
            robot2_probs = robot2_probs / np.sum(robot2_probs)
        
        # 安全選擇動作，備有備選方案
        try:
            robot1_action = np.random.choice(valid_frontiers, p=robot1_probs)
            robot2_action = np.random.choice(valid_frontiers, p=robot2_probs)
        except ValueError as e:
            # 詳細記錄問題
            print(f"警告：選擇動作時出錯: {str(e)}")
            print(f"Robot1概率: min={np.min(robot1_probs)}, max={np.max(robot1_probs)}, sum={np.sum(robot1_probs)}")
            print(f"Robot2概率: min={np.min(robot2_probs)}, max={np.max(robot2_probs)}, sum={np.sum(robot2_probs)}")
            
            # 備選方案1：使用argmax（確定性選擇）
            robot1_action = np.argmax(robot1_probs)
            robot2_action = np.argmax(robot2_probs)
            print(f"使用確定性選擇: robot1={robot1_action}, robot2={robot2_action}")
        
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
    
    

    def train(self, episodes=1000000, save_freq=10):
        """執行多機器人協同訓練
        
        Args:
            episodes (int): 訓練的總輪數
            save_freq (int): 保存模型的頻率（每多少輪保存一次）
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
                
                # 初始化或重置地圖追蹤器
                if self.map_tracker is not None:
                    self.map_tracker.start_tracking()
                
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
                
                # 計算地圖重疊比例
                overlap_ratio = 0.0
                if self.map_tracker is not None:
                    overlap_ratio = self.map_tracker.calculate_overlap()
                    self.training_history['overlap_ratios'].append(float(overlap_ratio))
                    
                    # 每個訓練回合結束時保存當前地圖
                    self.map_tracker.save_current_maps(episode)
                
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
                
                # 定期保存模型和檢查訓練狀態
                if (episode + 1) % save_freq == 0:
                    self.save_checkpoint(episode + 1)
                    self.plot_training_progress()
                    
                    if episode > self.convergence_window * 2:
                        print("\n" + "="*50)
                        print(f"Checking training status at episode {episode + 1}")
                        #self.check_training_status()
                        print("="*50)
                
                # 列印基本訓練信息
                print(f"\nEpisode {episode + 1}/{episodes} (Map {self.robot1.li_map})")
                print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
                print(f"Robot1 Reward: {robot1_total_reward:.2f}")
                print(f"Robot2 Reward: {robot2_total_reward:.2f}")
                print(f"Actor Loss: {float(actor_loss):.6f}")
                print(f"Critic Loss: {float(critic_loss):.6f}")
                print(f"Exploration Progress: {exploration_progress:.1%}")
                # 列印地圖重疊比例
                print(f"Map Overlap Ratio: {overlap_ratio:.2%}")
                
                if exploration_progress >= self.robot1.finish_percent:
                    print("Map Exploration Complete!")
                else:
                    print("Map Exploration Incomplete")
                
                # 添加收斂監控信息（根據可用的歷史數據進行調整）
                print("\nConvergence Monitoring:")
                print("-" * 20)
                
                # 計算可用的歷史數據長度
                available_history = len(self.training_history['episode_rewards'])
                
                if available_history > 0:
                    # 獎勵統計
                    print(f"Reward Statistics:")
                    recent_rewards = self.training_history['episode_rewards'][-min(available_history, self.convergence_window):]
                    current_mean_reward = np.mean(recent_rewards)
                    print(f"- Current mean reward: {current_mean_reward:.3f}")
                    
                    if available_history > self.convergence_window:
                        previous_rewards = self.training_history['episode_rewards'][-2*min(available_history//2, self.convergence_window):-min(available_history//2, self.convergence_window)]
                        previous_mean_reward = np.mean(previous_rewards)
                        reward_diff = abs(current_mean_reward - previous_mean_reward)
                        print(f"- Previous mean reward: {previous_mean_reward:.3f}")
                        print(f"- Reward change: {reward_diff:.3f}")
                    
                    # 損失統計
                    print(f"\nLoss Statistics:")
                    recent_actor_losses = self.training_history['actor_losses'][-min(available_history, self.convergence_window):]
                    recent_critic_losses = self.training_history['critic_losses'][-min(available_history, self.convergence_window):]
                    print(f"- Mean actor loss: {np.mean(recent_actor_losses):.6f}")
                    print(f"- Mean critic loss: {np.mean(recent_critic_losses):.6f}")
                    
                    # 探索進度統計
                    print(f"\nExploration Statistics:")
                    recent_progress = self.training_history['exploration_progress'][-min(available_history, self.convergence_window):]
                    mean_progress = np.mean(recent_progress)
                    print(f"- Current progress: {mean_progress:.1%}")
                    print(f"- Target: {self.target_exploration_rate:.1%}")
                else:
                    print("Insufficient history data for statistics")
                
                print("-" * 50)
                
                # 重置環境
                state = self.robot1.reset()
                self.robot2.reset()
            
            # 訓練結束後保存最終模型
            self.save_checkpoint(episodes)
            
            # 訓練結束後生成覆蓋率圖表
            if self.map_tracker is not None:
                self.map_tracker.stop_tracking()
                self.map_tracker.plot_coverage_over_time()
            
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
        # 決定圖形數量: 如果有重疊率數據, 則需要8個子圖
        n_plots = 7 if 'overlap_ratios' in self.training_history and self.training_history['overlap_ratios'] else 6
        
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
        if 'overlap_ratios' in self.training_history and self.training_history['overlap_ratios']:
            # 確保數據長度與 episodes 一致
            overlap_data = self.training_history['overlap_ratios']
            if len(overlap_data) < len(episodes):
                padded_data = overlap_data + [0.0] * (len(episodes) - len(overlap_data))
                overlap_data = padded_data[:len(episodes)]
            elif len(overlap_data) > len(episodes):
                overlap_data = overlap_data[:len(episodes)]
                
            axs[6].plot(episodes, overlap_data, color='#8B008B')  # 使用深紫色
            axs[6].set_title('Map Overlap Ratio')
            axs[6].set_xlabel('Episode')
            axs[6].set_ylabel('Overlap Ratio')
            axs[6].grid(True)
            axs[6].set_ylim(0, 1.0)
        
        # # 繪製探索進度
        # if n_plots > 7:
        #     axs[7].plot(episodes, self.training_history['exploration_progress'], color='#2F4F4F')
        # else:
        #     axs[6].plot(episodes, self.training_history['exploration_progress'], color='#2F4F4F')
        
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
            plt.savefig('map_overlap_ratio.png')
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
        print(f"\n正在保存檢查點 #{episode} 到: {model_path}")
        
        # 保存模型 (使用 h5 格式)
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
        
        try:
            with open(history_path, 'w') as f:
                json.dump(history_to_save, f, indent=4)
        except Exception as e:
            print(f"保存訓練歷史時出錯: {str(e)}")
        
        print(f"檢查點 #{episode} 保存完成")