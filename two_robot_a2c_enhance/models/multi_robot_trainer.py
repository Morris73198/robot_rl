import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from two_robot_a2c.config import MODEL_DIR, ROBOT_CONFIG, REWARD_CONFIG
from two_robot_a2c_enhance.environment.robot_local_map_tracker import RobotIndividualMapTracker

class EnhancedMultiRobotA2CTrainer:
    def __init__(self, model, robot1, robot2, memory_size=10000, batch_size=16, gamma=0.99):
        """初始化多機器人A2C訓練器
        
        Args:
            model: A2C模型
            robot1: 第一個機器人實例
            robot2: 第二個機器人實例
            memory_size: 經驗回放緩衝區大小
            batch_size: 批次大小
            gamma: 折扣因子
        """
        self.model = model
        self.robot1 = robot1
        self.robot2 = robot2
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = 0.95
        
        self.map_size = self.robot1.map_size
        
        # 訓練參數
        self.epsilon = 1.0  # 探索率 (用於動作選擇)
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰減

        self.overlap_ratios = []
        
        # 訓練歷史記錄
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'exploration_rates': [],
            'losses': [],
            'robot1_rewards': [],
            'robot2_rewards': [],
            'exploration_progress': [],
            'robot1_entropy': [],
            'robot2_entropy': [],
            'robot1_value_loss': [],
            'robot2_value_loss': [],
            'robot1_policy_loss': [],
            'robot2_policy_loss': []
        }
        
        # 創建機器人個人地圖追蹤器
        self.map_tracker = RobotIndividualMapTracker(robot1, robot2)
        
        # 優勢函數計算的標準化參數
        self.advantage_epsilon = 1e-8
        
        # 記憶緩衝區 (儲存軌跡)
        self.trajectory_buffer = {
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
            'robot1_logits': [],
            'robot2_logits': [],
            'dones': []
        }
    
    def reset_trajectory_buffer(self):
        """重置軌跡緩衝區"""
        self.trajectory_buffer = {
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
            'robot1_logits': [],
            'robot2_logits': [],
            'dones': []
        }
    
    def add_to_trajectory(self, state, frontiers, robot1_pos, robot2_pos, 
                        robot1_target, robot2_target, robot1_action, robot2_action, 
                        robot1_reward, robot2_reward, robot1_value, robot2_value, 
                        robot1_logits, robot2_logits, done):
        """添加到軌跡緩衝區"""
        self.trajectory_buffer['states'].append(state)
        self.trajectory_buffer['frontiers'].append(frontiers)
        self.trajectory_buffer['robot1_pos'].append(robot1_pos)
        self.trajectory_buffer['robot2_pos'].append(robot2_pos)
        self.trajectory_buffer['robot1_target'].append(robot1_target)
        self.trajectory_buffer['robot2_target'].append(robot2_target)
        self.trajectory_buffer['robot1_actions'].append(robot1_action)
        self.trajectory_buffer['robot2_actions'].append(robot2_action)
        self.trajectory_buffer['robot1_rewards'].append(robot1_reward)
        self.trajectory_buffer['robot2_rewards'].append(robot2_reward)
        self.trajectory_buffer['robot1_values'].append(robot1_value)
        self.trajectory_buffer['robot2_values'].append(robot2_value)
        self.trajectory_buffer['robot1_logits'].append(robot1_logits)
        self.trajectory_buffer['robot2_logits'].append(robot2_logits)
        self.trajectory_buffer['dones'].append(done)
    
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
            return np.array([0.0, 0.0])  # 如果沒有目標，返回原點
        normalized = np.array([
            target[0] / float(self.map_size[1]),
            target[1] / float(self.map_size[0])
        ])
        return normalized
    
    def choose_actions(self, state, frontiers, robot1_pos, robot2_pos, 
                    robot1_target, robot2_target):
        """基於策略機率選擇動作，不添加額外人工規則"""
        if len(frontiers) == 0:
            return 0, 0, 0.0, 0.0, np.zeros(self.model.max_frontiers), np.zeros(self.model.max_frontiers)
        
        # 準備輸入
        state_batch = np.expand_dims(state, 0).astype(np.float32)
        frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0).astype(np.float32)
        robot1_pos_batch = np.expand_dims(robot1_pos, 0).astype(np.float32)
        robot2_pos_batch = np.expand_dims(robot2_pos, 0).astype(np.float32)
        robot1_target_batch = np.expand_dims(robot1_target, 0).astype(np.float32)
        robot2_target_batch = np.expand_dims(robot2_target, 0).astype(np.float32)
        
        # 確定有效前沿點數量
        valid_frontiers = min(self.model.max_frontiers, len(frontiers))
        
        # 獲取模型預測
        try:
            predictions = self.model.predict(
                state_batch, 
                frontiers_batch, 
                robot1_pos_batch, 
                robot2_pos_batch,
                robot1_target_batch, 
                robot2_target_batch
            )
            
            robot1_policy = predictions['robot1_policy'][0]
            robot2_policy = predictions['robot2_policy'][0]
            robot1_value = float(predictions['robot1_value'][0][0])
            robot2_value = float(predictions['robot2_value'][0][0])
            robot1_logits = predictions['robot1_logits'][0]
            robot2_logits = predictions['robot2_logits'][0]
        except Exception as e:
            print(f"模型預測錯誤: {str(e)}")
            # 預測失敗時使用均勻分布
            robot1_policy = np.ones(self.model.max_frontiers) / self.model.max_frontiers
            robot2_policy = np.ones(self.model.max_frontiers) / self.model.max_frontiers
            robot1_value = 0.0
            robot2_value = 0.0
            robot1_logits = np.zeros(self.model.max_frontiers)
            robot2_logits = np.zeros(self.model.max_frontiers)
        
        # 只考慮有效的前沿點
        robot1_probs = robot1_policy[:valid_frontiers].copy()
        robot2_probs = robot2_policy[:valid_frontiers].copy()
        
        # 處理數值問題
        robot1_probs = np.nan_to_num(robot1_probs, nan=1.0/valid_frontiers)
        robot2_probs = np.nan_to_num(robot2_probs, nan=1.0/valid_frontiers)
        
        # 確保概率和為1
        robot1_sum = np.sum(robot1_probs)
        robot2_sum = np.sum(robot2_probs)
        
        if robot1_sum > 0:
            robot1_probs = robot1_probs / robot1_sum
        else:
            robot1_probs = np.ones(valid_frontiers) / valid_frontiers
            
        if robot2_sum > 0:
            robot2_probs = robot2_probs / robot2_sum
        else:
            robot2_probs = np.ones(valid_frontiers) / valid_frontiers
        
        # 根據是否使用epsilon-greedy策略決定動作選擇方式
        if np.random.random() < self.epsilon:
            # 探索模式：完全隨機選擇，不基於模型輸出
            robot1_action = np.random.randint(valid_frontiers)
            
            # Robot2也完全隨機選擇
            robot2_action = np.random.randint(valid_frontiers)
        else:
            # 利用模式：完全基於策略網絡的輸出
            # 使用策略機率進行採樣，而不是直接選擇最高概率的動作
            try:
                robot1_action = np.random.choice(valid_frontiers, p=robot1_probs)
                robot2_action = np.random.choice(valid_frontiers, p=robot2_probs)
            except ValueError as e:
                print(f"採樣策略時出錯: {str(e)}")
                print(f"robot1_probs: {robot1_probs}, sum: {np.sum(robot1_probs)}")
                print(f"robot2_probs: {robot2_probs}, sum: {np.sum(robot2_probs)}")
                
                # 發生錯誤時回退到均勻採樣
                robot1_action = np.random.randint(valid_frontiers)
                robot2_action = np.random.randint(valid_frontiers)
        
        return robot1_action, robot2_action, robot1_value, robot2_value, robot1_logits, robot2_logits

    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        """更穩定的回報和優勢計算"""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        
        # 保持合理的獎勵尺度
        reward_scale = 1.0
        scaled_rewards = rewards * reward_scale
        
        # GAE參數
        gamma = self.gamma
        lambda_param = 0.95
        
        # 從後向前計算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_val = values[t+1]
            
            # 計算TD誤差
            delta = scaled_rewards[t] + gamma * next_val * next_non_terminal - values[t]
            
            # 計算GAE
            gae = delta + gamma * lambda_param * next_non_terminal * gae
            
            # 保存優勢和回報
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # 穩健的標準化 - 使用百分位數裁剪極端值
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        
        # 標準化
        normalized_advantages = (advantages - adv_mean) / adv_std
        
        # 限制範圍，避免極端值
        normalized_advantages = np.clip(normalized_advantages, -3.0, 3.0)
        
        # 適當放大
        advantages_scale = 5.0
        scaled_advantages = normalized_advantages * advantages_scale
        
        return returns, scaled_advantages
    
    def process_trajectory(self, last_robot1_value=0, last_robot2_value=0):
        """處理軌跡數據，計算回報和優勢，並進行訓練"""
        if len(self.trajectory_buffer['states']) == 0:
            return {}
        
        # 提取數據
        states = np.array(self.trajectory_buffer['states'], dtype=np.float32)
        frontiers = np.array(self.trajectory_buffer['frontiers'], dtype=np.float32)
        robot1_pos = np.array(self.trajectory_buffer['robot1_pos'], dtype=np.float32)
        robot2_pos = np.array(self.trajectory_buffer['robot2_pos'], dtype=np.float32)
        robot1_target = np.array(self.trajectory_buffer['robot1_target'], dtype=np.float32)
        robot2_target = np.array(self.trajectory_buffer['robot2_target'], dtype=np.float32)
        robot1_actions = np.array(self.trajectory_buffer['robot1_actions'], dtype=np.int32)
        robot2_actions = np.array(self.trajectory_buffer['robot2_actions'], dtype=np.int32)
        robot1_rewards = np.array(self.trajectory_buffer['robot1_rewards'], dtype=np.float32)
        robot2_rewards = np.array(self.trajectory_buffer['robot2_rewards'], dtype=np.float32)
        robot1_values = np.array(self.trajectory_buffer['robot1_values'], dtype=np.float32)
        robot2_values = np.array(self.trajectory_buffer['robot2_values'], dtype=np.float32)
        robot1_logits = np.array(self.trajectory_buffer['robot1_logits'], dtype=np.float32)
        robot2_logits = np.array(self.trajectory_buffer['robot2_logits'], dtype=np.float32)
        dones = np.array(self.trajectory_buffer['dones'], dtype=np.float32)
        
        # 計算回報和優勢
        robot1_returns, robot1_advantages = self.compute_returns_and_advantages(
            robot1_rewards, robot1_values, dones, last_robot1_value
        )
        
        robot2_returns, robot2_advantages = self.compute_returns_and_advantages(
            robot2_rewards, robot2_values, dones, last_robot2_value
        )
        
        # 訓練模型
        loss_metrics = self.model.train_batch(
            states=states,
            frontiers=frontiers,
            robot1_pos=robot1_pos,
            robot2_pos=robot2_pos,
            robot1_target=robot1_target,
            robot2_target=robot2_target,
            robot1_actions=robot1_actions,
            robot2_actions=robot2_actions,
            robot1_advantages=robot1_advantages,
            robot2_advantages=robot2_advantages,
            robot1_returns=robot1_returns,
            robot2_returns=robot2_returns,
            robot1_old_values=robot1_values,
            robot2_old_values=robot2_values,
            robot1_old_logits=robot1_logits,
            robot2_old_logits=robot2_logits
        )
        
        return loss_metrics
    
    def train(self, episodes=1000000, save_freq=10):
        """執行多機器人協同訓練"""
        try:
            for episode in range(episodes):
                # 初始化環境和狀態
                state = self.robot1.begin()
                self.robot2.begin()
                
                # 啟動地圖追蹤器
                self.map_tracker.start_tracking()
                
                # 初始化episode統計
                total_reward = 0
                robot1_total_reward = 0
                robot2_total_reward = 0
                steps = 0
                episode_losses = []
                
                # 在循環外定義 MIN_TARGET_DISTANCE，避免未定義錯誤
                MIN_TARGET_DISTANCE = self.robot1.sensor_range * 1.5
                
                # 重置軌跡緩衝區
                self.reset_trajectory_buffer()
                
                # 實際訓練循環
                while not (self.robot1.check_done() or self.robot2.check_done() or steps >= 1500):
                    frontiers = self.robot1.get_frontiers()
                    if len(frontiers) == 0:
                        break
                        
                    # 獲取當前狀態
                    robot1_pos = self.robot1.get_normalized_position()
                    robot2_pos = self.robot2.get_normalized_position()
                    old_robot1_pos = self.robot1.robot_position.copy()
                    old_robot2_pos = self.robot2.robot_position.copy()
                    
                    # 標準化目標位置
                    map_dims = np.array([float(self.robot1.map_size[1]), float(self.robot1.map_size[0])])
                    robot1_target = (np.zeros(2) if self.robot1.current_target_frontier is None 
                                else self.robot1.current_target_frontier / map_dims)
                    robot2_target = (np.zeros(2) if self.robot2.current_target_frontier is None 
                                else self.robot2.current_target_frontier / map_dims)
                    
                    # 選擇動作
                    robot1_action, robot2_action, robot1_value, robot2_value, robot1_logits, robot2_logits = \
                        self.choose_actions(state, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target)
                    
                    # 設置目標並執行移動
                    robot1_target_point = frontiers[robot1_action]
                    robot2_target_point = frontiers[robot2_action]
                    
                    # 移動機器人
                    next_state1, r1, d1 = self.robot1.move_to_frontier(robot1_target_point)
                    robot1_reward = r1
                    # 更新共享地圖
                    self.robot2.op_map = self.robot1.op_map.copy()
                    
                    # 檢查robot2是否需要重新選擇目標（如果robot1選了相近的點）
                    if not d1 and self.robot2.current_target_frontier is not None:
                        distance_between_targets = np.linalg.norm(
                            self.robot2.current_target_frontier - robot1_target_point)
                        
                        if distance_between_targets < MIN_TARGET_DISTANCE:
                            # 如果需要重新選擇，且我們在使用模型預測模式時
                            if np.random.random() >= self.epsilon:
                                # 重新獲取模型預測用於協調
                                predictions = self.model.predict(
                                    np.expand_dims(state, 0),
                                    np.expand_dims(self.pad_frontiers(frontiers), 0),
                                    np.expand_dims(robot1_pos, 0),
                                    np.expand_dims(robot2_pos, 0),
                                    np.expand_dims(robot1_target, 0),
                                    np.expand_dims(robot2_target, 0)
                                )
                                
                                # 提取策略
                                robot2_policy = predictions['robot2_policy'][0]
                                valid_frontiers = min(self.model.max_frontiers, len(frontiers))
                                robot2_policy = robot2_policy[:valid_frontiers]
                                
                                # 處理 NaN 和數值穩定性
                                robot2_policy = np.nan_to_num(robot2_policy, nan=1.0/valid_frontiers)
                                robot2_policy = np.maximum(robot2_policy, 1e-10)
                                
                                robot2_sum = np.sum(robot2_policy)
                                if robot2_sum < 1e-8:
                                    robot2_policy = np.ones(valid_frontiers) / valid_frontiers
                                else:
                                    robot2_policy = robot2_policy / robot2_sum
                                
                                # 調整策略以避開robot1的目標
                                for i in range(valid_frontiers):
                                    distance = np.linalg.norm(frontiers[i] - robot1_target_point)
                                    if distance < MIN_TARGET_DISTANCE:
                                        penalty = 1.0 - (distance / MIN_TARGET_DISTANCE)
                                        robot2_policy[i] *= (1.0 - penalty * 0.95)
                                
                                # 再次確保策略有效
                                robot2_policy = np.maximum(robot2_policy, 1e-10)
                                robot2_sum = np.sum(robot2_policy)
                                
                                if robot2_sum < 1e-8:
                                    # 如果所有概率都接近零，選擇遠離robot1的點
                                    distances = np.linalg.norm(frontiers - robot1_target_point, axis=1)
                                    farthest_indices = np.argsort(distances)[-min(5, valid_frontiers):]
                                    robot2_action = farthest_indices[np.random.randint(len(farthest_indices))]
                                    robot2_target_point = frontiers[robot2_action]
                                else:
                                    # 重新歸一化並選擇
                                    robot2_policy = robot2_policy / robot2_sum
                                    robot2_policy = np.nan_to_num(robot2_policy, nan=1.0/valid_frontiers)
                                    try:
                                        robot2_action = np.random.choice(valid_frontiers, p=robot2_policy)
                                        robot2_target_point = frontiers[robot2_action]
                                    except ValueError:
                                        # 如果出錯，選擇最遠的點
                                        distances = np.linalg.norm(frontiers - robot1_target_point, axis=1)
                                        robot2_action = np.argmax(distances[:valid_frontiers])
                                        robot2_target_point = frontiers[robot2_action]
                            else:
                                # 在隨機模式下，簡單地選擇一個遠離robot1目標的點
                                distances = np.linalg.norm(frontiers - robot1_target_point, axis=1)
                                farthest_indices = np.argsort(distances)[-min(5, len(frontiers)):]
                                robot2_action = farthest_indices[np.random.randint(len(farthest_indices))]
                                robot2_target_point = frontiers[robot2_action]
                    
                    next_state2, r2, d2 = self.robot2.move_to_frontier(robot2_target_point)
                    robot2_reward = r2
                    # 更新共享地圖
                    self.robot1.op_map = self.robot2.op_map.copy()
                    
                    # 更新機器人位置
                    self.robot1.other_robot_position = self.robot2.robot_position.copy()
                    self.robot2.other_robot_position = self.robot1.robot_position.copy()
                    
                    # 更新地圖追蹤器
                    self.map_tracker.update()
                    
                    # 為了保持數值穩定性，裁剪獎勵值
                    robot1_reward = np.clip(robot1_reward, -10, 10)
                    robot2_reward = np.clip(robot2_reward, -10, 10)
                    
                    # 確定是否完成
                    done = d1 or d2
                    
                    # 添加到軌跡緩衝區
                    self.add_to_trajectory(
                        state=state,
                        frontiers=self.pad_frontiers(frontiers),
                        robot1_pos=robot1_pos,
                        robot2_pos=robot2_pos,
                        robot1_target=robot1_target,
                        robot2_target=robot2_target,
                        robot1_action=robot1_action,
                        robot2_action=robot2_action,
                        robot1_reward=robot1_reward,
                        robot2_reward=robot2_reward,
                        robot1_value=robot1_value,
                        robot2_value=robot2_value,
                        robot1_logits=robot1_logits,
                        robot2_logits=robot2_logits,
                        done=done
                    )
                    
                    # 更新狀態和統計
                    state = next_state1
                    total_reward += (robot1_reward + robot2_reward)
                    robot1_total_reward += robot1_reward
                    robot2_total_reward += robot2_reward
                    steps += 1
                    
                    # 更新視覺化
                    if steps % ROBOT_CONFIG['plot_interval'] == 0:
                        if self.robot1.plot:
                            self.robot1.plot_env()
                        if self.robot2.plot:
                            self.robot2.plot_env()
                    
                    # 如果完成或達到批次大小，處理軌跡並訓練
                    if done or steps % self.batch_size == 0:
                        # 獲取最後狀態的價值估計
                        if not done:
                            # 準備輸入
                            state_batch = np.expand_dims(state, 0)
                            next_frontiers = self.robot1.get_frontiers()
                            frontiers_batch = np.expand_dims(self.pad_frontiers(next_frontiers), 0)
                            next_robot1_pos = self.robot1.get_normalized_position()
                            next_robot2_pos = self.robot2.get_normalized_position()
                            robot1_pos_batch = np.expand_dims(next_robot1_pos, 0)
                            robot2_pos_batch = np.expand_dims(next_robot2_pos, 0)
                            next_robot1_target = self.get_normalized_target(self.robot1.current_target_frontier)
                            next_robot2_target = self.get_normalized_target(self.robot2.current_target_frontier)
                            robot1_target_batch = np.expand_dims(next_robot1_target, 0)
                            robot2_target_batch = np.expand_dims(next_robot2_target, 0)
                            
                            # 獲取最後狀態的價值估計
                            predictions = self.model.predict(
                                state_batch, frontiers_batch, 
                                robot1_pos_batch, robot2_pos_batch,
                                robot1_target_batch, robot2_target_batch
                            )
                            
                            last_robot1_value = predictions['robot1_value'][0][0]
                            last_robot2_value = predictions['robot2_value'][0][0]
                        else:
                            # 如果回合結束，最後狀態的價值為0
                            last_robot1_value = 0
                            last_robot2_value = 0
                            
                        # 處理軌跡並訓練
                        loss_metrics = self.process_trajectory(last_robot1_value, last_robot2_value)
                        
                        if loss_metrics:
                            episode_losses.append(loss_metrics['total_loss'].numpy())
                            
                            # 記錄額外的指標
                            self.training_history['robot1_entropy'].append(loss_metrics['robot1_entropy'].numpy())
                            self.training_history['robot2_entropy'].append(loss_metrics['robot2_entropy'].numpy())
                            self.training_history['robot1_value_loss'].append(loss_metrics['robot1_value_loss'].numpy())
                            self.training_history['robot2_value_loss'].append(loss_metrics['robot2_value_loss'].numpy())
                            self.training_history['robot1_policy_loss'].append(loss_metrics['robot1_policy_loss'].numpy())
                            self.training_history['robot2_policy_loss'].append(loss_metrics['robot2_policy_loss'].numpy())
                        
                        # 重置軌跡緩衝區
                        self.reset_trajectory_buffer()
                
                # 此處代碼在訓練循環外，一個回合結束時執行）
                overlap_ratio = self.map_tracker.calculate_overlap()
                robot1_ratio, robot2_ratio = self.map_tracker.get_exploration_ratio()
                
                # 記錄重疊率（注意：這行非常重要，確保它正確執行）
                self.overlap_ratios.append(overlap_ratio)
                
                # 停止追蹤
                self.map_tracker.stop_tracking()
                
                # Episode結束後的處理
                exploration_progress = self.robot1.get_exploration_progress()
                self.training_history['episode_rewards'].append(total_reward)
                self.training_history['robot1_rewards'].append(robot1_total_reward)
                self.training_history['robot2_rewards'].append(robot2_total_reward)
                self.training_history['episode_lengths'].append(steps)
                self.training_history['exploration_rates'].append(self.epsilon)
                self.training_history['losses'].append(
                    np.mean(episode_losses) if episode_losses else 0
                )
                self.training_history['exploration_progress'].append(exploration_progress)
                
                # 保存模型
                if (episode + 1) % save_freq == 0:
                    self.save_checkpoint(episode + 1)
                    self.plot_training_progress()
                    
                    # 儲存覆蓋率隨時間變化的圖表
                    if hasattr(self.map_tracker, 'plot_coverage_over_time'):
                        self.map_tracker.plot_coverage_over_time()
                
                # 更新探索率
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                # 列印訓練信息
                print(f"\n第 {episode + 1}/{episodes} 輪 (地圖 {self.robot1.li_map})")
                print(f"步數: {steps}, 總獎勵: {total_reward:.2f}")
                print(f"Robot1 獎勵: {robot1_total_reward:.2f}")
                print(f"Robot2 獎勵: {robot2_total_reward:.2f}")
                print(f"探索率: {self.epsilon:.3f}")
                print(f"平均損失: {self.training_history['losses'][-1]:.6f}")

                # 添加Actor和Critic損失輸出
                if 'robot1_policy_loss' in self.training_history and len(self.training_history['robot1_policy_loss']) > 0:
                    avg_actor_loss = (self.training_history['robot1_policy_loss'][-1] + 
                                    self.training_history['robot2_policy_loss'][-1]) / 2
                    avg_critic_loss = (self.training_history['robot1_value_loss'][-1] + 
                                    self.training_history['robot2_value_loss'][-1]) / 2
                    print(f"Actor Loss: {avg_actor_loss:.6f}")
                    print(f"Critic Loss: {avg_critic_loss:.6f}")

                print(f"探索進度: {exploration_progress:.1%}")

                # 印出機器人探索重疊信息
                print(f"Robot1 探索覆蓋率: {robot1_ratio:.2%}")
                print(f"Robot2 探索覆蓋率: {robot2_ratio:.2%}")
                print(f"機器人local map區域交集: {overlap_ratio:.2%}")
                
                if exploration_progress >= self.robot1.finish_percent:
                    print("地圖探索完成！")
                else:
                    print("地圖探索未完成")
                print("-" * 50)
                
                # 重置環境
                state = self.robot1.reset()
                self.robot2.reset()
            
            # 訓練結束後保存最終模型
            self.save_checkpoint(episodes)
            
        except Exception as e:
            print(f"訓練過程出現錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 確保清理資源
            if hasattr(self.robot1, 'cleanup_visualization'):
                self.robot1.cleanup_visualization()
            if hasattr(self.robot2, 'cleanup_visualization'):
                self.robot2.cleanup_visualization()
            self.map_tracker.cleanup()