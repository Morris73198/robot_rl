import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from two_robot_a2c.config import MODEL_DIR, ROBOT_CONFIG, REWARD_CONFIG
from two_robot_a2c.environment.robot_local_map_tracker import RobotIndividualMapTracker

class MultiRobotA2CTrainer:
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
        """基於策略網絡的純粹動作選擇，減少外部干擾"""
        if len(frontiers) == 0:
            return 0, 0, 0.0, 0.0, np.zeros(self.model.max_frontiers), np.zeros(self.model.max_frontiers)
        
        # 準備輸入
        state_batch = np.expand_dims(state, 0).astype(np.float32)
        frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0).astype(np.float32)
        robot1_pos_batch = np.expand_dims(robot1_pos, 0).astype(np.float32)
        robot2_pos_batch = np.expand_dims(robot2_pos, 0).astype(np.float32)
        robot1_target_batch = np.expand_dims(robot1_target, 0).astype(np.float32)
        robot2_target_batch = np.expand_dims(robot2_target, 0).astype(np.float32)
        
        try:
            # 獲取模型預測
            predictions = self.model.predict(
                state_batch, 
                frontiers_batch, 
                robot1_pos_batch, 
                robot2_pos_batch,
                robot1_target_batch, 
                robot2_target_batch
            )
            
            # 提取策略和價值
            robot1_policy = predictions['robot1_policy'][0]
            robot2_policy = predictions['robot2_policy'][0]
            robot1_value = float(predictions['robot1_value'][0][0])
            robot2_value = float(predictions['robot2_value'][0][0])
            robot1_logits = predictions['robot1_logits'][0]
            robot2_logits = predictions['robot2_logits'][0]
            
            # 確定有效前沿點數量
            valid_frontiers = min(self.model.max_frontiers, len(frontiers))
            
            # 只考慮有效的前沿點策略
            robot1_probs = robot1_policy[:valid_frontiers]
            robot2_probs = robot2_policy[:valid_frontiers]
            
            # 數值穩定性處理
            robot1_probs = np.nan_to_num(robot1_probs, nan=1.0/valid_frontiers)
            robot2_probs = np.nan_to_num(robot2_probs, nan=1.0/valid_frontiers)
            
            robot1_probs = np.maximum(robot1_probs, 1e-8)
            robot2_probs = np.maximum(robot2_probs, 1e-8)
            
            # 重新歸一化
            robot1_probs = robot1_probs / np.sum(robot1_probs)
            robot2_probs = robot2_probs / np.sum(robot2_probs)
            
            # 直接根據策略分佈選擇動作，無額外干擾
            try:
                # Robot1 直接按策略選擇
                robot1_action = np.random.choice(valid_frontiers, p=robot1_probs)
                robot1_target_point = frontiers[robot1_action]
                
                # Robot2 也按原始策略選擇
                robot2_action = np.random.choice(valid_frontiers, p=robot2_probs)
                
                # 唯一的協調邏輯：避免完全相同的目標點
                # 這是必要的安全措施，不算外部干擾
                if robot1_action == robot2_action:
                    # 如果兩個機器人選擇相同的點，讓Robot2重新選擇
                    # 排除已選的點，從剩餘策略中重新抽樣
                    remaining_probs = robot2_probs.copy()
                    remaining_probs[robot1_action] = 0
                    
                    # 如果還有其他有效選項
                    if np.sum(remaining_probs) > 0:
                        # 重新歸一化
                        remaining_probs = remaining_probs / np.sum(remaining_probs)
                        # 重新選擇
                        robot2_action = np.random.choice(valid_frontiers, p=remaining_probs)
                    else:
                        # 如果沒有其他選項，隨機選擇不同點
                        other_indices = [i for i in range(valid_frontiers) if i != robot1_action]
                        if other_indices:
                            robot2_action = np.random.choice(other_indices)
                
                return robot1_action, robot2_action, robot1_value, robot2_value, robot1_logits, robot2_logits
                
            except ValueError as e:
                print(f"動作選擇錯誤: {str(e)}")
                # 發生錯誤時的備用方案 - 選擇最高概率動作
                robot1_action = np.argmax(robot1_probs)
                robot2_action = np.argmax(robot2_probs)
                # 確保不選相同的點
                if robot1_action == robot2_action and valid_frontiers > 1:
                    # 找出Robot2的第二高概率動作
                    second_best = np.argsort(robot2_probs)[-2]
                    robot2_action = second_best
                
                return robot1_action, robot2_action, robot1_value, robot2_value, robot1_logits, robot2_logits
                    
        except Exception as e:
            print(f"模型預測錯誤: {str(e)}")
            # 發生嚴重錯誤時的備用策略 - 完全隨機選擇
            valid_frontiers = min(self.model.max_frontiers, len(frontiers))
            robot1_action = np.random.randint(valid_frontiers)
            # 避免選擇相同的點
            other_indices = [i for i in range(valid_frontiers) if i != robot1_action]
            if other_indices and valid_frontiers > 1:
                robot2_action = np.random.choice(other_indices)
            else:
                robot2_action = np.random.randint(valid_frontiers)
            
            return robot1_action, robot2_action, 0.0, 0.0, np.zeros(self.model.max_frontiers), np.zeros(self.model.max_frontiers)






    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        """改進的回報和優勢計算，加強學習信號"""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0
        
        # 增加獎勵尺度來放大學習信號
        reward_scale = 1.2
        scaled_rewards = rewards * reward_scale
        
        # GAE參數可調整
        gamma = self.gamma  # 保持原有折扣係數
        lambda_param = 0.97  # 略微提高lambda值以增強長期獎勵傳播
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_val = values[t+1]
                
            # 計算TD誤差，使用尺度調整的獎勵
            delta = scaled_rewards[t] + gamma * next_val * next_non_terminal - values[t]
            
            # 計算GAE
            gae = delta + gamma * lambda_param * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # 優化優勢函數標準化，確保足夠的變異性
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        
        # 確保標準差不會太小，防止信號過弱
        min_std = 0.1
        if adv_std < min_std:
            # 如果標準差過小，增加變異性
            noise_scale = 0.01
            # 添加小幅噪聲來增加變異性
            advantages = advantages + np.random.normal(0, noise_scale, size=advantages.shape)
            # 重新計算統計量
            adv_mean = np.mean(advantages)
            adv_std = max(np.std(advantages), min_std)
        
        # 標準化，但確保有足夠變異性
        normalized_advantages = (advantages - adv_mean) / adv_std
        
        # 適當裁剪以防止極端值，但保留較大範圍
        clipped_advantages = np.clip(normalized_advantages, -3.0, 3.0)
        
        return returns, clipped_advantages

    
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
                
                # 計算重疊區域（確保此處代碼在訓練循環外，一個回合結束時執行）
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
                # print(f"探索率: {self.epsilon:.3f}")
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
    
    def plot_training_progress(self):
        """繪製訓練進度圖"""
        # 添加調試信息
        print(f"繪圖信息 - 重疊率數據長度: {len(self.overlap_ratios)}")
        print(f"繪圖信息 - 重疊率數據前5個元素: {self.overlap_ratios[:5] if len(self.overlap_ratios) >= 5 else self.overlap_ratios}")

        # 檢查是否有重疊率數據
        has_overlap_data = hasattr(self, 'overlap_ratios') and len(self.overlap_ratios) > 0
        # n_plots = 7 + int(has_overlap_data)
        n_plots = 7


        fig, axs = plt.subplots(n_plots, 1, figsize=(12, n_plots * 3.5))
        if n_plots == 1:
            axs = [axs]

        # 確保所有資料的長度一致
        data_length = min(
            len(self.training_history['episode_rewards']),
            len(self.training_history['robot1_rewards']),
            len(self.training_history['robot2_rewards']),
            len(self.training_history['episode_lengths']),
            len(self.training_history['losses']),
            len(self.training_history['exploration_progress'])
        )

        episodes = range(1, data_length + 1)

        # 裁剪所有資料到相同長度
        episode_rewards = self.training_history['episode_rewards'][:data_length]
        robot1_rewards = self.training_history['robot1_rewards'][:data_length]
        robot2_rewards = self.training_history['robot2_rewards'][:data_length]
        episode_lengths = self.training_history['episode_lengths'][:data_length]
        losses = self.training_history['losses'][:data_length]
        exploration_progress = self.training_history['exploration_progress'][:data_length]

        # 繪製總獎勵
        axs[0].plot(episodes, episode_rewards, color='#2E8B57')
        axs[0].set_title('Total Reward')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        axs[0].grid(True)

        # 繪製各機器人獎勵
        axs[1].plot(episodes, robot1_rewards, color='#8A2BE2', label='Robot1')
        axs[1].plot(episodes, robot2_rewards, color='#FFA500', label='Robot2')
        axs[1].set_title('Reward per Robot')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Reward')
        axs[1].legend()
        axs[1].grid(True)

        # 繪製步數
        axs[2].plot(episodes, episode_lengths, color='#4169E1')
        axs[2].set_title('Steps per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Steps')
        axs[2].grid(True)

        # 繪製Actor損失（若有）
        if 'robot1_policy_loss' in self.training_history and len(self.training_history['robot1_policy_loss']) >= data_length:
            robot1_policy_loss = self.training_history['robot1_policy_loss'][:data_length]
            robot2_policy_loss = self.training_history['robot2_policy_loss'][:data_length]
            actor_losses = [(r1 + r2) / 2 for r1, r2 in zip(robot1_policy_loss, robot2_policy_loss)]
            axs[3].plot(episodes, actor_losses, color='#DC143C')
            axs[3].set_title('Actor Loss')
        else:
            axs[3].plot(episodes, losses, color='#DC143C')
            axs[3].set_title('Training Loss')
        axs[3].set_xlabel('Episode')
        axs[3].set_ylabel('Loss')
        axs[3].grid(True)

        # 繪製Critic損失（若有）
        if 'robot1_value_loss' in self.training_history and len(self.training_history['robot1_value_loss']) >= data_length:
            robot1_value_loss = self.training_history['robot1_value_loss'][:data_length]
            robot2_value_loss = self.training_history['robot2_value_loss'][:data_length]
            critic_losses = [(r1 + r2) / 2 for r1, r2 in zip(robot1_value_loss, robot2_value_loss)]
            axs[4].plot(episodes, critic_losses, color='#2F4F4F')
            axs[4].set_title('Critic Loss')
        else:
            axs[4].plot(episodes, [0] * data_length, color='#2F4F4F')
            axs[4].set_title('Critic Loss (Not Available)')
        axs[4].set_xlabel('Episode')
        axs[4].set_ylabel('Loss')
        axs[4].grid(True)

        # 繪製探索進度
        axs[5].plot(episodes, exploration_progress, color='#2F4F4F')
        axs[5].set_title('Exploration Progress')
        axs[5].set_xlabel('Episode')
        axs[5].set_ylabel('Completion Rate')
        axs[5].grid(True)

        # 繪製重疊率（如有）
        if has_overlap_data:
            overlap_data = self.overlap_ratios.copy()
            if len(overlap_data) < data_length:
                overlap_data += [0.0] * (data_length - len(overlap_data))
            elif len(overlap_data) > data_length:
                overlap_data = overlap_data[:data_length]
            axs[6].plot(episodes, overlap_data, color='#8B008B')
            axs[6].set_title('Map Overlap Ratio')
            axs[6].set_xlabel('Episode')
            axs[6].set_ylabel('Overlap Ratio')
            axs[6].grid(True)
            axs[6].set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

        # 額外繪製兩機器人獎勵對比圖
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, robot1_rewards, color='#8A2BE2', label='Robot1', alpha=0.7)
        plt.plot(episodes, robot2_rewards, color='#FFA500', label='Robot2', alpha=0.7)
        plt.fill_between(episodes, robot1_rewards, alpha=0.3, color='#9370DB')
        plt.fill_between(episodes, robot2_rewards, alpha=0.3, color='#FFB84D')
        plt.title('Robot Rewards Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig('robots_rewards_comparison.png')
        plt.close()

        # 額外繪製重疊率圖（如有）
        if has_overlap_data:
            overlap_data = self.overlap_ratios.copy()
            if len(overlap_data) < data_length:
                overlap_data += [0.0] * (data_length - len(overlap_data))
            elif len(overlap_data) > data_length:
                overlap_data = overlap_data[:data_length]
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

    
    def save_training_history(self, filename='a2c_training_history.npz'):
        """保存訓練歷史"""
        np.savez(
            filename,
            episode_rewards=self.training_history['episode_rewards'],
            robot1_rewards=self.training_history['robot1_rewards'],
            robot2_rewards=self.training_history['robot2_rewards'],
            episode_lengths=self.training_history['episode_lengths'],
            exploration_rates=self.training_history['exploration_rates'],
            losses=self.training_history['losses'],
            exploration_progress=self.training_history['exploration_progress'],
            robot1_entropy=self.training_history.get('robot1_entropy', []),
            robot2_entropy=self.training_history.get('robot2_entropy', []),
            robot1_value_loss=self.training_history.get('robot1_value_loss', []),
            robot2_value_loss=self.training_history.get('robot2_value_loss', []),
            robot1_policy_loss=self.training_history.get('robot1_policy_loss', []),
            robot2_policy_loss=self.training_history.get('robot2_policy_loss', []),
            overlap_ratios=self.overlap_ratios
        )
    
    # 修改載入訓練歷史的方法
    def load_training_history(self, filename='a2c_training_history.npz'):
        """載入訓練歷史"""
        data = np.load(filename, allow_pickle=True)
        self.training_history = {
            'episode_rewards': data['episode_rewards'].tolist(),
            'robot1_rewards': data['robot1_rewards'].tolist(),
            'robot2_rewards': data['robot2_rewards'].tolist(),
            'episode_lengths': data['episode_lengths'].tolist(),
            'exploration_rates': data['exploration_rates'].tolist(),
            'losses': data['losses'].tolist(),
            'exploration_progress': data['exploration_progress'].tolist(),
        }
        
        # 載入可選項
        if 'robot1_entropy' in data:
            self.training_history['robot1_entropy'] = data['robot1_entropy'].tolist()
            self.training_history['robot2_entropy'] = data['robot2_entropy'].tolist()
            self.training_history['robot1_value_loss'] = data['robot1_value_loss'].tolist()
            self.training_history['robot2_value_loss'] = data['robot2_value_loss'].tolist()
            self.training_history['robot1_policy_loss'] = data['robot1_policy_loss'].tolist()
            self.training_history['robot2_policy_loss'] = data['robot2_policy_loss'].tolist()
            
        # 載入重疊率
        if 'overlap_ratios' in data:
            self.overlap_ratios = data['overlap_ratios'].tolist()
        else:
            self.overlap_ratios = []
            
    # 在保存檢查點方法中添加重疊率
    def save_checkpoint(self, episode):
        """保存檢查點
        
        Args:
            episode: 當前訓練輪數
        """
        # 用零填充確保文件名排序正確
        ep_str = str(episode).zfill(6)
        
        # 保存模型
        model_path = os.path.join(MODEL_DIR, f'multi_robot_model_a2c_ep{ep_str}.h5')
        self.model.save(model_path)
        
        # 保存訓練歷史
        history_path = os.path.join(MODEL_DIR, f'multi_robot_a2c_history_ep{ep_str}.json')
        history_to_save = {
            'episode_rewards': [float(x) for x in self.training_history['episode_rewards']],
            'robot1_rewards': [float(x) for x in self.training_history['robot1_rewards']],
            'robot2_rewards': [float(x) for x in self.training_history['robot2_rewards']],
            'episode_lengths': [int(x) for x in self.training_history['episode_lengths']],
            'exploration_rates': [float(x) for x in self.training_history['exploration_rates']],
            'losses': [float(x) if x is not None else 0.0 for x in self.training_history['losses']],
            'exploration_progress': [float(x) for x in self.training_history['exploration_progress']],
            'overlap_ratios': [float(x) for x in self.overlap_ratios]
        }
        
        # 添加可選歷史項目
        if 'robot1_entropy' in self.training_history and self.training_history['robot1_entropy']:
            history_to_save['robot1_entropy'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot1_entropy']]
            history_to_save['robot2_entropy'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot2_entropy']]
            history_to_save['robot1_value_loss'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot1_value_loss']]
            history_to_save['robot2_value_loss'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot2_value_loss']]
            history_to_save['robot1_policy_loss'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot1_policy_loss']]
            history_to_save['robot2_policy_loss'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot2_policy_loss']]
        
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=4)
        
        print(f"已在第 {episode} 輪保存檢查點")