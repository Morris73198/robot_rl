import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from two_robot_a2c_enhance.config import MODEL_DIR, ROBOT_CONFIG, REWARD_CONFIG
from two_robot_a2c_enhance.environment.robot_local_map_tracker import RobotIndividualMapTracker

class EnhancedMultiRobotA2CTrainer:
    def __init__(self, model, robot1, robot2, memory_size=10000, batch_size=16, gamma=0.99):
        """Initialize the multi-robot A2C trainer with the new model architecture
        
        Args:
            model: A2C model with the new architecture from images
            robot1: First robot instance
            robot2: Second robot instance
            memory_size: Experience replay buffer size
            batch_size: Batch size
            gamma: Discount factor
        """
        self.model = model
        self.robot1 = robot1
        self.robot2 = robot2
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = 0.95
        
        self.map_size = self.robot1.map_size
        
        # Training parameters
        self.epsilon = 1.0  # Exploration rate (used for action selection)
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration rate decay

        # Track overlap ratios
        self.overlap_ratios = []
        
        # Training history
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
        
        # Create robot individual map tracker
        self.map_tracker = RobotIndividualMapTracker(robot1, robot2)
        
        # Parameters for advantage function normalization
        self.advantage_epsilon = 1e-8
        
        # Trajectory buffer
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
        """Reset the trajectory buffer"""
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
        """Add experience to trajectory buffer"""
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
        """Pad frontier points to fixed length and normalize coordinates"""
        padded = np.zeros((self.model.max_frontiers, 2))
        
        if len(frontiers) > 0:
            frontiers = np.array(frontiers)
            
            # Normalize coordinates
            normalized_frontiers = frontiers.copy()
            normalized_frontiers[:, 0] = frontiers[:, 0] / float(self.map_size[1])
            normalized_frontiers[:, 1] = frontiers[:, 1] / float(self.map_size[0])
            
            n_frontiers = min(len(frontiers), self.model.max_frontiers)
            padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
        
        return padded
    
    def get_normalized_target(self, target):
        """Normalize target position"""
        if target is None:
            return np.array([0.0, 0.0])  # Return origin if no target
        normalized = np.array([
            target[0] / float(self.map_size[1]),
            target[1] / float(self.map_size[0])
        ])
        return normalized
    
    # def choose_actions(self, state, frontiers, robot1_pos, robot2_pos, 
    #                 robot1_target, robot2_target):
    #     """Choose actions based on policy entropy and temperature sampling"""
    #     if len(frontiers) == 0:
    #         return 0, 0, 0.0, 0.0, np.zeros(self.model.max_frontiers), np.zeros(self.model.max_frontiers)
        
    #     # Prepare inputs
    #     state_batch = np.expand_dims(state, 0).astype(np.float32)
    #     frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0).astype(np.float32)
    #     robot1_pos_batch = np.expand_dims(robot1_pos, 0).astype(np.float32)
    #     robot2_pos_batch = np.expand_dims(robot2_pos, 0).astype(np.float32)
    #     robot1_target_batch = np.expand_dims(robot1_target, 0).astype(np.float32)
    #     robot2_target_batch = np.expand_dims(robot2_target, 0).astype(np.float32)
        
    #     # Determine valid frontier count
    #     valid_frontiers = min(self.model.max_frontiers, len(frontiers))
        
    #     # Get model predictions
    #     try:
    #         predictions = self.model.predict(
    #             state_batch, 
    #             frontiers_batch, 
    #             robot1_pos_batch, 
    #             robot2_pos_batch,
    #             robot1_target_batch, 
    #             robot2_target_batch
    #         )
            
    #         robot1_policy = predictions['robot1_policy'][0]
    #         robot2_policy = predictions['robot2_policy'][0]
    #         robot1_value = float(predictions['robot1_value'][0][0])
    #         robot2_value = float(predictions['robot2_value'][0][0])
    #         robot1_logits = predictions['robot1_logits'][0]
    #         robot2_logits = predictions['robot2_logits'][0]
    #     except Exception as e:
    #         print(f"Model prediction error: {str(e)}")
    #         # Use uniform distribution if prediction fails
    #         robot1_policy = np.ones(self.model.max_frontiers) / self.model.max_frontiers
    #         robot2_policy = np.ones(self.model.max_frontiers) / self.model.max_frontiers
    #         robot1_value = 0.0
    #         robot2_value = 0.0
    #         robot1_logits = np.zeros(self.model.max_frontiers)
    #         robot2_logits = np.zeros(self.model.max_frontiers)
        
    #     # Consider only valid frontiers
    #     robot1_probs = robot1_policy[:valid_frontiers].copy()
    #     robot2_probs = robot2_policy[:valid_frontiers].copy()
        
    #     # Handle numerical issues
    #     robot1_probs = np.nan_to_num(robot1_probs, nan=1.0/valid_frontiers)
    #     robot2_probs = np.nan_to_num(robot2_probs, nan=1.0/valid_frontiers)
        
    #     # Ensure probabilities sum to 1
    #     robot1_sum = np.sum(robot1_probs)
    #     robot2_sum = np.sum(robot2_probs)
        
    #     if robot1_sum > 0:
    #         robot1_probs = robot1_probs / robot1_sum
    #     else:
    #         robot1_probs = np.ones(valid_frontiers) / valid_frontiers
            
    #     if robot2_sum > 0:
    #         robot2_probs = robot2_probs / robot2_sum
    #     else:
    #         robot2_probs = np.ones(valid_frontiers) / valid_frontiers
        
    #     # Calculate policy entropy for dynamic temperature adjustment
    #     def calculate_entropy(probs):
    #         # Avoid log(0)
    #         log_probs = np.log(probs + 1e-10)
    #         entropy = -np.sum(probs * log_probs)
    #         # Normalize entropy to 0-1 range
    #         max_entropy = np.log(valid_frontiers)
    #         normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    #         return normalized_entropy
        
    #     robot1_entropy = calculate_entropy(robot1_probs)
    #     robot2_entropy = calculate_entropy(robot2_probs)
        
    #     # Dynamically adjust temperature based on training progress and entropy
    #     base_temperature = self.get_dynamic_temperature()
        
    #     # Adjust based on entropy - increase temperature when entropy is low
    #     robot1_temp = base_temperature * (1.0 + max(0, 0.5 - robot1_entropy) * 2)
    #     robot2_temp = base_temperature * (1.0 + max(0, 0.5 - robot2_entropy) * 2)
        
    #     # Apply temperature to reshape probability distribution
    #     def apply_temperature(probs, temperature):
    #         # Higher temperature = flatter distribution (more exploration)
    #         # Lower temperature = sharper distribution (more exploitation)
    #         if temperature != 1.0:
    #             logits = np.log(probs + 1e-10)
    #             logits = logits / temperature
    #             # Prevent overflow
    #             logits = logits - np.max(logits)
    #             exp_logits = np.exp(logits)
    #             probs = exp_logits / np.sum(exp_logits)
    #         return probs
        
    #     # Apply temperature
    #     robot1_probs = apply_temperature(robot1_probs, robot1_temp)
    #     robot2_probs = apply_temperature(robot2_probs, robot2_temp)
        
    #     # Avoid two robots choosing the same or nearby frontiers
    #     # Let robot1 choose first
    #     try:
    #         robot1_action = np.random.choice(valid_frontiers, p=robot1_probs)
    #     except ValueError as e:
    #         print(f"Robot1 sampling error: {str(e)}")
    #         print(f"Robot1 probabilities: {robot1_probs}, sum: {np.sum(robot1_probs)}")
    #         robot1_action = np.random.randint(valid_frontiers)
        
    #     # Then modify robot2's probabilities to reduce chance of choosing the same or nearby frontier
    #     robot2_adjusted_probs = robot2_probs.copy()
        
    #     # Define proximity threshold
    #     min_target_distance = self.robot1.sensor_range * 1.5
        
    #     # Adjust robot2's probabilities
    #     for i in range(valid_frontiers):
    #         if i == robot1_action:
    #             # Significantly reduce probability of choosing the same point
    #             robot2_adjusted_probs[i] *= 0.2
    #         else:
    #             # Calculate distance to robot1's chosen frontier
    #             dist = np.linalg.norm(frontiers[i] - frontiers[robot1_action])
    #             if dist < min_target_distance:
    #                 # Reduce probability more for closer points
    #                 reduction_factor = dist / min_target_distance
    #                 robot2_adjusted_probs[i] *= max(0.2, reduction_factor)
        
    #     # Renormalize
    #     robot2_sum = np.sum(robot2_adjusted_probs)
    #     if robot2_sum > 0:
    #         robot2_adjusted_probs = robot2_adjusted_probs / robot2_sum
    #     else:
    #         # Use uniform distribution if all probabilities are reduced to 0
    #         robot2_adjusted_probs = np.ones(valid_frontiers) / valid_frontiers
        
    #     # Choose action for robot2
    #     try:
    #         robot2_action = np.random.choice(valid_frontiers, p=robot2_adjusted_probs)
    #     except ValueError as e:
    #         print(f"Robot2 sampling error: {str(e)}")
    #         print(f"Robot2 adjusted probabilities: {robot2_adjusted_probs}, sum: {np.sum(robot2_adjusted_probs)}")
    #         # Choose a different point than robot1 if possible
    #         other_indices = [i for i in range(valid_frontiers) if i != robot1_action]
    #         if other_indices:
    #             robot2_action = np.random.choice(other_indices)
    #         else:
    #             robot2_action = np.random.randint(valid_frontiers)
        
    #     return robot1_action, robot2_action, robot1_value, robot2_value, robot1_logits, robot2_logits







    def choose_actions(self, state, frontiers, robot1_pos, robot2_pos, 
                        robot1_target, robot2_target):
        """選擇動作，使用ε-greedy策略增加探索"""
        if len(frontiers) == 0:
            return 0, 0, 0.0, 0.0, np.zeros(self.model.max_frontiers), np.zeros(self.model.max_frontiers)
        
        # 獲取有效frontier數量
        valid_frontiers = min(self.model.max_frontiers, len(frontiers))
        
        # ε-greedy策略：以epsilon的概率隨機探索，以1-epsilon的概率使用模型預測
        if np.random.rand() < self.epsilon:
            # 隨機探索 - 讓兩個機器人各自隨機選擇
            robot1_action = np.random.randint(valid_frontiers)
            robot2_action = np.random.randint(valid_frontiers)
            
            # 初始化默認價值和logits
            robot1_value = 0.0
            robot2_value = 0.0
            robot1_logits = np.zeros(self.model.max_frontiers)
            robot2_logits = np.zeros(self.model.max_frontiers)
            
            # 可以選擇性調用模型獲取價值估計，但不使用其策略選擇
            try:
                # 準備輸入
                state_batch = np.expand_dims(state, 0).astype(np.float32)
                frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0).astype(np.float32)
                robot1_pos_batch = np.expand_dims(robot1_pos, 0).astype(np.float32)
                robot2_pos_batch = np.expand_dims(robot2_pos, 0).astype(np.float32)
                robot1_target_batch = np.expand_dims(robot1_target, 0).astype(np.float32)
                robot2_target_batch = np.expand_dims(robot2_target, 0).astype(np.float32)
                
                # 獲取模型預測，僅用於價值估計
                predictions = self.model.predict(
                    state_batch, 
                    frontiers_batch, 
                    robot1_pos_batch, 
                    robot2_pos_batch,
                    robot1_target_batch, 
                    robot2_target_batch
                )
                
                robot1_value = float(predictions['robot1_value'][0][0])
                robot2_value = float(predictions['robot2_value'][0][0])
                robot1_logits = predictions['robot1_logits'][0]
                robot2_logits = predictions['robot2_logits'][0]
            except Exception as e:
                print(f"模型預測錯誤 (探索模式): {str(e)}")
                # 使用默認值
                pass
        else:
            # 利用當前策略 - 使用模型預測
            try:
                # 準備輸入
                state_batch = np.expand_dims(state, 0).astype(np.float32)
                frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0).astype(np.float32)
                robot1_pos_batch = np.expand_dims(robot1_pos, 0).astype(np.float32)
                robot2_pos_batch = np.expand_dims(robot2_pos, 0).astype(np.float32)
                robot1_target_batch = np.expand_dims(robot1_target, 0).astype(np.float32)
                robot2_target_batch = np.expand_dims(robot2_target, 0).astype(np.float32)
                
                # 獲取模型預測
                predictions = self.model.predict(
                    state_batch, 
                    frontiers_batch, 
                    robot1_pos_batch, 
                    robot2_pos_batch,
                    robot1_target_batch, 
                    robot2_target_batch
                )
                
                # 提取預測結果
                robot1_policy = predictions['robot1_policy'][0]
                robot2_policy = predictions['robot2_policy'][0]
                robot1_value = float(predictions['robot1_value'][0][0])
                robot2_value = float(predictions['robot2_value'][0][0])
                robot1_logits = predictions['robot1_logits'][0]
                robot2_logits = predictions['robot2_logits'][0]
                
                # 處理有效的frontier機率
                robot1_probs = robot1_policy[:valid_frontiers].copy()
                robot2_probs = robot2_policy[:valid_frontiers].copy()
                
                # 處理數值問題
                robot1_probs = np.nan_to_num(robot1_probs, nan=1.0/valid_frontiers)
                robot2_probs = np.nan_to_num(robot2_probs, nan=1.0/valid_frontiers)
                
                # 確保機率和為1
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
                
                # 根據機率分布選擇動作
                try:
                    robot1_action = np.random.choice(valid_frontiers, p=robot1_probs)
                except ValueError as e:
                    print(f"Robot1 抽樣錯誤: {str(e)}")
                    print(f"Robot1 機率: {robot1_probs}, 和: {np.sum(robot1_probs)}")
                    robot1_action = np.random.randint(valid_frontiers)
                
                try:
                    robot2_action = np.random.choice(valid_frontiers, p=robot2_probs)
                except ValueError as e:
                    print(f"Robot2 抽樣錯誤: {str(e)}")
                    print(f"Robot2 機率: {robot2_probs}, 和: {np.sum(robot2_probs)}")
                    robot2_action = np.random.randint(valid_frontiers)
                
            except Exception as e:
                print(f"模型預測錯誤 (利用模式): {str(e)}")
                # 預測失敗時使用隨機策略
                robot1_action = np.random.randint(valid_frontiers)
                robot2_action = np.random.randint(valid_frontiers)
                robot1_value = 0.0
                robot2_value = 0.0
                robot1_logits = np.zeros(self.model.max_frontiers)
                robot2_logits = np.zeros(self.model.max_frontiers)
        
        # 動態減少epsilon，鼓勵逐漸從探索轉向利用
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return robot1_action, robot2_action, robot1_value, robot2_value, robot1_logits, robot2_logits

    def get_dynamic_temperature(self):
        """Dynamically adjust temperature parameter based on training progress"""
        # Estimate current training progress
        if len(self.training_history['episode_rewards']) < 10:
            # Early training, use high temperature to promote exploration
            return 2.0
        
        # Calculate recent performance improvement
        recent_rewards = self.training_history['episode_rewards'][-10:]
        recent_avg = np.mean(recent_rewards)
        
        if len(self.training_history['episode_rewards']) >= 20:
            previous_rewards = self.training_history['episode_rewards'][-20:-10]
            previous_avg = np.mean(previous_rewards)
            
            # If performance is improving, reduce temperature (more exploitation)
            # If performance is declining, increase temperature (more exploration)
            improvement = (recent_avg - previous_avg) / (abs(previous_avg) + 1e-8)
            
            if improvement > 0.1:  # Significant improvement
                temperature = 0.8  # Lower temperature, increase exploitation
            elif improvement < -0.05:  # Performance decline
                temperature = 1.5  # Higher temperature, increase exploration
            else:  # Stable performance
                temperature = 1.0
        else:
            # Insufficient data, use moderate temperature
            temperature = 1.0
        
        # Periodically increase temperature if long-term stagnation detected
        episodes = len(self.training_history['episode_rewards'])
        if episodes % 100 == 0 and episodes > 0:
            # Compare last 100 episodes with previous 100
            if episodes >= 200:
                recent_100 = np.mean(self.training_history['episode_rewards'][-100:])
                previous_100 = np.mean(self.training_history['episode_rewards'][-200:-100])
                
                # If recent performance hasn't significantly improved, increase temperature
                if recent_100 <= previous_100 * 1.05:
                    temperature = max(2.0, temperature * 1.5)
                    print(f"Training stagnation detected, increasing temperature to {temperature:.2f} to enhance exploration")
        
        # Limit temperature range to avoid extreme values
        temperature = np.clip(temperature, 0.5, 3.0)
        
        return temperature

    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        """Compute returns and advantages using GAE (Generalized Advantage Estimation)"""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        
        # Keep rewards scale reasonable
        reward_scale = 1.0
        scaled_rewards = rewards * reward_scale
        
        # GAE parameters
        gamma = 0.99  # Discount factor
        lambda_param = 0.95  # GAE lambda parameter
        
        # Compute backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_val = values[t+1]
            
            # Calculate TD error
            delta = scaled_rewards[t] + gamma * next_val * next_non_terminal - values[t]
            
            # Calculate GAE
            gae = delta + gamma * lambda_param * next_non_terminal * gae
            
            # Store advantage and return
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Robust normalization
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        
        # Normalize advantages to mean 0, std 1
        normalized_advantages = (advantages - adv_mean) / adv_std
        
        # Clip range to avoid extreme values
        clip_range = 3.0  # Wider range to preserve some extreme signals
        clipped_advantages = np.clip(normalized_advantages, -clip_range, clip_range)
        
        # Maintain moderate advantage strength
        scale_factor = 1.0
        final_advantages = clipped_advantages * scale_factor
        
        return returns, final_advantages
    
    def process_trajectory(self, last_robot1_value=0, last_robot2_value=0):
        """Process trajectory data, compute returns and advantages, and train the model"""
        if len(self.trajectory_buffer['states']) == 0:
            return {}
        
        # Extract data
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
        
        # Compute returns and advantages
        robot1_returns, robot1_advantages = self.compute_returns_and_advantages(
            robot1_rewards, robot1_values, dones, last_robot1_value
        )
        
        robot2_returns, robot2_advantages = self.compute_returns_and_advantages(
            robot2_rewards, robot2_values, dones, last_robot2_value
        )
        
        # Train model
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
            robot2_old_logits=robot2_logits,
            training_history=self.training_history
        )
        
        return loss_metrics
    
    def train(self, episodes=1000000, save_freq=10):
        """Execute multi-robot collaborative training"""
        try:
            for episode in range(episodes):
                # Initialize environment and state
                state = self.robot1.begin()
                self.robot2.begin()
                
                # Start map tracker
                self.map_tracker.start_tracking()
                
                # Initialize episode statistics
                total_reward = 0
                robot1_total_reward = 0
                robot2_total_reward = 0
                steps = 0
                episode_losses = []
                
                # Define minimum target distance outside loop to avoid undefined error
                MIN_TARGET_DISTANCE = self.robot1.sensor_range * 1.5
                
                # Reset trajectory buffer
                self.reset_trajectory_buffer()
                
                # Training loop
                while not (self.robot1.check_done() or self.robot2.check_done() or steps >= 1500):
                    frontiers = self.robot1.get_frontiers()
                    if len(frontiers) == 0:
                        break
                        
                    # Get current state
                    robot1_pos = self.robot1.get_normalized_position()
                    robot2_pos = self.robot2.get_normalized_position()
                    old_robot1_pos = self.robot1.robot_position.copy()
                    old_robot2_pos = self.robot2.robot_position.copy()
                    
                    # Normalize target position
                    map_dims = np.array([float(self.robot1.map_size[1]), float(self.robot1.map_size[0])])
                    robot1_target = (np.zeros(2) if self.robot1.current_target_frontier is None 
                                else self.robot1.current_target_frontier / map_dims)
                    robot2_target = (np.zeros(2) if self.robot2.current_target_frontier is None 
                                else self.robot2.current_target_frontier / map_dims)
                    
                    # Choose actions
                    robot1_action, robot2_action, robot1_value, robot2_value, robot1_logits, robot2_logits = \
                        self.choose_actions(state, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target)
                    
                    # Set targets and execute moves
                    robot1_target_point = frontiers[robot1_action]
                    robot2_target_point = frontiers[robot2_action]
                    
                    # Move robot1
                    next_state1, r1, d1 = self.robot1.move_to_frontier(robot1_target_point)
                    robot1_reward = r1
                    # Update shared map
                    self.robot2.op_map = self.robot1.op_map.copy()
                    
                    # Check if robot2 needs to rechoose target (if robot1 chose nearby point)
                    if not d1 and self.robot2.current_target_frontier is not None:
                        distance_between_targets = np.linalg.norm(
                            self.robot2.current_target_frontier - robot1_target_point)
                        
                        if distance_between_targets < MIN_TARGET_DISTANCE:
                            # Rechoose if using model prediction mode
                            if np.random.random() >= self.epsilon:
                                # Get model predictions for coordination
                                predictions = self.model.predict(
                                    np.expand_dims(state, 0),
                                    np.expand_dims(self.pad_frontiers(frontiers), 0),
                                    np.expand_dims(robot1_pos, 0),
                                    np.expand_dims(robot2_pos, 0),
                                    np.expand_dims(robot1_target, 0),
                                    np.expand_dims(robot2_target, 0)
                                )
                                
                                # Extract policy
                                robot2_policy = predictions['robot2_policy'][0]
                                valid_frontiers = min(self.model.max_frontiers, len(frontiers))
                                robot2_policy = robot2_policy[:valid_frontiers]
                                
                                # Handle NaN and numerical stability
                                robot2_policy = np.nan_to_num(robot2_policy, nan=1.0/valid_frontiers)
                                robot2_policy = np.maximum(robot2_policy, 1e-10)
                                
                                robot2_sum = np.sum(robot2_policy)
                                if robot2_sum < 1e-8:
                                    robot2_policy = np.ones(valid_frontiers) / valid_frontiers
                                else:
                                    robot2_policy = robot2_policy / robot2_sum
                                
                                # Adjust policy to avoid robot1's target
                                for i in range(valid_frontiers):
                                    distance = np.linalg.norm(frontiers[i] - robot1_target_point)
                                    if distance < MIN_TARGET_DISTANCE:
                                        penalty = 1.0 - (distance / MIN_TARGET_DISTANCE)
                                        robot2_policy[i] *= (1.0 - penalty * 0.95)
                                
                                # Ensure policy is valid
                                robot2_policy = np.maximum(robot2_policy, 1e-10)
                                robot2_sum = np.sum(robot2_policy)
                                
                                if robot2_sum < 1e-8:
                                    # If all probabilities near zero, choose point far from robot1
                                    distances = np.linalg.norm(frontiers - robot1_target_point, axis=1)
                                    farthest_indices = np.argsort(distances)[-min(5, valid_frontiers):]
                                    robot2_action = farthest_indices[np.random.randint(len(farthest_indices))]
                                    robot2_target_point = frontiers[robot2_action]
                                else:
                                    # Renormalize and choose
                                    robot2_policy = robot2_policy / robot2_sum
                                    robot2_policy = np.nan_to_num(robot2_policy, nan=1.0/valid_frontiers)
                                    try:
                                        robot2_action = np.random.choice(valid_frontiers, p=robot2_policy)
                                        robot2_target_point = frontiers[robot2_action]
                                    except ValueError:
                                        # If error, choose farthest point
                                        distances = np.linalg.norm(frontiers - robot1_target_point, axis=1)
                                        robot2_action = np.argmax(distances[:valid_frontiers])
                                        robot2_target_point = frontiers[robot2_action]
                            else:
                                # In random mode, simply choose a point far from robot1's target
                                distances = np.linalg.norm(frontiers - robot1_target_point, axis=1)
                                farthest_indices = np.argsort(distances)[-min(5, len(frontiers)):]
                                robot2_action = farthest_indices[np.random.randint(len(farthest_indices))]
                                robot2_target_point = frontiers[robot2_action]
                    
                    # Move robot2
                    next_state2, r2, d2 = self.robot2.move_to_frontier(robot2_target_point)
                    robot2_reward = r2
                    # Update shared map
                    self.robot1.op_map = self.robot2.op_map.copy()
                    
                    # Update robot positions
                    self.robot1.other_robot_position = self.robot2.robot_position.copy()
                    self.robot2.other_robot_position = self.robot1.robot_position.copy()
                    
                    # Update map tracker
                    self.map_tracker.update()
                    
                    # Clip reward values for numerical stability
                    robot1_reward = np.clip(robot1_reward, -10, 10)
                    robot2_reward = np.clip(robot2_reward, -10, 10)
                    
                    # Determine if episode is done
                    done = d1 or d2
                    
                    # Add to trajectory buffer
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
                    
                    # Update state and statistics
                    state = next_state1
                    total_reward += (robot1_reward + robot2_reward)
                    robot1_total_reward += robot1_reward
                    robot2_total_reward += robot2_reward
                    steps += 1
                    
                    # Update visualization
                    if steps % ROBOT_CONFIG['plot_interval'] == 0:
                        if self.robot1.plot:
                            self.robot1.plot_env()
                        if self.robot2.plot:
                            self.robot2.plot_env()
                    
                    # Process trajectory and train if done or batch size reached
                    if done or steps % self.batch_size == 0:
                        # Get value estimates for the last state
                        if not done:
                            # Prepare inputs
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
                            
                            # Get value estimates for last state
                            predictions = self.model.predict(
                                state_batch, frontiers_batch, 
                                robot1_pos_batch, robot2_pos_batch,
                                robot1_target_batch, robot2_target_batch
                            )
                            
                            last_robot1_value = predictions['robot1_value'][0][0]
                            last_robot2_value = predictions['robot2_value'][0][0]
                        else:
                            # If episode ends, last state value is 0
                            last_robot1_value = 0
                            last_robot2_value = 0
                        
                        # Process trajectory to get training data
                        returns_metrics = self.process_trajectory(last_robot1_value, last_robot2_value)
                        
                        # PPO-style multiple updates
                        if len(self.trajectory_buffer['states']) >= self.batch_size:
                            # Get processed training data
                            train_states = np.array(self.trajectory_buffer['states'], dtype=np.float32)
                            train_frontiers = np.array(self.trajectory_buffer['frontiers'], dtype=np.float32)
                            train_robot1_pos = np.array(self.trajectory_buffer['robot1_pos'], dtype=np.float32)
                            train_robot2_pos = np.array(self.trajectory_buffer['robot2_pos'], dtype=np.float32)
                            train_robot1_target = np.array(self.trajectory_buffer['robot1_target'], dtype=np.float32)
                            train_robot2_target = np.array(self.trajectory_buffer['robot2_target'], dtype=np.float32)
                            train_robot1_actions = np.array(self.trajectory_buffer['robot1_actions'], dtype=np.int32)
                            train_robot2_actions = np.array(self.trajectory_buffer['robot2_actions'], dtype=np.int32)
                            
                            # Compute returns and advantages
                            robot1_rewards = np.array(self.trajectory_buffer['robot1_rewards'], dtype=np.float32)
                            robot2_rewards = np.array(self.trajectory_buffer['robot2_rewards'], dtype=np.float32)
                            robot1_values = np.array(self.trajectory_buffer['robot1_values'], dtype=np.float32)
                            robot2_values = np.array(self.trajectory_buffer['robot2_values'], dtype=np.float32)
                            dones = np.array(self.trajectory_buffer['dones'], dtype=np.float32)
                            
                            robot1_returns, robot1_advantages = self.compute_returns_and_advantages(
                                robot1_rewards, robot1_values, dones, last_robot1_value
                            )
                            
                            robot2_returns, robot2_advantages = self.compute_returns_and_advantages(
                                robot2_rewards, robot2_values, dones, last_robot2_value
                            )
                            
                            # Save old logits and policies
                            robot1_old_logits = np.array(self.trajectory_buffer['robot1_logits'], dtype=np.float32)
                            robot2_old_logits = np.array(self.trajectory_buffer['robot2_logits'], dtype=np.float32)
                            
                            # Multiple update loop (PPO-style)
                            n_updates = 4  # Number of updates per batch of data
                            ppo_losses = []

                            for update_idx in range(n_updates):
                                # Random sample mini-batch
                                batch_indices = np.random.permutation(len(train_states))[:min(32, len(train_states))]
                                
                                # Prepare mini-batch data
                                batch_states = train_states[batch_indices]
                                batch_frontiers = train_frontiers[batch_indices]
                                batch_robot1_pos = train_robot1_pos[batch_indices]
                                batch_robot2_pos = train_robot2_pos[batch_indices]
                                batch_robot1_target = train_robot1_target[batch_indices]
                                batch_robot2_target = train_robot2_target[batch_indices]
                                batch_robot1_actions = train_robot1_actions[batch_indices]
                                batch_robot2_actions = train_robot2_actions[batch_indices]
                                batch_robot1_advantages = robot1_advantages[batch_indices]
                                batch_robot2_advantages = robot2_advantages[batch_indices]
                                batch_robot1_returns = robot1_returns[batch_indices]
                                batch_robot2_returns = robot2_returns[batch_indices]
                                batch_robot1_old_logits = robot1_old_logits[batch_indices]
                                batch_robot2_old_logits = robot2_old_logits[batch_indices]
                                
                                # Current episode number
                                current_episode = len(self.training_history['episode_rewards'])
                                
                                # Train actor
                                actor_metrics = self.model.train_actor(
                                    batch_states, batch_frontiers, 
                                    batch_robot1_pos, batch_robot2_pos,
                                    batch_robot1_target, batch_robot2_target,
                                    batch_robot1_actions, batch_robot2_actions,
                                    batch_robot1_advantages, batch_robot2_advantages,
                                    batch_robot1_old_logits, batch_robot2_old_logits,
                                    self.training_history, current_episode
                                )
                                
                                # Train critic
                                critic_metrics = self.model.train_critic(
                                    batch_states, batch_frontiers,
                                    batch_robot1_pos, batch_robot2_pos,
                                    batch_robot1_target, batch_robot2_target,
                                    batch_robot1_returns, batch_robot2_returns
                                )
                                
                                # Record losses
                                ppo_losses.append(actor_metrics['total_policy_loss'] + critic_metrics['total_value_loss'])
                            
                            # Record average loss
                            if ppo_losses:
                                loss_metrics = {
                                    'total_loss': np.mean(ppo_losses),
                                    'robot1_entropy': actor_metrics['robot1_entropy'],
                                    'robot2_entropy': actor_metrics['robot2_entropy'],
                                    'robot1_value_loss': critic_metrics['robot1_value_loss'],
                                    'robot2_value_loss': critic_metrics['robot2_value_loss'],
                                    'robot1_policy_loss': actor_metrics['robot1_policy_loss'],
                                    'robot2_policy_loss': actor_metrics['robot2_policy_loss']
                                }
                                episode_losses.append(float(loss_metrics['total_loss']) if hasattr(loss_metrics['total_loss'], 'numpy') else loss_metrics['total_loss'])
                                
                                # Record entropy-related metrics
                                self.training_history['robot1_entropy'].append(loss_metrics['robot1_entropy'].numpy())
                                self.training_history['robot2_entropy'].append(loss_metrics['robot2_entropy'].numpy())
                                self.training_history['robot1_value_loss'].append(loss_metrics['robot1_value_loss'].numpy())
                                self.training_history['robot2_value_loss'].append(loss_metrics['robot2_value_loss'].numpy())
                                self.training_history['robot1_policy_loss'].append(loss_metrics['robot1_policy_loss'].numpy())
                                self.training_history['robot2_policy_loss'].append(loss_metrics['robot2_policy_loss'].numpy())
                        
                        # Reset trajectory buffer
                        self.reset_trajectory_buffer()
                
                # Calculate overlap area (this code runs outside the training loop, after an episode ends)
                overlap_ratio = self.map_tracker.calculate_overlap()
                robot1_ratio, robot2_ratio = self.map_tracker.get_exploration_ratio()
                
                # Record overlap ratio
                self.overlap_ratios.append(overlap_ratio)
                
                # Stop tracking
                self.map_tracker.stop_tracking()
                
                # Post-episode processing
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
                
                # Save model
                if (episode + 1) % save_freq == 0:
                    self.save_checkpoint(episode + 1)
                    self.plot_training_progress()
                    
                    # Save coverage over time chart
                    if hasattr(self.map_tracker, 'plot_coverage_over_time'):
                        self.map_tracker.plot_coverage_over_time()
                
                # Update exploration rate
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                # Print training information
                print(f"\nEpisode {episode + 1}/{episodes} (Map {self.robot1.li_map})")
                print(f"Steps: {steps}, Total reward: {total_reward:.2f}")
                print(f"Robot1 reward: {robot1_total_reward:.2f}")
                print(f"Robot2 reward: {robot2_total_reward:.2f}")
                print(f"Exploration rate: {self.epsilon:.3f}")
                print(f"Average loss: {self.training_history['losses'][-1]:.6f}")

                # Add Actor and Critic loss output
                if 'robot1_policy_loss' in self.training_history and len(self.training_history['robot1_policy_loss']) > 0:
                    avg_actor_loss = (self.training_history['robot1_policy_loss'][-1] + 
                                    self.training_history['robot2_policy_loss'][-1]) / 2
                    avg_critic_loss = (self.training_history['robot1_value_loss'][-1] + 
                                    self.training_history['robot2_value_loss'][-1]) / 2
                    print(f"Actor Loss: {avg_actor_loss:.6f}")
                    print(f"Critic Loss: {avg_critic_loss:.6f}")

                print(f"Exploration progress: {exploration_progress:.1%}")

                # Print robot exploration overlap information
                print(f"Robot1 exploration coverage: {robot1_ratio:.2%}")
                print(f"Robot2 exploration coverage: {robot2_ratio:.2%}")
                print(f"Robot local map area overlap: {overlap_ratio:.2%}")
                
                if exploration_progress >= self.robot1.finish_percent:
                    print("Map exploration complete!")
                else:
                    print("Map exploration incomplete")
                print("-" * 50)
                
                # Reset environment
                state = self.robot1.reset()
                self.robot2.reset()
            
            # Save final model after training ends
            self.save_checkpoint(episodes)
            
        except Exception as e:
            print(f"Training process error: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Ensure resources are cleaned up
            if hasattr(self.robot1, 'cleanup_visualization'):
                self.robot1.cleanup_visualization()
            if hasattr(self.robot2, 'cleanup_visualization'):
                self.robot2.cleanup_visualization()
            self.map_tracker.cleanup()
    
    def plot_training_progress(self):
        """Plot training progress charts"""
        # Debug information
        print(f"Plot info - Overlap ratio data length: {len(self.overlap_ratios)}")
        print(f"Plot info - First 5 overlap ratio elements: {self.overlap_ratios[:5] if len(self.overlap_ratios) >= 5 else self.overlap_ratios}")

        # Check if overlap ratio data exists
        has_overlap_data = hasattr(self, 'overlap_ratios') and len(self.overlap_ratios) > 0
        n_plots = 7

        fig, axs = plt.subplots(n_plots, 1, figsize=(12, n_plots * 3.5))
        if n_plots == 1:
            axs = [axs]

        # Ensure all data has consistent length
        data_length = min(
            len(self.training_history['episode_rewards']),
            len(self.training_history['robot1_rewards']),
            len(self.training_history['robot2_rewards']),
            len(self.training_history['episode_lengths']),
            len(self.training_history['losses']),
            len(self.training_history['exploration_progress'])
        )

        episodes = range(1, data_length + 1)

        # Clip all data to the same length
        episode_rewards = self.training_history['episode_rewards'][:data_length]
        robot1_rewards = self.training_history['robot1_rewards'][:data_length]
        robot2_rewards = self.training_history['robot2_rewards'][:data_length]
        episode_lengths = self.training_history['episode_lengths'][:data_length]
        losses = self.training_history['losses'][:data_length]
        exploration_progress = self.training_history['exploration_progress'][:data_length]

        # Plot total reward
        axs[0].plot(episodes, episode_rewards, color='#2E8B57')
        axs[0].set_title('Total Reward')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        axs[0].grid(True)

        # Plot per-robot rewards
        axs[1].plot(episodes, robot1_rewards, color='#8A2BE2', label='Robot1')
        axs[1].plot(episodes, robot2_rewards, color='#FFA500', label='Robot2')
        axs[1].set_title('Reward per Robot')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('Reward')
        axs[1].legend()
        axs[1].grid(True)

        # Plot steps per episode
        axs[2].plot(episodes, episode_lengths, color='#4169E1')
        axs[2].set_title('Steps per Episode')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Steps')
        axs[2].grid(True)

        # Plot Actor loss (if available)
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

        # Plot Critic loss (if available)
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

        # Plot exploration progress
        axs[5].plot(episodes, exploration_progress, color='#2F4F4F')
        axs[5].set_title('Exploration Progress')
        axs[5].set_xlabel('Episode')
        axs[5].set_ylabel('Completion Rate')
        axs[5].grid(True)

        # Plot overlap ratio (if available)
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

        # Additional plot for robot rewards comparison
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

        # Additional plot for overlap ratio (if available)
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
        """Save training history"""
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
    
    def load_training_history(self, filename='a2c_training_history.npz'):
        """Load training history"""
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
        
        # Load optional items
        if 'robot1_entropy' in data:
            self.training_history['robot1_entropy'] = data['robot1_entropy'].tolist()
            self.training_history['robot2_entropy'] = data['robot2_entropy'].tolist()
            self.training_history['robot1_value_loss'] = data['robot1_value_loss'].tolist()
            self.training_history['robot2_value_loss'] = data['robot2_value_loss'].tolist()
            self.training_history['robot1_policy_loss'] = data['robot1_policy_loss'].tolist()
            self.training_history['robot2_policy_loss'] = data['robot2_policy_loss'].tolist()
            
        # Load overlap ratios
        if 'overlap_ratios' in data:
            self.overlap_ratios = data['overlap_ratios'].tolist()
        else:
            self.overlap_ratios = []
    
    def save_checkpoint(self, episode):
        """Save checkpoint
        
        Args:
            episode: Current training episode
        """
        # Zero-pad episode number for correct sorting
        ep_str = str(episode).zfill(6)
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'multi_robot_model_a2c_ep{ep_str}.h5')
        self.model.save(model_path)
        
        # Save training history
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
        
        # Add optional history items
        if 'robot1_entropy' in self.training_history and self.training_history['robot1_entropy']:
            history_to_save['robot1_entropy'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot1_entropy']]
            history_to_save['robot2_entropy'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot2_entropy']]
            history_to_save['robot1_value_loss'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot1_value_loss']]
            history_to_save['robot2_value_loss'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot2_value_loss']]
            history_to_save['robot1_policy_loss'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot1_policy_loss']]
            history_to_save['robot2_policy_loss'] = [float(x) if hasattr(x, 'numpy') else float(x) for x in self.training_history['robot2_policy_loss']]
        
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=4)
        
        print(f"Checkpoint saved at episode {episode}")