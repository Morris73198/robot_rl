import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from two_robot_dueling_dqn_attention.config import MODEL_DIR, ROBOT_CONFIG, REWARD_CONFIG
from two_robot_dueling_dqn_attention.environment.robot_local_map_tracker import RobotIndividualMapTracker  # Import the tracker

class MultiRobotTrainer:
    def __init__(self, model, robot1, robot2, memory_size=10000, batch_size=16, gamma=0.99):
        self.model = model
        self.robot1 = robot1
        self.robot2 = robot2
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.map_size = self.robot1.map_size
        
        # 訓練參數
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # 訓練歷史記錄
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'exploration_rates': [],
            'losses': [],
            'robot1_rewards': [],
            'robot2_rewards': [],
            'exploration_progress': []
        }
        
        # 創建機器人個人地圖追蹤器
        self.map_tracker = RobotIndividualMapTracker(robot1, robot2)
    
    def remember(self, state, frontiers, robot1_pos, robot2_pos, 
                robot1_target, robot2_target,
                robot1_action, robot2_action, robot1_reward, robot2_reward,
                next_state, next_frontiers, next_robot1_pos, next_robot2_pos, 
                next_robot1_target, next_robot2_target, done):
        """存儲經驗到回放緩衝區"""
        self.memory.append((
            state, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target,
            robot1_action, robot2_action, robot1_reward, robot2_reward,
            next_state, next_frontiers, next_robot1_pos, next_robot2_pos,
            next_robot1_target, next_robot2_target, done
        ))
    
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
    
    # def choose_actions(self, state, frontiers, robot1_pos, robot2_pos):
    #     """為兩個機器人選擇動作"""
    #     if len(frontiers) == 0:
    #         return 0, 0
            
    #     MIN_TARGET_DISTANCE = 50
        
    #     # epsilon-greedy策略
    #     if np.random.random() < self.epsilon:
    #         valid_frontiers1 = list(range(min(self.model.max_frontiers, len(frontiers))))
    #         valid_frontiers2 = valid_frontiers1.copy()
            
    #         if self.robot2.current_target_frontier is not None:
    #             valid_frontiers1 = [
    #                 i for i in valid_frontiers1 
    #                 if np.linalg.norm(frontiers[i] - self.robot2.current_target_frontier) >= MIN_TARGET_DISTANCE
    #             ]
                
    #         if self.robot1.current_target_frontier is not None:
    #             valid_frontiers2 = [
    #                 i for i in valid_frontiers2 
    #                 if np.linalg.norm(frontiers[i] - self.robot1.current_target_frontier) >= MIN_TARGET_DISTANCE
    #             ]
                
    #         if not valid_frontiers1:
    #             valid_frontiers1 = list(range(min(self.model.max_frontiers, len(frontiers))))
    #         if not valid_frontiers2:
    #             valid_frontiers2 = list(range(min(self.model.max_frontiers, len(frontiers))))
                
    #         robot1_action = np.random.choice(valid_frontiers1)
    #         robot2_action = np.random.choice(valid_frontiers2)
    #         return robot1_action, robot2_action
        
    #     # 使用模型預測
    #     state_batch = np.expand_dims(state, 0)
    #     frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0)
    #     robot1_pos_batch = np.expand_dims(robot1_pos, 0)
    #     robot2_pos_batch = np.expand_dims(robot2_pos, 0)
        
    #     # 獲取當前目標位置
    #     robot1_target = self.get_normalized_target(self.robot1.current_target_frontier)
    #     robot2_target = self.get_normalized_target(self.robot2.current_target_frontier)
    #     robot1_target_batch = np.expand_dims(robot1_target, 0)
    #     robot2_target_batch = np.expand_dims(robot2_target, 0)
        
    #     predictions = self.model.predict(
    #         state_batch, frontiers_batch, 
    #         robot1_pos_batch, robot2_pos_batch,
    #         robot1_target_batch, robot2_target_batch
    #     )
        
    #     valid_frontiers = min(self.model.max_frontiers, len(frontiers))
    #     robot1_q = predictions['robot1'][0, :valid_frontiers].copy()
    #     robot2_q = predictions['robot2'][0, :valid_frontiers].copy()
        
    #     # 根據其他機器人的目標調整Q值
    #     if self.robot2.current_target_frontier is not None:
    #         for i in range(valid_frontiers):
    #             if np.linalg.norm(frontiers[i] - self.robot2.current_target_frontier) < MIN_TARGET_DISTANCE:
    #                 robot1_q[i] *= 0.0001
                    
    #     if self.robot1.current_target_frontier is not None:
    #         for i in range(valid_frontiers):
    #             if np.linalg.norm(frontiers[i] - self.robot1.current_target_frontier) < MIN_TARGET_DISTANCE:
    #                 robot2_q[i] *= 0.0001
        
    #     robot1_action = np.argmax(robot1_q)
    #     robot2_action = np.argmax(robot2_q)
        
    #     return robot1_action, robot2_action

    def train_step(self):
        """執行一步訓練"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = []
        frontiers_batch = []
        robot1_pos_batch = []
        robot2_pos_batch = []
        robot1_target_batch = []
        robot2_target_batch = []
        next_states = []
        next_frontiers_batch = []
        next_robot1_pos_batch = []
        next_robot2_pos_batch = []
        next_robot1_target_batch = []
        next_robot2_target_batch = []
        
        for (state, frontiers, robot1_pos, robot2_pos, 
             robot1_target, robot2_target, _, _, _, _,
             next_state, next_frontiers, next_robot1_pos, next_robot2_pos,
             next_robot1_target, next_robot2_target, _) in batch:
            
            if len(state.shape) == 2:
                state = np.expand_dims(state, axis=-1)
            if len(next_state.shape) == 2:
                next_state = np.expand_dims(next_state, axis=-1)
            
            states.append(state)
            frontiers_batch.append(self.pad_frontiers(frontiers))
            robot1_pos_batch.append(robot1_pos)
            robot2_pos_batch.append(robot2_pos)
            robot1_target_batch.append(self.get_normalized_target(robot1_target))
            robot2_target_batch.append(self.get_normalized_target(robot2_target))
            
            next_states.append(next_state)
            next_frontiers_batch.append(self.pad_frontiers(next_frontiers))
            next_robot1_pos_batch.append(next_robot1_pos)
            next_robot2_pos_batch.append(next_robot2_pos)
            next_robot1_target_batch.append(self.get_normalized_target(next_robot1_target))
            next_robot2_target_batch.append(self.get_normalized_target(next_robot2_target))
        
        states = np.array(states)
        frontiers_batch = np.array(frontiers_batch)
        robot1_pos_batch = np.array(robot1_pos_batch)
        robot2_pos_batch = np.array(robot2_pos_batch)
        robot1_target_batch = np.array(robot1_target_batch)
        robot2_target_batch = np.array(robot2_target_batch)
        next_states = np.array(next_states)
        next_frontiers_batch = np.array(next_frontiers_batch)
        next_robot1_pos_batch = np.array(next_robot1_pos_batch)
        next_robot2_pos_batch = np.array(next_robot2_pos_batch)
        next_robot1_target_batch = np.array(next_robot1_target_batch)
        next_robot2_target_batch = np.array(next_robot2_target_batch)
        
        # 使用目標網絡計算下一個狀態的Q值
        target_predictions = self.model.target_model.predict({
            'map_input': next_states,
            'frontier_input': next_frontiers_batch,
            'robot1_pos_input': next_robot1_pos_batch,
            'robot2_pos_input': next_robot2_pos_batch,
            'robot1_target_input': next_robot1_target_batch,
            'robot2_target_input': next_robot2_target_batch
        })
        
        # 使用當前網絡計算當前Q值
        current_predictions = self.model.model.predict({
            'map_input': states,
            'frontier_input': frontiers_batch,
            'robot1_pos_input': robot1_pos_batch,
            'robot2_pos_input': robot2_pos_batch,
            'robot1_target_input': robot1_target_batch,
            'robot2_target_input': robot2_target_batch
        })
        
        # 準備訓練目標
        robot1_targets = current_predictions['robot1'].copy()
        robot2_targets = current_predictions['robot2'].copy()
        
        # 更新Q值
        for i, (_, _, _, _, _, _, robot1_action, robot2_action, 
               robot1_reward, robot2_reward, _, _, _, _, _, _, done) in enumerate(batch):
            
            robot1_action = min(robot1_action, self.model.max_frontiers - 1)
            robot2_action = min(robot2_action, self.model.max_frontiers - 1)
            
            if done:
                robot1_targets[i][robot1_action] = robot1_reward
                robot2_targets[i][robot2_action] = robot2_reward
            else:
                robot1_targets[i][robot1_action] = robot1_reward + \
                    self.gamma * np.max(target_predictions['robot1'][i])
                robot2_targets[i][robot2_action] = robot2_reward + \
                    self.gamma * np.max(target_predictions['robot2'][i])
        
        # 訓練模型
        loss = self.model.train_on_batch(
            states, frontiers_batch, 
            robot1_pos_batch, robot2_pos_batch,
            robot1_target_batch, robot2_target_batch,
            robot1_targets, robot2_targets
        )
        
        return loss
    
    # def train(self, episodes=1000000, target_update_freq=10, save_freq=10):
    #     """執行多機器人協同訓練"""
    #     try:
    #         for episode in range(episodes):
    #             # 初始化環境和狀態
    #             state = self.robot1.begin()
    #             self.robot2.begin()
                
    #             # 啟動地圖追蹤器
    #             self.map_tracker.start_tracking()
                
    #             # 初始化episode統計
    #             total_reward = 0
    #             robot1_total_reward = 0
    #             robot2_total_reward = 0
    #             steps = 0
    #             episode_losses = []
                
    #             # while not (self.robot1.check_done() or self.robot2.check_done()):
    #             while not (self.robot1.check_done() or self.robot2.check_done() or steps >= 1500):
    #                 frontiers = self.robot1.get_frontiers()
    #                 if len(frontiers) == 0:
    #                     break
                        
    #                 # 獲取當前狀態
    #                 robot1_pos = self.robot1.get_normalized_position()
    #                 robot2_pos = self.robot2.get_normalized_position()
    #                 old_robot1_pos = self.robot1.robot_position.copy()
    #                 old_robot2_pos = self.robot2.robot_position.copy()
                    
    #                 # 標準化目標位置
    #                 map_dims = np.array([float(self.robot1.map_size[1]), float(self.robot1.map_size[0])])
    #                 robot1_target = (np.zeros(2) if self.robot1.current_target_frontier is None 
    #                             else self.robot1.current_target_frontier / map_dims)
    #                 robot2_target = (np.zeros(2) if self.robot2.current_target_frontier is None 
    #                             else self.robot2.current_target_frontier / map_dims)
                    
    #                 # 選擇動作（使用epsilon-greedy策略）
    #                 if np.random.random() < self.epsilon:
    #                     valid_frontiers = min(self.model.max_frontiers, len(frontiers))
    #                     robot1_action = np.random.randint(valid_frontiers)
    #                     robot2_action = np.random.randint(valid_frontiers)
    #                 else:
    #                     # 使用模型預測
    #                     predictions = self.model.predict(
    #                         np.expand_dims(state, 0),
    #                         np.expand_dims(self.pad_frontiers(frontiers), 0),
    #                         np.expand_dims(robot1_pos, 0),
    #                         np.expand_dims(robot2_pos, 0),
    #                         np.expand_dims(robot1_target, 0),
    #                         np.expand_dims(robot2_target, 0)
    #                     )
                        
    #                     valid_frontiers = min(self.model.max_frontiers, len(frontiers))
    #                     robot1_action = np.argmax(predictions['robot1'][0, :valid_frontiers])
    #                     robot2_action = np.argmax(predictions['robot2'][0, :valid_frontiers])
                    
    #                 # 設置目標並執行移動
    #                 robot1_target = frontiers[robot1_action]
    #                 robot2_target = frontiers[robot2_action]
                    
    #                 # 移動機器人
    #                 next_state1, r1, d1 = self.robot1.move_to_frontier(robot1_target)
    #                 robot1_reward = r1
    #                 self.robot2.op_map = self.robot1.op_map.copy()
                    
    #                 next_state2, r2, d2 = self.robot2.move_to_frontier(robot2_target)
    #                 robot2_reward = r2
    #                 self.robot1.op_map = self.robot2.op_map.copy()
                    
    #                 # 更新機器人位置
    #                 self.robot1.other_robot_position = self.robot2.robot_position.copy()
    #                 self.robot2.other_robot_position = self.robot1.robot_position.copy()
                    
    #                 # 更新地圖追蹤器
    #                 self.map_tracker.update()
                    
    #                 # 保存經驗到回放緩衝區
    #                 if d1 or d2:
    #                     self.remember(
    #                         state, frontiers, robot1_pos, robot2_pos,
    #                         self.robot1.current_target_frontier, self.robot2.current_target_frontier,
    #                         robot1_action, robot2_action,
    #                         robot1_reward, robot2_reward,
    #                         next_state1, frontiers, 
    #                         self.robot1.get_normalized_position(), 
    #                         self.robot2.get_normalized_position(),
    #                         self.robot1.current_target_frontier,
    #                         self.robot2.current_target_frontier,
    #                         True
    #                     )
                        
    #                     # 進行訓練
    #                     loss = self.train_step()
    #                     if loss is not None:
    #                         if isinstance(loss, list):
    #                             episode_losses.append(np.mean(loss))
    #                         else:
    #                             episode_losses.append(loss)
                    
    #                 # 更新狀態和獎勵
    #                 state = next_state1
    #                 total_reward += (robot1_reward + robot2_reward)
    #                 robot1_total_reward += robot1_reward
    #                 robot2_total_reward += robot2_reward
    #                 steps += 1
                    
    #                 # 更新視覺化
    #                 if steps % ROBOT_CONFIG['plot_interval'] == 0:
    #                     if self.robot1.plot:
    #                         self.robot1.plot_env()
    #                     if self.robot2.plot:
    #                         self.robot2.plot_env()
                            
    #             # if steps < 1500:
    #             #     total_reward += REWARD_CONFIG['completion_reward']
                
    #             # 計算重疊區域
    #             overlap_ratio = self.map_tracker.calculate_overlap()
    #             robot1_ratio, robot2_ratio = self.map_tracker.get_exploration_ratio()
                
    #             # 停止追蹤
    #             self.map_tracker.stop_tracking()
                
    #             # Episode結束後的處理
    #             exploration_progress = self.robot1.get_exploration_progress()
    #             self.training_history['episode_rewards'].append(total_reward)
    #             self.training_history['robot1_rewards'].append(robot1_total_reward)
    #             self.training_history['robot2_rewards'].append(robot2_total_reward)
    #             self.training_history['episode_lengths'].append(steps)
    #             self.training_history['exploration_rates'].append(self.epsilon)
    #             self.training_history['losses'].append(
    #                 np.mean(episode_losses) if episode_losses else 0
    #             )
    #             self.training_history['exploration_progress'].append(exploration_progress)
                
    #             # 更新目標網絡和保存模型
    #             if (episode + 1) % target_update_freq == 0:
    #                 self.model.update_target_model()
                
    #             if (episode + 1) % save_freq == 0:
    #                 self.save_checkpoint(episode + 1)
    #                 self.plot_training_progress()
                
    #             # 更新探索率
    #             if self.epsilon > self.epsilon_min:
    #                 self.epsilon *= self.epsilon_decay
                
    #             # 列印訓練信息
    #             print(f"\n第 {episode + 1}/{episodes} 輪 (地圖 {self.robot1.li_map})")
    #             print(f"步數: {steps}, 總獎勵: {total_reward:.2f}")
    #             print(f"Robot1 獎勵: {robot1_total_reward:.2f}")
    #             print(f"Robot2 獎勵: {robot2_total_reward:.2f}")
    #             print(f"探索率: {self.epsilon:.3f}")
    #             print(f"平均損失: {self.training_history['losses'][-1]:.6f}")
    #             print(f"探索進度: {exploration_progress:.1%}")
                
    #             # 印出機器人探索重疊信息
    #             print(f"Robot1 探索覆蓋率: {robot1_ratio:.2%}")
    #             print(f"Robot2 探索覆蓋率: {robot2_ratio:.2%}")
    #             print(f"機器人local map區域交集: {overlap_ratio:.2%}")
                
    #             if exploration_progress >= self.robot1.finish_percent:
    #                 print("地圖探索完成！")
    #             else:
    #                 print("地圖探索未完成")
    #             print("-" * 50)
                
    #             # 重置環境
    #             state = self.robot1.reset()
    #             self.robot2.reset()
            
    #         # 訓練結束後保存最終模型
    #         self.save_checkpoint(episodes)
            
    #     except Exception as e:
    #         print(f"訓練過程出現錯誤: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
            
    #     finally:
    #         # 確保清理資源
    #         if hasattr(self.robot1, 'cleanup_visualization'):
    #             self.robot1.cleanup_visualization()
    #         if hasattr(self.robot2, 'cleanup_visualization'):
    #             self.robot2.cleanup_visualization()
    #         self.map_tracker.cleanup()
    
    
    
    def train(self, episodes=1000000, target_update_freq=10, save_freq=10):
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
                    
                    valid_frontiers = min(self.model.max_frontiers, len(frontiers))
                    
                    # 初始化 robot2_q，避免未定義錯誤
                    robot2_q = np.zeros(valid_frontiers)
                    
                    # 選擇動作（使用增強的epsilon-greedy策略）
                    if np.random.random() < self.epsilon:
                        # 增加隨機性：有30%的概率選擇距離當前位置最遠的frontier點
                        if np.random.random() < 0.3 and len(frontiers) > 0:
                            distances1 = np.linalg.norm(frontiers - self.robot1.robot_position, axis=1)
                            distances2 = np.linalg.norm(frontiers - self.robot2.robot_position, axis=1)
                            robot1_action = np.argmax(distances1)
                            robot2_action = np.argmax(distances2)
                        else:
                            # 常規隨機選擇
                            robot1_action = np.random.randint(valid_frontiers)
                            robot2_action = np.random.randint(valid_frontiers)
                    else:
                        # 使用模型預測
                        predictions = self.model.predict(
                            np.expand_dims(state, 0),
                            np.expand_dims(self.pad_frontiers(frontiers), 0),
                            np.expand_dims(robot1_pos, 0),
                            np.expand_dims(robot2_pos, 0),
                            np.expand_dims(robot1_target, 0),
                            np.expand_dims(robot2_target, 0)
                        )
                        
                        robot1_q = predictions['robot1'][0, :valid_frontiers].copy()
                        robot2_q = predictions['robot2'][0, :valid_frontiers].copy()
                        
                        # 先讓robot1選擇
                        robot1_action = np.argmax(robot1_q)
                        robot1_target_point = frontiers[robot1_action]
                        
                        # 為robot2調整Q值，降低接近robot1目標的點的價值
                        for i in range(valid_frontiers):
                            distance = np.linalg.norm(frontiers[i] - robot1_target_point)
                            if distance < MIN_TARGET_DISTANCE:
                                # 使用距離相關的懲罰因子
                                penalty = (1.0 - (distance / MIN_TARGET_DISTANCE)) * 0.9
                                
                                # 根據Q值符號選擇適當懲罰方式
                                if robot2_q[i] >= 0:
                                    robot2_q[i] *= (1.0 - penalty)  # 正值：乘以小於1的數使其變小
                                else:
                                    robot2_q[i] *= (1.0 + penalty)  # 負值：乘以大於1的數使其更負

                        robot2_action = np.argmax(robot2_q)
                    
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
                                # 如果目標太近，為robot2重新選擇目標
                                adjusted_robot2_q = robot2_q.copy()
                                for i in range(valid_frontiers):
                                    distance = np.linalg.norm(frontiers[i] - robot1_target_point)
                                    if distance < MIN_TARGET_DISTANCE:
                                        penalty = (1.0 - (distance / MIN_TARGET_DISTANCE)) * 0.9
                                        
                                        # 根據Q值符號選擇適當懲罰方式
                                        if adjusted_robot2_q[i] >= 0:
                                            adjusted_robot2_q[i] *= (1.0 - penalty)  # 正值：乘以小於1的數使其變小
                                        else:
                                            adjusted_robot2_q[i] *= (1.0 + penalty)  # 負值：乘以大於1的數使其更負
                                        
                                new_robot2_action = np.argmax(adjusted_robot2_q)
                                robot2_target_point = frontiers[new_robot2_action]
                            else:
                                # 在隨機模式下，簡單地選擇一個遠離robot1目標的點
                                distances = np.linalg.norm(frontiers - robot1_target_point, axis=1)
                                farthest_indices = np.argsort(distances)[-min(5, valid_frontiers):]  # 選最遠的5個點
                                new_robot2_action = farthest_indices[np.random.randint(len(farthest_indices))]
                                robot2_target_point = frontiers[new_robot2_action]
                    
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
                    
                    # 保存經驗到回放緩衝區
                    if d1 or d2:
                        self.remember(
                            state, frontiers, robot1_pos, robot2_pos,
                            self.robot1.current_target_frontier, self.robot2.current_target_frontier,
                            robot1_action, robot2_action,
                            robot1_reward, robot2_reward,
                            next_state1, frontiers, 
                            self.robot1.get_normalized_position(), 
                            self.robot2.get_normalized_position(),
                            self.robot1.current_target_frontier,
                            self.robot2.current_target_frontier,
                            True
                        )
                        
                        # 進行訓練
                        loss = self.train_step()
                        if loss is not None:
                            if isinstance(loss, list):
                                episode_losses.append(np.mean(loss))
                            else:
                                episode_losses.append(loss)
                    
                    # 更新狀態和獎勵
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
                
                # 計算重疊區域
                overlap_ratio = self.map_tracker.calculate_overlap()
                robot1_ratio, robot2_ratio = self.map_tracker.get_exploration_ratio()
                
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
                
                # 更新目標網絡和保存模型
                if (episode + 1) % target_update_freq == 0:
                    self.model.update_target_model()
                
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
        fig, axs = plt.subplots(6, 1, figsize=(12, 20))
        
        episodes = range(1, len(self.training_history['episode_rewards']) + 1)
        
        # 繪製總獎勵
        axs[0].plot(episodes, self.training_history['episode_rewards'], color='#2E8B57')  # 深綠色表示總體
        axs[0].set_title('total reward')
        axs[0].set_xlabel('episode')
        axs[0].set_ylabel('reward')
        axs[0].grid(True)
        
        # 繪製各機器人獎勵
        axs[1].plot(episodes, self.training_history['robot1_rewards'], 
                    color='#8A2BE2', label='Robot1')  # 紫色
        axs[1].plot(episodes, self.training_history['robot2_rewards'], 
                    color='#FFA500', label='Robot2')  # 橘色
        axs[1].set_title('reward per robot')
        axs[1].set_xlabel('episode')
        axs[1].set_ylabel('reward')
        axs[1].legend()
        axs[1].grid(True)
        
        # 繪製步數
        axs[2].plot(episodes, self.training_history['episode_lengths'], color='#4169E1')  # 藍色
        axs[2].set_title('step per episode')
        axs[2].set_xlabel('episode')
        axs[2].set_ylabel('step')
        axs[2].grid(True)
        
        # 繪製探索率
        axs[3].plot(episodes, self.training_history['exploration_rates'], color='#DC143C')  # 深紅色
        axs[3].set_title('epsilon rate')
        axs[3].set_xlabel('episode')
        axs[3].set_ylabel('Epsilon')
        axs[3].grid(True)
        
        # 繪製損失
        axs[4].plot(episodes, self.training_history['losses'], color='#2F4F4F')  # 深灰色
        axs[4].set_title('training loss')
        axs[4].set_xlabel('episode')
        axs[4].set_ylabel('loss')
        axs[4].grid(True)
        
        # 繪製探索進度
        axs[5].plot(episodes, self.training_history['exploration_progress'], color='#228B22')  # 森林綠
        axs[5].set_title('exploration progress')
        axs[5].set_xlabel('episode')
        axs[5].set_ylabel('exploration rate')
        axs[5].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
        
        # 另外繪製一個單獨的兩機器人獎勵對比圖
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, self.training_history['robot1_rewards'], 
                color='#8A2BE2', label='Robot1', alpha=0.7)  # 紫色
        plt.plot(episodes, self.training_history['robot2_rewards'], 
                color='#FFA500', label='Robot2', alpha=0.7)  # 橘色
        plt.fill_between(episodes, self.training_history['robot1_rewards'], 
                        alpha=0.3, color='#9370DB')  # 淺紫色填充
        plt.fill_between(episodes, self.training_history['robot2_rewards'], 
                        alpha=0.3, color='#FFB84D')  # 淺橘色填充
        plt.title('機器人獎勵對比')
        plt.xlabel('輪數')
        plt.ylabel('獎勵')
        plt.legend()
        plt.grid(True)
        plt.savefig('robots_rewards_comparison.png')
        plt.close()
    
    def save_training_history(self, filename='training_history.npz'):
        """保存訓練歷史"""
        np.savez(
            filename,
            episode_rewards=self.training_history['episode_rewards'],
            robot1_rewards=self.training_history['robot1_rewards'],
            robot2_rewards=self.training_history['robot2_rewards'],
            episode_lengths=self.training_history['episode_lengths'],
            exploration_rates=self.training_history['exploration_rates'],
            losses=self.training_history['losses'],
            exploration_progress=self.training_history['exploration_progress']
        )
    
    def load_training_history(self, filename='training_history.npz'):
        """載入訓練歷史"""
        data = np.load(filename)
        self.training_history = {
            'episode_rewards': data['episode_rewards'].tolist(),
            'robot1_rewards': data['robot1_rewards'].tolist(),
            'robot2_rewards': data['robot2_rewards'].tolist(),
            'episode_lengths': data['episode_lengths'].tolist(),
            'exploration_rates': data['exploration_rates'].tolist(),
            'losses': data['losses'].tolist(),
            'exploration_progress': data['exploration_progress'].tolist()
        }
        
    def save_checkpoint(self, episode):
        """保存檢查點
        
        Args:
            episode: 當前訓練輪數
        """
        # 用零填充確保文件名排序正確
        ep_str = str(episode).zfill(6)
        
        # 保存模型
        model_path = os.path.join(MODEL_DIR, f'multi_robot_model_attention_ep{ep_str}.h5')
        self.model.save(model_path)
        
        # 保存訓練歷史
        history_path = os.path.join(MODEL_DIR, f'multi_robot_training_history_ep{ep_str}.json')
        history_to_save = {
            'episode_rewards': [float(x) for x in self.training_history['episode_rewards']],
            'robot1_rewards': [float(x) for x in self.training_history['robot1_rewards']],
            'robot2_rewards': [float(x) for x in self.training_history['robot2_rewards']],
            'episode_lengths': [int(x) for x in self.training_history['episode_lengths']],
            'exploration_rates': [float(x) for x in self.training_history['exploration_rates']],
            'losses': [float(x) if x is not None else 0.0 for x in self.training_history['losses']],
            'exploration_progress': [float(x) for x in self.training_history['exploration_progress']]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=4)
        
        print(f"已在第 {episode} 輪保存檢查點")

    
    def choose_new_target_with_coordination(self, robot, other_robot, frontiers):
        """
        為單個機器人選擇新目標，同時考慮另一個機器人的當前目標
        
        Args:
            robot: 需要選擇新目標的機器人
            other_robot: 另一個機器人
            frontiers: 可用的前沿點列表
        
        Returns:
            選擇的目標前沿點
        """
        if len(frontiers) == 0:
            return None
        
        # 獲取當前機器人的狀態
        robot_pos = robot.get_normalized_position()
        robot_target = None if robot.current_target_frontier is None else self.get_normalized_target(robot.current_target_frontier)
        
        # 獲取另一個機器人的狀態
        other_pos = other_robot.get_normalized_position()
        other_target = None if other_robot.current_target_frontier is None else self.get_normalized_target(other_robot.current_target_frontier)
        
        # 使用模型預測
        predictions = self.model.predict(
            np.expand_dims(robot.get_observation(), 0),
            np.expand_dims(self.pad_frontiers(frontiers), 0),
            np.expand_dims(robot_pos, 0),
            np.expand_dims(other_pos, 0),
            np.expand_dims(robot_target if robot_target is not None else np.zeros(2), 0),
            np.expand_dims(other_target if other_target is not None else np.zeros(2), 0)
        )
        
        # 提取當前機器人的Q值
        valid_frontiers = min(self.model.max_frontiers, len(frontiers))
        if robot == self.robot1:
            q_values = predictions['robot1'][0, :valid_frontiers].copy()
        else:
            q_values = predictions['robot2'][0, :valid_frontiers].copy()
        
        # 協調選擇：避免與另一個機器人選擇相近的目標
        MIN_TARGET_DISTANCE = robot.sensor_range * 1.5
        
        # 檢查另一個機器人是否有目標，如果有則調整Q值
        if other_robot.current_target_frontier is not None:
            for i in range(valid_frontiers):
                distance = np.linalg.norm(frontiers[i] - other_robot.current_target_frontier)
                if distance < MIN_TARGET_DISTANCE:
                    # 使用距離相關的懲罰因子
                    penalty = 1.0 - (distance / MIN_TARGET_DISTANCE)
                    q_values[i] *= (1.0 - penalty * 0.9)
        
        # 選擇最優的動作
        action = np.argmax(q_values)
        return frontiers[action]