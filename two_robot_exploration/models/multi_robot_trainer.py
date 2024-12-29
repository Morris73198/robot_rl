import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from two_robot_exploration.config import MODEL_DIR, ROBOT_CONFIG

class MultiRobotTrainer:
    def __init__(self, model, robot1, robot2, memory_size=10000, batch_size=16, gamma=0.99):
        """初始化多機器人訓練器

        Args:
            model: MultiRobotNetworkModel實例
            robot1: 第一個Robot實例
            robot2: 第二個Robot實例
            memory_size: 經驗回放緩衝區大小
            batch_size: 訓練批次大小
            gamma: 獎勵折扣因子
        """
        self.model = model
        self.robot1 = robot1
        self.robot2 = robot2
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.map_size = self.robot1.map_size  # 假設兩個機器人使用相同地圖
        
        # 訓練參數
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰減
        
        # 訓練歷史記錄
        self.training_history = {
            'episode_rewards': [],  # 每一輪的總獎勵
            'episode_lengths': [],  # 每一輪的步數
            'exploration_rates': [],  # 探索率變化
            'losses': [],  # 損失值
            'robot1_rewards': [],  # Robot1的獎勵
            'robot2_rewards': [],  # Robot2的獎勵
            'exploration_progress': []  # 探索進度
        }
    
    def remember(self, state, frontiers, robot1_pos, robot2_pos, 
                robot1_action, robot2_action, robot1_reward, robot2_reward,
                next_state, next_frontiers, next_robot1_pos, next_robot2_pos, done):
        """存儲經驗到回放緩衝區
        
        Args:
            state: 當前狀態
            frontiers: 當前可用的frontier點
            robot1_pos: Robot1的位置
            robot2_pos: Robot2的位置
            robot1_action: Robot1選擇的動作
            robot2_action: Robot2選擇的動作
            robot1_reward: Robot1獲得的獎勵
            robot2_reward: Robot2獲得的獎勵
            next_state: 下一個狀態
            next_frontiers: 下一個狀態的frontier點
            next_robot1_pos: 下一個狀態Robot1的位置
            next_robot2_pos: 下一個狀態Robot2的位置
            done: 是否結束
        """
        self.memory.append((
            state, frontiers, robot1_pos, robot2_pos,
            robot1_action, robot2_action, robot1_reward, robot2_reward,
            next_state, next_frontiers, next_robot1_pos, next_robot2_pos, done
        ))
    
    def pad_frontiers(self, frontiers):
        """填充frontier點到固定長度並進行標準化
        
        Args:
            frontiers: 原始frontier點列表
            
        Returns:
            標準化且填充後的frontier數組
        """
        padded = np.zeros((self.model.max_frontiers, 2))
        
        if len(frontiers) > 0:
            frontiers = np.array(frontiers)
            
            # 標準化座標
            normalized_frontiers = frontiers.copy()
            normalized_frontiers[:, 0] = frontiers[:, 0] / float(self.map_size[1])  # x座標
            normalized_frontiers[:, 1] = frontiers[:, 1] / float(self.map_size[0])  # y座標
            
            # 填充
            n_frontiers = min(len(frontiers), self.model.max_frontiers)
            padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
        
        return padded
    
    def choose_actions(self, state, frontiers, robot1_pos, robot2_pos):
        """為兩個機器人選擇動作
        
        Args:
            state: 當前狀態
            frontiers: 可用的frontier點
            robot1_pos: Robot1的位置
            robot2_pos: Robot2的位置
            
        Returns:
            tuple(int, int): (Robot1的動作, Robot2的動作)
        """
        if len(frontiers) == 0:
            return 0, 0
            
        MIN_TARGET_DISTANCE = 50  # 最小目標距離閾值
        
        # epsilon-greedy策略
        if np.random.random() < self.epsilon:
            valid_frontiers1 = list(range(min(self.model.max_frontiers, len(frontiers))))
            valid_frontiers2 = valid_frontiers1.copy()
            
            # 檢查Robot2的當前目標
            if self.robot2.current_target_frontier is not None:
                valid_frontiers1 = [
                    i for i in valid_frontiers1 
                    if np.linalg.norm(frontiers[i] - self.robot2.current_target_frontier) >= MIN_TARGET_DISTANCE
                ]
                
            # 檢查Robot1的當前目標
            if self.robot1.current_target_frontier is not None:
                valid_frontiers2 = [
                    i for i in valid_frontiers2 
                    if np.linalg.norm(frontiers[i] - self.robot1.current_target_frontier) >= MIN_TARGET_DISTANCE
                ]
                
            if not valid_frontiers1:
                valid_frontiers1 = list(range(min(self.model.max_frontiers, len(frontiers))))
            if not valid_frontiers2:
                valid_frontiers2 = list(range(min(self.model.max_frontiers, len(frontiers))))
                
            robot1_action = np.random.choice(valid_frontiers1)
            robot2_action = np.random.choice(valid_frontiers2)
            return robot1_action, robot2_action
        
        # 使用模型預測
        state_batch = np.expand_dims(state, 0)
        frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0)
        robot1_pos_batch = np.expand_dims(robot1_pos, 0)
        robot2_pos_batch = np.expand_dims(robot2_pos, 0)
        
        predictions = self.model.predict(
            state_batch, frontiers_batch, robot1_pos_batch, robot2_pos_batch
        )
        
        valid_frontiers = min(self.model.max_frontiers, len(frontiers))
        robot1_q = predictions['robot1'][0, :valid_frontiers].copy()
        robot2_q = predictions['robot2'][0, :valid_frontiers].copy()
        
        # 根據其他機器人的目標調整Q值
        if self.robot2.current_target_frontier is not None:
            for i in range(valid_frontiers):
                if np.linalg.norm(frontiers[i] - self.robot2.current_target_frontier) < MIN_TARGET_DISTANCE:
                    robot1_q[i] *= 0.0001  # 大幅降低太近的目標的Q值
                    
        if self.robot1.current_target_frontier is not None:
            for i in range(valid_frontiers):
                if np.linalg.norm(frontiers[i] - self.robot1.current_target_frontier) < MIN_TARGET_DISTANCE:
                    robot2_q[i] *= 0.0001  # 大幅降低太近的目標的Q值
        
        robot1_action = np.argmax(robot1_q)
        robot2_action = np.argmax(robot2_q)
        
        return robot1_action, robot2_action

    def train_step(self):
        """執行一步訓練
        
        Returns:
            float: 訓練損失值
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        
        # 準備批次數據
        states = []
        frontiers_batch = []
        robot1_pos_batch = []
        robot2_pos_batch = []
        next_states = []
        next_frontiers_batch = []
        next_robot1_pos_batch = []
        next_robot2_pos_batch = []
        
        for state, frontiers, robot1_pos, robot2_pos, _, _, _, _, \
            next_state, next_frontiers, next_robot1_pos, next_robot2_pos, _ in batch:
            
            if len(state.shape) == 2:
                state = np.expand_dims(state, axis=-1)
            if len(next_state.shape) == 2:
                next_state = np.expand_dims(next_state, axis=-1)
            
            states.append(state)
            frontiers_batch.append(self.pad_frontiers(frontiers))
            robot1_pos_batch.append(robot1_pos)
            robot2_pos_batch.append(robot2_pos)
            
            next_states.append(next_state)
            next_frontiers_batch.append(self.pad_frontiers(next_frontiers))
            next_robot1_pos_batch.append(next_robot1_pos)
            next_robot2_pos_batch.append(next_robot2_pos)
        
        states = np.array(states)
        frontiers_batch = np.array(frontiers_batch)
        robot1_pos_batch = np.array(robot1_pos_batch)
        robot2_pos_batch = np.array(robot2_pos_batch)
        next_states = np.array(next_states)
        next_frontiers_batch = np.array(next_frontiers_batch)
        next_robot1_pos_batch = np.array(next_robot1_pos_batch)
        next_robot2_pos_batch = np.array(next_robot2_pos_batch)
        
        # 使用目標網絡計算下一個狀態的Q值
        target_predictions = self.model.target_model.predict({
            'map_input': next_states,
            'frontier_input': next_frontiers_batch,
            'robot1_pos_input': next_robot1_pos_batch,
            'robot2_pos_input': next_robot2_pos_batch
        })
        
        # 使用當前網絡計算當前Q值
        current_predictions = self.model.model.predict({
            'map_input': states,
            'frontier_input': frontiers_batch,
            'robot1_pos_input': robot1_pos_batch,
            'robot2_pos_input': robot2_pos_batch
        })
        
        # 準備訓練目標
        robot1_targets = current_predictions['robot1'].copy()
        robot2_targets = current_predictions['robot2'].copy()
        
        # 更新Q值
        for i, (_, _, _, _, robot1_action, robot2_action, robot1_reward, robot2_reward, 
               _, _, _, _, done) in enumerate(batch):
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
                    
                    
        # print("Robot1 target values:", robot1_targets[0])
        # print("Robot2 target values:", robot2_targets[0])
        
        # 訓練模型
        loss = self.model.model.train_on_batch(
            {
                'map_input': states,
                'frontier_input': frontiers_batch,
                'robot1_pos_input': robot1_pos_batch,
                'robot2_pos_input': robot2_pos_batch
            },
            {
                'robot1': robot1_targets,
                'robot2': robot2_targets
            }
        )
        # print("Training loss details:", loss)
        
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def train(self, episodes=1000000, target_update_freq=10, save_freq=10):
        """執行多機器人協同訓練"""
        try:
            for episode in range(episodes):
                # 初始化環境和狀態
                state = self.robot1.begin()
                self.robot2.begin()
                
                # 初始化episode統計
                total_reward = 0
                robot1_total_reward = 0
                robot2_total_reward = 0
                steps = 0
                episode_losses = []
                
                # 初始化機器人狀態
                robot1_state = state.copy()
                robot2_state = state.copy()
                robot1_target = None
                robot2_target = None
                robot1_action = None
                robot2_action = None
                robot1_in_progress = False  # 表示是否正在執行任務
                robot2_in_progress = False
                
                while not (self.robot1.check_done() or self.robot2.check_done()):
                    frontiers = self.robot1.get_frontiers()
                    if len(frontiers) == 0:
                        break

                    # 獲取當前狀態
                    robot1_pos = self.robot1.get_normalized_position()
                    robot2_pos = self.robot2.get_normalized_position()
                    
                    # Robot 1 目標選擇和移動
                    if not robot1_in_progress:
                        # 只有在沒有進行中的任務時才選擇新目標
                        robot1_action, _ = self.choose_actions(
                            state, frontiers, robot1_pos, robot2_pos
                        )
                        robot1_target = frontiers[robot1_action]
                        robot1_in_progress = True
                    
                    # Robot 2 目標選擇和移動
                    if not robot2_in_progress:
                        # 只有在沒有進行中的任務時才選擇新目標
                        _, robot2_action = self.choose_actions(
                            state, frontiers, robot1_pos, robot2_pos
                        )
                        robot2_target = frontiers[robot2_action]
                        robot2_in_progress = True

                    # 執行移動並獲取獎勵
                    robot1_reward = 0
                    robot2_reward = 0
                    
                    # Robot 1 移動
                    if robot1_in_progress and robot1_target is not None:
                        next_state1, r1, d1 = self.robot1.move_to_frontier(robot1_target)
                        robot1_reward = r1
                        if d1:  # 只有在完成目標時才重置狀態
                            robot1_in_progress = False
                            robot1_target = None
                    else:
                        next_state1 = robot1_state
                    
                    # 更新地圖
                    self.robot2.op_map = self.robot1.op_map.copy()
                    
                    # Robot 2 移動
                    if robot2_in_progress and robot2_target is not None:
                        next_state2, r2, d2 = self.robot2.move_to_frontier(robot2_target)
                        robot2_reward = r2
                        if d2:  # 只有在完成目標時才重置狀態
                            robot2_in_progress = False
                            robot2_target = None
                    else:
                        next_state2 = robot2_state
                    
                    # 確保地圖同步
                    self.robot1.op_map = self.robot2.op_map.copy()
                    
                    # 更新機器人位置信息
                    self.robot1.other_robot_position = self.robot2.robot_position.copy()
                    self.robot2.other_robot_position = self.robot1.robot_position.copy()
                    
                    # 更新狀態
                    robot1_state = next_state1
                    robot2_state = next_state2
                    state = next_state1  # 使用共享地圖狀態
                    
                    # 只有在任務完成時才進行經驗存儲和訓練
                    if not robot1_in_progress or not robot2_in_progress:
                        next_frontiers = self.robot1.get_frontiers()
                        next_robot1_pos = self.robot1.get_normalized_position()
                        next_robot2_pos = self.robot2.get_normalized_position()
                        
                        self.remember(
                            state, frontiers, robot1_pos, robot2_pos,
                            robot1_action if not robot1_in_progress else 0,
                            robot2_action if not robot2_in_progress else 0,
                            robot1_reward, robot2_reward,
                            state, next_frontiers, next_robot1_pos, next_robot2_pos,
                            not robot1_in_progress or not robot2_in_progress
                        )
                        
                        loss = self.train_step()
                        # if loss is not None and isinstance(loss, (int, float)):
                        #     episode_losses.append(loss)
                        if loss is not None:
                            # 如果是列表，取平均值
                            if isinstance(loss, list):
                                episode_losses.append(np.mean(loss))
                            elif isinstance(loss, (int, float)):
                                episode_losses.append(loss)
                    
                    # 更新獎勵統計
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
                
                # Episode結束後的處理（代碼保持不變）
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
                
                if (episode + 1) % target_update_freq == 0:
                    self.model.update_target_model()
                
                if (episode + 1) % save_freq == 0:
                    self.save_checkpoint(episode + 1)
                    self.plot_training_progress()
                
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
                
                if exploration_progress >= self.robot1.finish_percent:
                    print("地圖探索完成！")
                else:
                    print("地圖探索未完成")
                print("-" * 50)
                
                # 準備下一個地圖
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
    
    def plot_training_progress(self):
        """繪製訓練進度圖"""
        fig, axs = plt.subplots(6, 1, figsize=(12, 20))
        
        episodes = range(1, len(self.training_history['episode_rewards']) + 1)
        
        # 繪製總獎勵
        axs[0].plot(episodes, self.training_history['episode_rewards'], color='#2E8B57')  # 深綠色表示總體
        axs[0].set_title('總獎勵')
        axs[0].set_xlabel('輪數')
        axs[0].set_ylabel('獎勵')
        axs[0].grid(True)
        
        # 繪製各機器人獎勵
        axs[1].plot(episodes, self.training_history['robot1_rewards'], 
                    color='#8A2BE2', label='Robot1')  # 紫色
        axs[1].plot(episodes, self.training_history['robot2_rewards'], 
                    color='#FFA500', label='Robot2')  # 橘色
        axs[1].set_title('各機器人獎勵')
        axs[1].set_xlabel('輪數')
        axs[1].set_ylabel('獎勵')
        axs[1].legend()
        axs[1].grid(True)
        
        # 繪製步數
        axs[2].plot(episodes, self.training_history['episode_lengths'], color='#4169E1')  # 藍色
        axs[2].set_title('每輪步數')
        axs[2].set_xlabel('輪數')
        axs[2].set_ylabel('步數')
        axs[2].grid(True)
        
        # 繪製探索率
        axs[3].plot(episodes, self.training_history['exploration_rates'], color='#DC143C')  # 深紅色
        axs[3].set_title('探索率')
        axs[3].set_xlabel('輪數')
        axs[3].set_ylabel('Epsilon')
        axs[3].grid(True)
        
        # 繪製損失
        axs[4].plot(episodes, self.training_history['losses'], color='#2F4F4F')  # 深灰色
        axs[4].set_title('訓練損失')
        axs[4].set_xlabel('輪數')
        axs[4].set_ylabel('損失值')
        axs[4].grid(True)
        
        # 繪製探索進度
        axs[5].plot(episodes, self.training_history['exploration_progress'], color='#228B22')  # 森林綠
        axs[5].set_title('探索進度')
        axs[5].set_xlabel('輪數')
        axs[5].set_ylabel('探索完成率')
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
        model_path = os.path.join(MODEL_DIR, f'multi_robot_model_ep{ep_str}.h5')
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