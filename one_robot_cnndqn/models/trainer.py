import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from one_robot_cnndqn.config import MODEL_DIR

class FrontierTrainer:
    def __init__(self, model, robot, memory_size=10000, batch_size=16, gamma=0.99):
        """
        初始化訓練器
        
        參數:
            model: FrontierNetworkModel實例
            robot: Robot實例
            memory_size: 經驗回放緩衝區大小
            batch_size: 訓練批次大小
            gamma: 獎勵折扣因子
        """
        self.model = model
        self.robot = robot
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.map_size = self.robot.map_size

        
        # 訓練參數
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰減
        
        # 訓練歷史記錄
        self.training_history = {
            'episode_rewards': [],  # 每一輪的總獎勵
            'episode_lengths': [],  # 每一輪的步數
            'exploration_rates': [],  # 探索率變化
            'losses': []  # 損失值
        }
    
    def remember(self, state, frontiers, robot_pos, action, reward, next_state, next_frontiers, next_robot_pos, done):
        """
        將經驗存儲到回放緩衝區
        """
        self.memory.append((state, frontiers, robot_pos, action, reward, next_state, next_frontiers, next_robot_pos, done))
    
    def pad_frontiers(self, frontiers):
        """
        將frontier點填充到固定長度，並進行座標標準化
        
        參數:
            frontiers: 原始的frontier點座標列表
            
        返回:
            標準化且填充後的frontier座標數組
        """
        padded = np.zeros((self.model.max_frontiers, 2))
        
        if len(frontiers) > 0:
            # 確保frontiers是numpy數組
            frontiers = np.array(frontiers)
            
            # 標準化座標
            normalized_frontiers = frontiers.copy()
            normalized_frontiers[:, 0] = frontiers[:, 0] / float(self.map_size[1])  # x座標除以地圖寬度
            normalized_frontiers[:, 1] = frontiers[:, 1] / float(self.map_size[0])  # y座標除以地圖高度
            
            # 填充
            n_frontiers = min(len(frontiers), self.model.max_frontiers)
            padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
        
        return padded
    
    def choose_action(self, state, frontiers, robot_pos):
        """
        選擇動作（frontier點）
        """
        if len(frontiers) == 0:
            state = self.robot.reset()
            frontiers = self.robot.get_frontiers()
            robot_pos = self.robot.get_normalized_position()
            
            if len(frontiers) == 0:
                return 0
        
        if np.random.random() < self.epsilon:
            return np.random.randint(min(self.model.max_frontiers, len(frontiers)))
        
        state_batch = np.expand_dims(state, 0)
        frontiers_batch = np.expand_dims(self.pad_frontiers(frontiers), 0)
        robot_pos_batch = np.expand_dims(robot_pos, 0)
        
        q_values = self.model.predict(state_batch, frontiers_batch, robot_pos_batch)
        
        valid_q = q_values[0, :len(frontiers)]
        return np.argmax(valid_q)

    def train_step(self):
        """
        執行一步訓練
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        states = []
        frontiers_batch = []
        robot_pos_batch = []
        next_states = []
        next_frontiers_batch = []
        next_robot_pos_batch = []
        
        for state, frontiers, robot_pos, action, reward, next_state, next_frontiers, next_robot_pos, done in batch:
            if len(state.shape) == 2:
                state = np.expand_dims(state, axis=-1)
            if len(next_state.shape) == 2:
                next_state = np.expand_dims(next_state, axis=-1)
            
            action = min(action, self.model.max_frontiers - 1)
            
            states.append(state)
            robot_pos_batch.append(robot_pos)
            next_robot_pos_batch.append(next_robot_pos)
            
            if len(frontiers) == 0:
                frontiers = np.zeros((1, 2))
            frontiers_batch.append(self.pad_frontiers(frontiers))
            
            next_states.append(next_state)
            if len(next_frontiers) == 0:
                next_frontiers = np.zeros((1, 2))
            next_frontiers_batch.append(self.pad_frontiers(next_frontiers))
        
        states = np.array(states)
        frontiers_batch = np.array(frontiers_batch)
        robot_pos_batch = np.array(robot_pos_batch)
        next_states = np.array(next_states)
        next_frontiers_batch = np.array(next_frontiers_batch)
        next_robot_pos_batch = np.array(next_robot_pos_batch)
        
        target_q = self.model.target_model.predict({
            'map_input': next_states,
            'frontier_input': next_frontiers_batch,
            'robot_pos_input': next_robot_pos_batch
        })
        
        current_q = self.model.model.predict({
            'map_input': states,
            'frontier_input': frontiers_batch,
            'robot_pos_input': robot_pos_batch
        })
        
        for i, (_, _, _, action, reward, _, _, _, done) in enumerate(batch):
            action = min(action, self.model.max_frontiers - 1)
            if done:
                current_q[i][action] = reward
            else:
                current_q[i][action] = reward + self.gamma * np.max(target_q[i])
        
        loss = self.model.model.train_on_batch(
            x={
                'map_input': states,
                'frontier_input': frontiers_batch,
                'robot_pos_input': robot_pos_batch
            },
            y=current_q
        )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def train(self, episodes=1000000, target_update_freq=10, save_freq=10):
        """
        訓練過程
        
        參數:
            episodes: 訓練輪數(地圖數)
            target_update_freq: 目標網絡更新頻率
            save_freq: 保存檢查點頻率
        """
        for episode in range(episodes):
            state = self.robot.begin()  # 開始新地圖
            total_reward = 0
            steps = 0
            episode_losses = []
            
            # 在這個地圖上訓練直到完成為止
            while not self.robot.check_done():  # 使用check_done檢查地圖是否真正探索完成
                frontiers = self.robot.get_frontiers()
                if len(frontiers) == 0:
                    break  # 無可用的frontier點
                    
                # 獲取正規化後的機器人位置
                robot_pos = self.robot.get_normalized_position()
                
                # 選擇動作時包含機器人位置
                action = self.choose_action(state, frontiers, robot_pos)
                selected_frontier = frontiers[action]
                
                # 移動到選定的frontier點
                next_state, reward, move_done = self.robot.move_to_frontier(selected_frontier)
                next_frontiers = self.robot.get_frontiers()
                next_robot_pos = self.robot.get_normalized_position()
                
                # 存儲經驗並訓練
                self.remember(state, frontiers, robot_pos, action, reward, 
                            next_state, next_frontiers, next_robot_pos, move_done)
                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                # 如果這次移動失敗了，繼續嘗試其他frontier點
                if move_done and not self.robot.check_done():
                    continue
            
            # 更新訓練歷史
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_lengths'].append(steps)
            self.training_history['exploration_rates'].append(self.epsilon)
            self.training_history['losses'].append(np.mean(episode_losses) if episode_losses else 0)
            
            # 更新目標網絡
            if (episode + 1) % target_update_freq == 0:
                self.model.update_target_model()
            
            # 保存檢查點
            if (episode + 1) % save_freq == 0:
                self.save_checkpoint(episode + 1)
            
            # 列印訓練信息
            exploration_progress = self.robot.get_exploration_progress()
            print(f"\n第 {episode + 1}/{episodes} 輪 (地圖 {self.robot.li_map})")
            print(f"步數: {steps}, 總獎勵: {total_reward:.2f}")
            print(f"探索率: {self.epsilon:.3f}")
            print(f"平均損失: {self.training_history['losses'][-1]:.6f}")
            print(f"探索進度: {exploration_progress:.1%}")
            
            if exploration_progress >= self.robot.finish_percent:
                print("地圖探索完成！")
            else:
                print("地圖探索未完成")
            print("-" * 50)
            
            # 準備下一個地圖
            state = self.robot.reset()
        
        # 訓練結束後保存最終模型
        self.save_checkpoint(episodes)
    
    def plot_training_progress(self):
        """
        繪製訓練進度圖
        """
        fig, axs = plt.subplots(4, 1, figsize=(10, 15))
        
        # 繪製獎勵
        episodes = range(1, len(self.training_history['episode_rewards']) + 1)
        axs[0].plot(episodes, self.training_history['episode_rewards'])
        axs[0].set_title('每輪獎勵')
        axs[0].set_xlabel('輪數')
        axs[0].set_ylabel('總獎勵')
        
        # 繪製步數
        axs[1].plot(episodes, self.training_history['episode_lengths'])
        axs[1].set_title('每輪步數')
        axs[1].set_xlabel('輪數')
        axs[1].set_ylabel('步數')
        
        # 繪製探索率
        axs[2].plot(episodes, self.training_history['exploration_rates'])
        axs[2].set_title('探索率')
        axs[2].set_xlabel('輪數')
        axs[2].set_ylabel('探索率')
        
        # 繪製損失
        axs[3].plot(episodes, self.training_history['losses'])
        axs[3].set_title('訓練損失')
        axs[3].set_xlabel('輪數')
        axs[3].set_ylabel('損失值')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
    
    def save_training_history(self, filename='training_history.npz'):
        """
        保存訓練歷史
        
        參數:
            filename: 保存文件名
        """
        np.savez(filename, 
                episode_rewards=self.training_history['episode_rewards'],
                episode_lengths=self.training_history['episode_lengths'],
                exploration_rates=self.training_history['exploration_rates'],
                losses=self.training_history['losses'])
    
    def load_training_history(self, filename='training_history.npz'):
        """
        載入訓練歷史
        
        參數:
            filename: 要載入的文件名
        """
        data = np.load(filename)
        self.training_history = {
            'episode_rewards': data['episode_rewards'].tolist(),
            'episode_lengths': data['episode_lengths'].tolist(),
            'exploration_rates': data['exploration_rates'].tolist(),
            'losses': data['losses'].tolist()
        }
        
    def save_checkpoint(self, episode):
        """
        保存檢查點
        
        參數:
            episode: 當前訓練輪數
        """
        # 用零填充確保文件名排序正確
        ep_str = str(episode).zfill(6)
        
        # 保存模型
        model_path = os.path.join(MODEL_DIR, f'frontier_model_ep{ep_str}.h5')
        self.model.save(model_path)
        
        # 保存訓練歷史
        history_path = os.path.join(MODEL_DIR, f'training_history_ep{ep_str}.json')
        history_to_save = {
            'episode_rewards': [float(x) for x in self.training_history['episode_rewards']],
            'episode_lengths': [int(x) for x in self.training_history['episode_lengths']],
            'exploration_rates': [float(x) for x in self.training_history['exploration_rates']],
            'losses': [float(x) if x is not None else 0.0 for x in self.training_history['losses']]
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=4)
        
        print(f"已在第 {episode} 輪保存檢查點")
        
        
        
        
