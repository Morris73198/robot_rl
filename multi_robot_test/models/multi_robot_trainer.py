import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from two_robot_dueling_dqn_attention.config import MODEL_DIR, ROBOT_CONFIG, REWARD_CONFIG

class MultiRobotTrainer:
    def __init__(self, model, robots, memory_size=10000, batch_size=16, gamma=0.99):
        """初始化多機器人訓練器
        
        Args:
            model: MultiRobotNetworkModel實例
            robots: 機器人實例列表
            memory_size: 記憶體緩衝區大小
            batch_size: 批次大小
            gamma: 折扣因子
        """
        self.model = model
        self.robots = robots
        self.num_robots = len(robots)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # 確保機器人數量不超過模型支援的最大數量
        if self.num_robots > self.model.max_robots:
            raise ValueError(f"Number of robots ({self.num_robots}) exceeds model capacity ({self.model.max_robots})")
        
        # 獲取地圖大小（假設所有機器人共享相同地圖）
        self.map_size = self.robots[0].map_size
        
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
            'robot_rewards': {f'robot{i}': [] for i in range(self.num_robots)},
            'exploration_progress': []
        }
        
        # 創建機器人地圖追蹤器（如果可用）
        self.map_tracker = None
        try:
            from two_robot_dueling_dqn_attention.environment.robot_local_map_tracker import RobotIndividualMapTracker
            if self.num_robots >= 2:  # 地圖追蹤器至少需要2個機器人
                self.map_tracker = RobotIndividualMapTracker(self.robots[0], self.robots[1])
        except ImportError:
            print("Warning: RobotIndividualMapTracker not available")
    
    def remember(self, state, frontiers, robots_poses, robots_targets,
                robot_actions, robot_rewards,
                next_state, next_frontiers, next_robots_poses, next_robots_targets, done):
        """存儲經驗到回放緩衝區
        
        Args:
            state: 當前環境狀態
            frontiers: frontier點座標
            robots_poses: 所有機器人當前位置列表
            robots_targets: 所有機器人當前目標列表
            robot_actions: 所有機器人選擇的動作列表
            robot_rewards: 所有機器人獲得的獎勵列表
            next_state: 下一步環境狀態
            next_frontiers: 下一步frontier點
            next_robots_poses: 下一步機器人位置列表
            next_robots_targets: 下一步機器人目標列表
            done: 是否結束
        """
        self.memory.append((
            state, frontiers, robots_poses, robots_targets,
            robot_actions, robot_rewards,
            next_state, next_frontiers, next_robots_poses, next_robots_targets, done
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
    
    def get_normalized_positions(self, positions):
        """標準化機器人位置"""
        if not positions:
            return []
        
        normalized = []
        for pos in positions:
            if pos is None:
                normalized.append(np.zeros(2))
            else:
                norm_pos = np.array(pos, dtype=np.float32)
                norm_pos[0] = pos[0] / float(self.map_size[1])  # x座標標準化
                norm_pos[1] = pos[1] / float(self.map_size[0])  # y座標標準化
                normalized.append(norm_pos)
        
        return normalized
    
    def get_normalized_targets(self, targets):
        """標準化機器人目標位置"""
        if not targets:
            return []
        
        normalized = []
        map_dims = np.array([float(self.map_size[1]), float(self.map_size[0])])
        
        for target in targets:
            if target is None:
                normalized.append(np.zeros(2))
            else:
                normalized.append(np.array(target) / map_dims)
        
        return normalized
    
    def _calculate_exploration_progress(self):
        """計算探索進度的備用方法"""
        try:
            # 使用第一個機器人的探索信息
            robot = self.robots[0]
            
            # 計算已探索的區域比例
            if hasattr(robot, 'op_map') and hasattr(robot, 'global_map'):
                # op_map 中已知的區域（非127的區域）
                total_pixels = robot.global_map.size
                unknown_pixels = np.sum(robot.op_map == 127)
                known_pixels = total_pixels - unknown_pixels
                
                # 計算探索進度
                exploration_progress = known_pixels / total_pixels if total_pixels > 0 else 0.0
                return min(exploration_progress, 1.0)  # 確保不超過1.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"計算探索進度時出錯: {str(e)}")
            return 0.0
    
    def select_actions(self, state, frontiers, num_valid_frontiers):
        """為所有機器人選擇動作"""
        actions = []
        
        # 收集所有機器人的位置和目標
        robots_poses = []
        robots_targets = []
        
        for robot in self.robots:
            robots_poses.append(robot.get_normalized_position())
            
            # 獲取標準化目標
            if robot.current_target_frontier is None:
                robots_targets.append(np.zeros(2))
            else:
                map_dims = np.array([float(self.map_size[1]), float(self.map_size[0])])
                robots_targets.append(robot.current_target_frontier / map_dims)
        
        # 預測動作值
        if np.random.random() <= self.epsilon:
            # 探索：隨機選擇動作
            for _ in range(self.num_robots):
                actions.append(np.random.choice(num_valid_frontiers))
        else:
            # 利用：使用模型預測
            predictions = self.model.predict(
                np.expand_dims(state, 0),
                np.expand_dims(self.pad_frontiers(frontiers), 0),
                robots_poses,
                robots_targets,
                self.num_robots
            )
            
            # 為每個機器人選擇最佳動作
            for i in range(self.num_robots):
                robot_q_values = predictions[f'robot{i}'][0, :num_valid_frontiers]
                actions.append(np.argmax(robot_q_values))
        
        return actions
    
    def train_step(self):
        """執行一步訓練"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 從記憶體中採樣批次資料
        batch = random.sample(self.memory, self.batch_size)
        
        # 準備批次資料
        states = []
        frontiers_batch = []
        robots_poses_batch = []
        robots_targets_batch = []
        robot_actions_batch = {f'robot{i}': [] for i in range(self.model.max_robots)}
        robot_rewards_batch = {f'robot{i}': [] for i in range(self.model.max_robots)}
        next_states = []
        next_frontiers_batch = []
        next_robots_poses_batch = []
        next_robots_targets_batch = []
        dones = []
        
        for experience in batch:
            (state, frontiers, robots_poses, robots_targets,
             robot_actions, robot_rewards,
             next_state, next_frontiers, next_robots_poses, next_robots_targets, done) = experience
            
            states.append(state)
            frontiers_batch.append(self.pad_frontiers(frontiers))
            
            # 填充當前狀態的機器人位置和目標到max_robots維度
            current_normalized_poses = self.get_normalized_positions(robots_poses)
            current_normalized_targets = self.get_normalized_targets(robots_targets)
            current_padded_poses, current_padded_targets = self.model.pad_robot_states(
                current_normalized_poses, current_normalized_targets, len(current_normalized_poses)
            )
            robots_poses_batch.append(current_padded_poses)
            robots_targets_batch.append(current_padded_targets)
            
            # 存儲每個機器人的動作和獎勵（填充到max_robots）
            for i in range(self.model.max_robots):
                if i < len(robot_actions):
                    robot_actions_batch[f'robot{i}'].append(robot_actions[i])
                    robot_rewards_batch[f'robot{i}'].append(robot_rewards[i])
                else:
                    # 如果機器人數量不足，填充默認值
                    robot_actions_batch[f'robot{i}'].append(0)
                    robot_rewards_batch[f'robot{i}'].append(0.0)
            
            next_states.append(next_state)
            next_frontiers_batch.append(self.pad_frontiers(next_frontiers))
            
            # 填充下一狀態的機器人位置和目標到max_robots維度
            next_normalized_poses = self.get_normalized_positions(next_robots_poses)
            next_normalized_targets = self.get_normalized_targets(next_robots_targets)
            next_padded_poses, next_padded_targets = self.model.pad_robot_states(
                next_normalized_poses, next_normalized_targets, len(next_normalized_poses)
            )
            next_robots_poses_batch.append(next_padded_poses)
            next_robots_targets_batch.append(next_padded_targets)
            
            dones.append(done)
        
        # 轉換為numpy數組
        states = np.array(states)
        frontiers_batch = np.array(frontiers_batch)
        robots_poses_batch = np.array(robots_poses_batch)
        robots_targets_batch = np.array(robots_targets_batch)
        next_states = np.array(next_states)
        next_frontiers_batch = np.array(next_frontiers_batch)
        next_robots_poses_batch = np.array(next_robots_poses_batch)
        next_robots_targets_batch = np.array(next_robots_targets_batch)
        dones = np.array(dones)
        
        # 使用目標網絡計算下一個狀態的Q值
        target_predictions = self.model.target_model.predict({
            'map_input': next_states,
            'frontier_input': next_frontiers_batch,
            'robots_poses_input': next_robots_poses_batch,
            'robots_targets_input': next_robots_targets_batch
        })
        
        # 使用當前網絡計算當前Q值
        current_predictions = self.model.model.predict({
            'map_input': states,
            'frontier_input': frontiers_batch,
            'robots_poses_input': robots_poses_batch,
            'robots_targets_input': robots_targets_batch
        })
        
        # 準備訓練目標
        robot_targets_dict = {}
        
        for robot_id in range(self.model.max_robots):
            robot_key = f'robot{robot_id}'
            current_q_values = current_predictions[robot_key].copy()
            
            for i in range(self.batch_size):
                action = min(robot_actions_batch[robot_key][i], self.model.max_frontiers - 1)
                reward = robot_rewards_batch[robot_key][i]
                
                if dones[i]:
                    current_q_values[i][action] = reward
                else:
                    current_q_values[i][action] = reward + \
                        self.gamma * np.max(target_predictions[robot_key][i])
            
            robot_targets_dict[robot_key] = current_q_values
        
        # 訓練模型
        loss = self.model.train_on_batch(
            states, frontiers_batch,
            robots_poses_batch, robots_targets_batch,
            robot_targets_dict,
            [self.num_robots] * self.batch_size  # 每個樣本的活躍機器人數量
        )
        
        return loss
    
    def train(self, episodes=1000, target_update_freq=10, save_freq=100):
        """執行多機器人協同訓練"""
        try:
            for episode in range(episodes):
                # 初始化環境和狀態
                state = self.robots[0].begin()  # 主機器人初始化環境
                for robot in self.robots[1:]:   # 其他機器人同步開始
                    robot.begin()
                
                # 啟動地圖追蹤器
                if self.map_tracker:
                    self.map_tracker.start_tracking()
                
                # 初始化episode統計
                total_reward = 0
                robot_total_rewards = [0] * self.num_robots
                steps = 0
                episode_losses = []
                
                # 定義最小目標距離
                MIN_TARGET_DISTANCE = self.robots[0].sensor_range * 1.5
                
                while not any(robot.check_done() for robot in self.robots) and steps < 1500:
                    frontiers = self.robots[0].get_frontiers()
                    if len(frontiers) == 0:
                        break
                    
                    # 獲取當前狀態
                    old_positions = [robot.robot_position.copy() for robot in self.robots]
                    
                    # 收集機器人位置和目標
                    robots_poses = [robot.robot_position.copy() for robot in self.robots]
                    robots_targets = []
                    
                    for robot in self.robots:
                        if robot.current_target_frontier is None:
                            robots_targets.append(np.zeros(2))
                        else:
                            robots_targets.append(robot.current_target_frontier.copy())
                    
                    # 選擇動作
                    valid_frontiers = min(self.model.max_frontiers, len(frontiers))
                    actions = self.select_actions(state, frontiers, valid_frontiers)
                    
                    # 執行動作並收集獎勵
                    next_states = []
                    rewards = []
                    dones = []
                    
                    for i, (robot, action) in enumerate(zip(self.robots, actions)):
                        if action < len(frontiers):
                            target = frontiers[action]
                            next_state, reward, done = robot.move_to_frontier(target)
                            
                            # 同步其他機器人的地圖狀態
                            for other_robot in self.robots:
                                if other_robot != robot:
                                    other_robot.op_map = robot.op_map.copy()
                            
                            next_states.append(next_state)
                            rewards.append(reward)
                            dones.append(done)
                            robot_total_rewards[i] += reward
                        else:
                            # 如果動作無效，給予小懲罰
                            next_states.append(state)
                            rewards.append(-0.1)
                            dones.append(False)
                            robot_total_rewards[i] -= 0.1
                    
                    # 更新機器人間的位置信息
                    for i, robot in enumerate(self.robots):
                        robot.other_robots_positions = [
                            other_robot.robot_position.copy() 
                            for j, other_robot in enumerate(self.robots) if j != i
                        ]
                    
                    # 更新地圖追蹤器
                    if self.map_tracker:
                        self.map_tracker.update()
                    
                    # 收集下一步狀態資料
                    next_robots_poses = [robot.robot_position.copy() for robot in self.robots]
                    next_robots_targets = []
                    
                    for robot in self.robots:
                        if robot.current_target_frontier is None:
                            next_robots_targets.append(np.zeros(2))
                        else:
                            next_robots_targets.append(robot.current_target_frontier.copy())
                    
                    # 存儲經驗
                    any_done = any(dones)
                    self.remember(
                        state, frontiers, robots_poses, robots_targets,
                        actions, rewards,
                        next_states[0], frontiers, next_robots_poses, next_robots_targets,
                        any_done
                    )
                    
                    # 訓練模型
                    if any_done:
                        loss = self.train_step()
                        if loss is not None:
                            if isinstance(loss, list):
                                episode_losses.append(np.mean(loss))
                            else:
                                episode_losses.append(loss)
                    
                    # 更新狀態
                    state = next_states[0]
                    total_reward = sum(rewards)
                    steps += 1
                    
                    # 檢查是否探索完成
                    if any(robot.check_done() for robot in self.robots):
                        break
                
                # 記錄訓練歷史
                self.training_history['episode_rewards'].append(total_reward)
                self.training_history['episode_lengths'].append(steps)
                self.training_history['exploration_rates'].append(self.epsilon)
                
                for i in range(self.num_robots):
                    self.training_history['robot_rewards'][f'robot{i}'].append(robot_total_rewards[i])
                
                if episode_losses:
                    self.training_history['losses'].append(np.mean(episode_losses))
                
                # 計算探索進度
                if self.map_tracker:
                    try:
                        # 嘗試使用地圖追蹤器的方法
                        if hasattr(self.map_tracker, 'get_exploration_progress'):
                            exploration_progress = self.map_tracker.get_exploration_progress()
                            self.training_history['exploration_progress'].append(exploration_progress)
                        else:
                            # 如果方法不存在，使用機器人的探索進度
                            exploration_progress = self._calculate_exploration_progress()
                            self.training_history['exploration_progress'].append(exploration_progress)
                    except Exception as e:
                        # 如果計算失敗，記錄警告但繼續訓練
                        print(f"警告: 無法計算探索進度: {str(e)}")
                        exploration_progress = 0.0
                        self.training_history['exploration_progress'].append(exploration_progress)
                
                # 更新epsilon（探索率衰減）
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                # 更新目標網絡
                if episode % target_update_freq == 0:
                    self.model.update_target_model()
                
                # 打印訓練進度
                if episode % 10 == 0:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-10:]) if self.training_history['episode_rewards'] else 0
                    print(f"Episode {episode:4d} | Avg Reward: {avg_reward:7.2f} | "
                          f"Steps: {steps:3d} | Epsilon: {self.epsilon:.3f}")
                    
                    for i in range(self.num_robots):
                        robot_avg_reward = np.mean(self.training_history['robot_rewards'][f'robot{i}'][-10:]) if self.training_history['robot_rewards'][f'robot{i}'] else 0
                        print(f"  Robot{i} Avg Reward: {robot_avg_reward:7.2f}")
                    
                    # 打印探索進度
                    if self.training_history['exploration_progress']:
                        current_exploration = self.training_history['exploration_progress'][-1]
                        print(f"  探索進度: {current_exploration:.1%}")
                
                # 保存檢查點
                if episode % save_freq == 0 and episode > 0:
                    self.save_checkpoint(episode)
                
                # 重置環境
                state = self.robots[0].reset()
                for robot in self.robots[1:]:
                    robot.reset()
            
            # 訓練結束後保存最終模型
            self.save_checkpoint(episodes)
            
        except KeyboardInterrupt:
            print("收到中斷信號，正在保存模型...")
            self.save_checkpoint(episode)
            raise
            
        except Exception as e:
            print(f"訓練過程出現錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 嘗試保存當前進度
            try:
                self.save_checkpoint(episode)
                print("已保存當前訓練進度")
            except:
                print("保存進度失敗")
            
        finally:
            # 清理資源
            for robot in self.robots:
                if hasattr(robot, 'cleanup_visualization'):
                    robot.cleanup_visualization()
            if self.map_tracker:
                if hasattr(self.map_tracker, 'cleanup'):
                    self.map_tracker.cleanup()
                elif hasattr(self.map_tracker, 'stop_tracking'):
                    self.map_tracker.stop_tracking()
    
    def save_checkpoint(self, episode):
        """保存檢查點"""
        try:
            # 確保保存目錄存在
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            
            # 保存模型
            model_path = os.path.join(MODEL_DIR, f'multi_robot_model_episode_{episode}.h5')
            self.model.save(model_path)
            
            # 保存訓練歷史
            history_path = os.path.join(MODEL_DIR, f'training_history_episode_{episode}.json')
            with open(history_path, 'w') as f:
                # 轉換numpy數組為列表以便JSON序列化
                history_to_save = {}
                for key, value in self.training_history.items():
                    if isinstance(value, dict):
                        history_to_save[key] = {k: [float(x) for x in v] for k, v in value.items()}
                    else:
                        history_to_save[key] = [float(x) for x in value]
                json.dump(history_to_save, f, indent=2)
            
            print(f"檢查點已保存: Episode {episode}")
            
        except Exception as e:
            print(f"保存檢查點時出錯: {str(e)}")
    
    def load_checkpoint(self, filepath):
        """載入檢查點"""
        try:
            self.model.load(filepath)
            
            # 嘗試載入訓練歷史
            history_path = filepath.replace('.h5', '_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
            
            print(f"檢查點已載入: {filepath}")
            
        except Exception as e:
            print(f"載入檢查點時出錯: {str(e)}")
    
    def plot_training_progress(self, save_path=None):
        """繪製訓練進度圖"""
        if not self.training_history['episode_rewards']:
            print("沒有訓練歷史資料可供繪製")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 總獎勵趨勢
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Total Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 每個機器人的獎勵趨勢
        for i in range(self.num_robots):
            robot_key = f'robot{i}'
            if robot_key in self.training_history['robot_rewards']:
                axes[0, 1].plot(self.training_history['robot_rewards'][robot_key], 
                               label=f'Robot {i}')
        axes[0, 1].set_title('Individual Robot Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()
        
        # 探索率
        axes[1, 0].plot(self.training_history['exploration_rates'])
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        
        # 損失
        if self.training_history['losses']:
            axes[1, 1].plot(self.training_history['losses'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"訓練進度圖已保存至: {save_path}")
        else:
            plt.show()