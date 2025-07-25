import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from multi_robot_test.config import MODEL_DIR, ROBOT_CONFIG, REWARD_CONFIG

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
        """存儲經驗到回放緩衝區"""
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
        """計算探索進度的正確方法"""
        try:
            robot = self.robots[0]
            
            if hasattr(robot, 'op_map') and hasattr(robot, 'global_map'):
                # 正確的計算方法：已探索的自由空間 / 總自由空間
                explored_free_space = np.sum(robot.op_map == 255)  # 已探索的自由區域
                total_free_space = np.sum(robot.global_map == 255)  # 總自由區域
                
                if total_free_space > 0:
                    exploration_progress = explored_free_space / total_free_space
                    return min(exploration_progress, 1.0)
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"計算探索進度時出錯: {str(e)}")
            return 0.0
    
    def _analyze_exploration_status(self):
        """分析探索狀態，診斷為什麼提前結束"""
        try:
            robot = self.robots[0]
            
            # 基本統計
            total_pixels = robot.global_map.size
            free_pixels = np.sum(robot.global_map == 255)  # 總自由空間
            obstacle_pixels = np.sum(robot.global_map == 0)  # 障礙物
            
            explored_free = np.sum(robot.op_map == 255)  # 已探索自由空間
            unknown_pixels = np.sum(robot.op_map == 127)  # 未知區域
            explored_obstacles = np.sum(robot.op_map == 0)  # 已知障礙物
            
            frontiers = robot.get_frontiers()
            
            print(f"\n探索狀態分析:")
            print(f"  地圖總像素: {total_pixels}")
            print(f"  總自由空間: {free_pixels} ({free_pixels/total_pixels:.1%})")
            print(f"  總障礙物: {obstacle_pixels} ({obstacle_pixels/total_pixels:.1%})")
            print(f"  已探索自由空間: {explored_free} ({explored_free/free_pixels:.1%})")
            print(f"  未知區域: {unknown_pixels} ({unknown_pixels/total_pixels:.1%})")
            print(f"  可用frontier數: {len(frontiers)}")
            
            # 診斷問題
            exploration_ratio = explored_free / free_pixels if free_pixels > 0 else 0
            
            if len(frontiers) == 0 and exploration_ratio < 0.9:
                print(f"  問題: 無frontier但探索度只有{exploration_ratio:.1%}")
                print(f"       可能原因: 剩餘區域被障礙物隔離")
            elif exploration_ratio >= robot.finish_percent:
                print(f"  正常: 探索度{exploration_ratio:.1%}已達到閾值{robot.finish_percent:.1%}")
            
            return {
                'total_free_space': free_pixels,
                'explored_free_space': explored_free,
                'exploration_ratio': exploration_ratio,
                'unknown_pixels': unknown_pixels,
                'frontier_count': len(frontiers)
            }
            
        except Exception as e:
            print(f"分析探索狀態時出錯: {str(e)}")
            return None
    
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
        """執行多機器人協同訓練 - 傳統方式：每個episode結束後訓練"""
        try:
            for episode in range(episodes):
                # 重置環境 - 確保每個episode都是新的地圖
                print(f"\n=== Episode {episode+1:4d} ===")
                
                # 完全重置所有機器人環境
                state = self.robots[0].reset()  # 主機器人重置環境（新地圖）
                for robot in self.robots[1:]:   # 其他機器人同步重置
                    robot.reset()
                
                # 重新開始環境
                state = self.robots[0].begin()  # 開始新episode
                for robot in self.robots[1:]:   
                    robot.begin()
                
                # 啟動地圖追蹤器（只啟動一次）
                if self.map_tracker and episode == 0:
                    print("初始化地圖追蹤器...")
                elif self.map_tracker:
                    # 重置追蹤器狀態
                    if hasattr(self.map_tracker, 'reset'):
                        self.map_tracker.reset()
                    else:
                        self.map_tracker.start_tracking()
                
                # 初始化episode統計
                total_reward = 0
                robot_total_rewards = [0] * self.num_robots
                steps = 0
                episode_step_rewards = []  # 記錄每步的總獎勵
                
                # 定義最小目標距離
                MIN_TARGET_DISTANCE = self.robots[0].sensor_range * 1.5
                
                # ===== Episode步驟循環 =====
                while not any(robot.check_done() for robot in self.robots) and steps < 1500:
                    frontiers = self.robots[0].get_frontiers()
                    if len(frontiers) == 0:
                        print(f"  Episode {episode+1}: 沒有frontier，結束探索")
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
                        try:
                            self.map_tracker.update()
                        except Exception as e:
                            # 如果地圖追蹤器出錯，不影響訓練
                            pass
                    
                    # 收集下一步狀態資料
                    next_robots_poses = [robot.robot_position.copy() for robot in self.robots]
                    next_robots_targets = []
                    
                    for robot in self.robots:
                        if robot.current_target_frontier is None:
                            next_robots_targets.append(np.zeros(2))
                        else:
                            next_robots_targets.append(robot.current_target_frontier.copy())
                    
                    # 存儲經驗（但不訓練）
                    any_done = any(dones)
                    self.remember(
                        state, frontiers, robots_poses, robots_targets,
                        actions, rewards,
                        next_states[0], frontiers, next_robots_poses, next_robots_targets,
                        any_done
                    )
                    
                    # *** 移除每步的訓練邏輯 ***
                    # 原本的代碼：
                    # if len(self.memory) >= self.batch_size and steps % 10 == 0:
                    #     loss = self.train_step()
                    
                    # 更新狀態
                    state = next_states[0]
                    step_total_reward = sum(rewards)
                    total_reward += step_total_reward
                    episode_step_rewards.append(step_total_reward)
                    steps += 1
                    
                    # 檢查是否探索完成
                    if any(robot.check_done() for robot in self.robots):
                        print(f"  Episode {episode+1}: 探索完成，步數: {steps}")
                        break
                
                # ===== Episode結束後的訓練邏輯 =====
                episode_losses = []
                
                # 只要有足夠的經驗就進行一次訓練（傳統方式）
                if len(self.memory) >= self.batch_size:
                    print(f"  Episode {episode+1}: 開始訓練，記憶體大小: {len(self.memory)}")
                    
                    # 每個episode結束後只訓練一次
                    loss = self.train_step()
                    if loss is not None:
                        if isinstance(loss, list):
                            episode_losses.append(np.mean(loss))
                        else:
                            episode_losses.append(loss)
                    
                    print(f"  Episode {episode+1}: 完成訓練")
                
                # 記錄訓練歷史
                self.training_history['episode_rewards'].append(total_reward)
                self.training_history['episode_lengths'].append(steps)
                self.training_history['exploration_rates'].append(self.epsilon)
                
                for i in range(self.num_robots):
                    self.training_history['robot_rewards'][f'robot{i}'].append(robot_total_rewards[i])
                
                if episode_losses:
                    self.training_history['losses'].append(np.mean(episode_losses))
                
                # 計算探索進度（在episode結束時）
                exploration_progress = 0.0
                
                # 使用機器人自己的探索進度計算方法
                if hasattr(self.robots[0], 'get_exploration_progress'):
                    exploration_progress = self.robots[0].get_exploration_progress()
                else:
                    exploration_progress = self._calculate_exploration_progress()
                
                # 詳細分析探索狀態
                exploration_analysis = self._analyze_exploration_status()
                
                self.training_history['exploration_progress'].append(exploration_progress)
                
                # 更新epsilon（探索率衰減）
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                # 更新目標網絡
                if episode % target_update_freq == 0:
                    self.model.update_target_model()
                    print(f"更新目標網絡")
                
                # 打印訓練進度（增強版）- 修改此部分添加單回合獎勵
                if True:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-10:]) if self.training_history['episode_rewards'] else 0
                    avg_loss = np.mean(episode_losses) if episode_losses else 0
                    
                    # *** 主要修改：添加當前回合獎勵 ***
                    print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:8.2f} | Current Reward: {total_reward:8.2f} | "
                        f"Steps: {steps:3d} | Epsilon: {self.epsilon:.3f} | Loss: {avg_loss:.4f}")
                    
                    # *** 主要修改：為每個機器人添加當前回合獎勵 ***
                    for i in range(self.num_robots):
                        robot_avg_reward = np.mean(self.training_history['robot_rewards'][f'robot{i}'][-10:]) if self.training_history['robot_rewards'][f'robot{i}'] else 0
                        robot_current_reward = robot_total_rewards[i]  # 當前回合該機器人的獎勵
                        print(f"Robot{i} Avg Reward: {robot_avg_reward:8.2f} | Current Reward: {robot_current_reward:8.2f}")
                    
                    print(f"實際探索進度: {exploration_progress:.1%}")
                    print(f"總獎勵: {total_reward:.2f}, 記憶體大小: {len(self.memory)}")
                    
                    # 檢查是否提前結束
                    any_done = any(robot.check_done() for robot in self.robots)
                    if any_done:
                        frontiers = self.robots[0].get_frontiers()
                        if len(frontiers) == 0:
                            print(f"結束原因: 無可到達的frontier點")
                        elif exploration_progress >= getattr(self.robots[0], 'finish_percent', 0.98):
                            print(f"結束原因: 探索完成")
                        else:
                            print(f"結束原因: 其他")
                
                # 保存檢查點
                if episode % save_freq == 0 and episode > 0:
                    self.save_checkpoint(episode)
                
                # 確保清理資源（每個episode結束後）
                for robot in self.robots:
                    if hasattr(robot, 'current_target_frontier'):
                        robot.current_target_frontier = None
                
                # 停止地圖追蹤（如果需要）
                if self.map_tracker and hasattr(self.map_tracker, 'stop_tracking'):
                    try:
                        self.map_tracker.stop_tracking()
                    except:
                        pass
            
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
        """保存檢查點並生成訓練進度圖"""
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
            
            # 生成訓練進度圖
            self.plot_training_progress_in_checkpoint(episode)
            
            print(f"檢查點已保存: Episode {episode}")
            
        except Exception as e:
            print(f"保存檢查點時出錯: {str(e)}")

    def plot_training_progress_in_checkpoint(self, episode):
        """在檢查點保存時生成訓練進度圖（不包含機器人個別獎勵）"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式後端
        
        try:
            # 檢查是否有數據
            if not self.training_history.get('episode_rewards'):
                print(f"Episode {episode}: 沒有訓練歷史數據，跳過圖表生成")
                return
            
            episodes_count = len(self.training_history['episode_rewards'])
            print(f"Episode {episode}: 生成訓練進度圖 (包含 {episodes_count} 個episode的數據)")
            
            # 創建子圖 - 2x2布局，不包含機器人個別獎勵
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Training Progress - Episode {episode} ({episodes_count} episodes)', fontsize=16)
            
            episodes = range(1, episodes_count + 1)
            
            # 1. 總獎勵趨勢
            axes[0, 0].plot(episodes, self.training_history['episode_rewards'], 'b-', linewidth=2)
            axes[0, 0].set_title('Total Episode Rewards', fontsize=14)
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加最新數值標註
            if episodes_count > 0:
                latest_reward = self.training_history['episode_rewards'][-1]
                axes[0, 0].annotate(f'Latest: {latest_reward:.1f}', 
                                xy=(episodes_count, latest_reward),
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                fontsize=10)
            
            # 2. Episode 長度（步數）
            if 'episode_lengths' in self.training_history and self.training_history['episode_lengths']:
                axes[0, 1].plot(episodes, self.training_history['episode_lengths'], 'g-', linewidth=2)
                axes[0, 1].set_title('Episode Lengths (Steps)', fontsize=14)
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Steps')
                axes[0, 1].grid(True, alpha=0.3)
                
                # 添加平均線
                avg_steps = sum(self.training_history['episode_lengths']) / len(self.training_history['episode_lengths'])
                axes[0, 1].axhline(y=avg_steps, color='r', linestyle='--', alpha=0.7, label=f'Avg: {avg_steps:.1f}')
                axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'No episode length data', 
                            ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
                axes[0, 1].set_title('Episode Lengths (No Data)', fontsize=14)
            
            # 3. 探索率 (Epsilon)
            if 'exploration_rates' in self.training_history and self.training_history['exploration_rates']:
                axes[1, 0].plot(episodes, self.training_history['exploration_rates'], 'orange', linewidth=2)
                axes[1, 0].set_title('Exploration Rate (Epsilon)', fontsize=14)
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Epsilon')
                axes[1, 0].grid(True, alpha=0.3)
                
                # 添加當前值標註
                current_epsilon = self.training_history['exploration_rates'][-1]
                axes[1, 0].annotate(f'Current: {current_epsilon:.4f}', 
                                xy=(episodes_count, current_epsilon),
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                                fontsize=10)
            else:
                axes[1, 0].text(0.5, 0.5, 'No exploration rate data', 
                            ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
                axes[1, 0].set_title('Exploration Rate (No Data)', fontsize=14)
            
            # 4. 訓練損失
            if 'losses' in self.training_history and self.training_history['losses']:
                # 過濾有效的損失值
                valid_losses = []
                valid_indices = []
                for i, loss in enumerate(self.training_history['losses']):
                    if loss is not None and not (isinstance(loss, float) and (loss != loss)):  # 檢查 NaN
                        valid_losses.append(loss)
                        valid_indices.append(i + 1)
                
                if valid_losses:
                    axes[1, 1].plot(valid_indices, valid_losses, 'r-', linewidth=2)
                    axes[1, 1].set_title('Training Loss', fontsize=14)
                    axes[1, 1].set_xlabel('Training Step')
                    axes[1, 1].set_ylabel('Loss')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # 添加移動平均線（如果數據點足夠）
                    if len(valid_losses) > 10:
                        window_size = min(20, len(valid_losses) // 5)
                        moving_avg = []
                        for i in range(len(valid_losses)):
                            start_idx = max(0, i - window_size + 1)
                            moving_avg.append(sum(valid_losses[start_idx:i+1]) / (i - start_idx + 1))
                        
                        axes[1, 1].plot(valid_indices, moving_avg, '--', color='darkred', 
                                    alpha=0.8, linewidth=1.5, label=f'Moving Avg ({window_size})')
                        axes[1, 1].legend()
                else:
                    axes[1, 1].text(0.5, 0.5, 'No valid loss data', 
                                ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
                    axes[1, 1].set_title('Training Loss (No Valid Data)', fontsize=14)
            else:
                axes[1, 1].text(0.5, 0.5, 'No loss data', 
                            ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Training Loss (No Data)', fontsize=14)
            
            plt.tight_layout()
            
            # 保存圖片
            plot_path = os.path.join(MODEL_DIR, f'training_progress_episode_{episode}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close('all')  # 關閉圖形釋放內存
            
            # 驗證保存結果
            if os.path.exists(plot_path):
                file_size = os.path.getsize(plot_path)
                print(f"訓練進度圖已保存: {plot_path} ({file_size:,} bytes)")
            else:
                print(f"圖片保存失敗: {plot_path}")
                
        except Exception as e:
            print(f"Episode {episode}: 生成訓練進度圖時出錯: {str(e)}")
            import traceback
            traceback.print_exc()

    
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