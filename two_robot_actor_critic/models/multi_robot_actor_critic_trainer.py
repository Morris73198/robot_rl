import os
import numpy as np
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from two_robot_actor_critic.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR,ROBOT_CONFIG


class MultiRobotActorCriticTrainer:
    def __init__(self, model, robot1, robot2, memory_size=10000, batch_size=32, gamma=0.99):
        """初始化訓練器
        
        Args:
            model: Actor-Critic 模型實例
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
        
        
        
        # 初始化步數計數器
        self.total_steps = 0
        self.episode_steps = 0
        
        # PPO相關超參數
        self.clip_epsilon = 0.2
        self.c1 = 1.0  # 值損失係數
        self.c2 = 0.01  # 熵係數
        self.gae_lambda = 0.95  # GAE係數
        
        # 訓練紀錄
        self.training_history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'actor_losses': [],     
        'critic_losses': [],    
        'robot1_rewards': [],
        'robot2_rewards': [],
        'exploration_progress': [],
        'entropies': [],
        'value_estimates': []
    }
        
        # 創建日誌目錄
        self.log_dir = self._create_log_directory()
        
    def _create_log_directory(self):
        """創建日誌目錄"""
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join('logs', 'multi_robot_ac_' + current_time)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
        
    def remember(self, state, frontiers, robot1_pos, robot2_pos,
                robot1_target, robot2_target, robot1_action, robot2_action,
                robot1_reward, robot2_reward, next_state, next_frontiers,
                next_robot1_pos, next_robot2_pos, next_robot1_target,
                next_robot2_target, done):
        """存儲經驗到回放緩衝區
        
        Args:
            state: 當前狀態
            frontiers: frontier點列表
            robot1_pos: 機器人1位置
            robot2_pos: 機器人2位置
            robot1_target: 機器人1目標
            robot2_target: 機器人2目標
            robot1_action: 機器人1動作
            robot2_action: 機器人2動作
            robot1_reward: 機器人1獎勵
            robot2_reward: 機器人2獎勵
            next_state: 下一狀態
            next_frontiers: 下一步frontier點列表
            next_robot1_pos: 下一步機器人1位置
            next_robot2_pos: 下一步機器人2位置
            next_robot1_target: 下一步機器人1目標
            next_robot2_target: 下一步機器人2目標
            done: 是否結束
        """
        self.memory.append((
            state, frontiers, robot1_pos, robot2_pos,
            robot1_target, robot2_target, robot1_action, robot2_action,
            robot1_reward, robot2_reward, next_state, next_frontiers,
            next_robot1_pos, next_robot2_pos, next_robot1_target,
            next_robot2_target, done
        ))
        
    def compute_advantages(self, rewards, values, next_values, dones):
        """計算優勢函數值
        
        使用通用優勢估計(GAE)方法計算優勢值
        
        Args:
            rewards: 獎勵序列
            values: 當前狀態值估計
            next_values: 下一狀態值估計
            dones: 結束標記序列
            
        Returns:
            advantages: 優勢值
            returns: 目標回報值
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            if dones[t]:
                gae = 0
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            
        # 標準化優勢值
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
        
    def train_batch(self, states, frontiers, robot1_pos, robot2_pos,
                   robot1_target, robot2_target, robot1_action, robot2_action,
                   robot1_advantages, robot2_advantages,
                   robot1_returns, robot2_returns):
        """訓練一個批次
        
        Args:
            states: 狀態批次
            frontiers: frontier點批次
            robot1_pos: 機器人1位置批次
            robot2_pos: 機器人2位置批次
            robot1_target: 機器人1目標批次
            robot2_target: 機器人2目標批次
            robot1_action: 機器人1動作批次
            robot2_action: 機器人2動作批次
            robot1_advantages: 機器人1優勢值批次
            robot2_advantages: 機器人2優勢值批次
            robot1_returns: 機器人1回報值批次
            robot2_returns: 機器人2回報值批次
            
        Returns:
            actor_loss: Actor損失值
            critic_loss: Critic損失值
        """
        # 更新Actor網路
        actor_loss = self.model.actor_model.train_on_batch(
            {
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            },
            {
                'robot1': robot1_advantages,
                'robot2': robot2_advantages
            }
        )
        
        # 更新Critic網路
        critic_loss = self.model.critic_model.train_on_batch(
            {
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            },
            {
                'robot1': robot1_returns,
                'robot2': robot2_returns
            }
        )
        
        self.training_history['actor_losses'].append(float(actor_loss))
        self.training_history['critic_losses'].append(float(critic_loss))
        
        return actor_loss, critic_loss
        
    def train(self, episodes=1000000, save_freq=10):
        """改进的训练循环"""
        try:
            for episode in range(episodes):
                print(f"\n{'='*50}")
                print(f"开始第 {episode + 1} 轮训练")
                
                # 重置环境和状态
                state = self.robot1.reset()  # 使用reset而不是begin
                self.robot2.reset()
                
                print(f"初始状态形状: {state.shape}")
                
                episode_reward = 0
                robot1_episode_reward = 0
                robot2_episode_reward = 0
                self.episode_steps = 0
                
                # 重置episode相关的计数器
                self.robot1.steps = 0
                self.robot2.steps = 0
                exploration_progress = 0
                
                # 主训练循环
                while True:
                    # 检查是否需要结束当前episode
                    if (self.robot1.check_done() or 
                        self.robot2.check_done() or 
                        self.episode_steps >= TRAIN_CONFIG['max_steps_per_episode']):
                        break
                    
                    # 获取当前可用的frontiers
                    frontiers = self.robot1.get_frontiers()
                    if len(frontiers) == 0:
                        print("没有可用的frontiers，结束本轮训练")
                        break
                    
                    # 获取机器人状态
                    robot1_pos = self.robot1.get_normalized_position()
                    robot2_pos = self.robot2.get_normalized_position()
                    robot1_target = self.get_normalized_target(self.robot1)
                    robot2_target = self.get_normalized_target(self.robot2)
                    
                    # 获取动作
                    actions = self.model.get_actions(
                        state, frontiers, robot1_pos, robot2_pos,
                        robot1_target, robot2_target
                    )
                    
                    if actions is None:
                        print("无法获取动作，结束本轮训练")
                        break
                    
                    # 选择和执行动作
                    robot1_action = self.select_action(
                        actions['robot1'][0], 
                        min(len(frontiers), self.model.max_frontiers)
                    )
                    robot2_action = self.select_action(
                        actions['robot2'][0], 
                        min(len(frontiers), self.model.max_frontiers)
                    )
                    
                    # 记录当前探索进度
                    current_progress = self.robot1.get_exploration_progress()
                    exploration_reward = (current_progress - exploration_progress) * 100
                    exploration_progress = current_progress
                    
                    # 执行动作
                    robot1_target = frontiers[robot1_action]
                    robot2_target = frontiers[robot2_action]
                    
                    next_state1, r1, d1 = self.robot1.move_to_frontier(robot1_target)
                    self.robot2.op_map = self.robot1.op_map.copy()  # 同步地图
                    
                    next_state2, r2, d2 = self.robot2.move_to_frontier(robot2_target)
                    self.robot1.op_map = self.robot2.op_map.copy()  # 同步地图
                    
                    # 更新奖励和统计信息
                    r1 += exploration_reward
                    r2 += exploration_reward
                    
                    step_reward = r1 + r2
                    episode_reward += step_reward
                    robot1_episode_reward += r1
                    robot2_episode_reward += r2
                    
                    # 更新状态
                    state = next_state1
                    self.episode_steps += 1
                    self.total_steps += 1
                    
                    # 更新机器人位置信息
                    self.robot1.other_robot_position = self.robot2.robot_position.copy()
                    self.robot2.other_robot_position = self.robot1.robot_position.copy()
                    
                    # 定期输出训练状态
                    if self.episode_steps % 10 == 0:
                        print(f"\n当前训练状态:")
                        print(f"总步数: {self.total_steps}")
                        print(f"本轮步数: {self.episode_steps}")
                        print(f"探索进度: {exploration_progress:.2%}")
                        print(f"累积奖励: {episode_reward:.2f}")
                        print(f"Robot1 奖励: {robot1_episode_reward:.2f}")
                        print(f"Robot2 奖励: {robot2_episode_reward:.2f}")
                    
                    # 更新可视化
                    if self.robot1.plot and self.episode_steps % ROBOT_CONFIG['plot_interval'] == 0:
                        self.robot1.plot_env()
                    if self.robot2.plot and self.episode_steps % ROBOT_CONFIG['plot_interval'] == 0:
                        self.robot2.plot_env()
                
                # 更新训练历史
                self.update_training_history(
                    episode_reward,
                    robot1_episode_reward,
                    robot2_episode_reward,
                    self.episode_steps,
                    exploration_progress
                )
                
                # 输出本轮训练总结
                print(f"\n{'='*20} 第 {episode + 1} 轮训练结束 {'='*20}")
                print(f"总步数: {self.episode_steps}")
                print(f"探索的 Frontiers: {self.episode_steps * 2}")
                print(f"最终探索进度: {exploration_progress:.2%}")
                print(f"总奖励: {episode_reward:.2f}")
                print(f"Robot1 总奖励: {robot1_episode_reward:.2f}")
                print(f"Robot2 总奖励: {robot2_episode_reward:.2f}")
                
                # 定期保存检查点和绘制训练进度
                if (episode + 1) % save_freq == 0:
                    self.save_checkpoint(episode + 1)
                    self.plot_training_progress()
        
        except Exception as e:
            print(f"训练发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理资源
            if hasattr(self.robot1, 'cleanup_visualization'):
                self.robot1.cleanup_visualization()
            if hasattr(self.robot2, 'cleanup_visualization'):
                self.robot2.cleanup_visualization()
                
                
                
    def select_action(self, action_probs, valid_frontiers):
        """Select action with improved probability handling
        
        Args:
            action_probs: Action probability distribution
            valid_frontiers: Number of valid frontiers
            
        Returns:
            Selected action index
        """
        try:
            # Ensure we're only using probabilities for valid frontiers
            probs = action_probs[:valid_frontiers]
            
            # Handle numerical stability
            probs = np.clip(probs, 1e-6, None)
            probs = probs / np.sum(probs)
            
            # Validate probability distribution
            if not np.isclose(np.sum(probs), 1.0):
                print(f"Warning: Invalid probability distribution detected: {probs}")
                # Fall back to uniform distribution
                probs = np.ones(valid_frontiers) / valid_frontiers
            
            # Select action using validated probabilities
            return np.random.choice(valid_frontiers, p=probs)
            
        except Exception as e:
            print(f"Error in action selection: {str(e)}")
            print(f"Probabilities: {probs}")
            print(f"Sum of probabilities: {np.sum(probs)}")
            # Return random action as fallback
            return np.random.randint(valid_frontiers)
                
    def save_checkpoint(self, episode):
        """保存訓練檢查點
        
        Args:
            episode: 當前訓練輪數
        """
        # 建立檢查點目錄
        checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(checkpoint_dir, f'model_ep{episode:06d}')
        self.model.save(model_path)
        
        # 保存訓練歷史
        history_path = os.path.join(checkpoint_dir, f'history_ep{episode:06d}.npz')
        np.savez(history_path, **self.training_history)
        
    def plot_training_progress(self):
        """繪製訓練進度圖"""
        plt.figure(figsize=(15, 20))
        episodes = range(1, len(self.training_history['episode_rewards']) + 1)
        
        # 繪製子圖
        plt.subplot(4, 2, 1)
        plt.plot(episodes, self.training_history['episode_rewards'])
        plt.title('總獎勵')
        plt.xlabel('輪數')
        plt.ylabel('獎勵')
        plt.grid(True)
        
        plt.subplot(4, 2, 2)
        plt.plot(episodes, self.training_history['robot1_rewards'], label='Robot1')
        plt.plot(episodes, self.training_history['robot2_rewards'], label='Robot2')
        plt.title('個別機器人獎勵')
        plt.xlabel('輪數')
        plt.ylabel('獎勵')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(4, 2, 3)
        plt.plot(episodes, self.training_history['actor_losses'])
        plt.title('Actor損失')
        plt.xlabel('輪數')
        plt.ylabel('損失')
        plt.grid(True)
        
        plt.subplot(4, 2, 4)
        plt.plot(episodes, self.training_history['critic_losses'])
        plt.title('Critic損失')
        plt.xlabel('輪數')
        plt.ylabel('損失')
        plt.grid(True)
        
        plt.subplot(4, 2, 5)
        plt.plot(episodes, self.training_history['episode_lengths'])
        plt.title('每輪步數')
        plt.xlabel('輪數')
        plt.ylabel('步數')
        plt.grid(True)
        
        plt.subplot(4, 2, 6)
        plt.plot(episodes, self.training_history['exploration_progress'])
        plt.title('探索進度')
        plt.xlabel('輪數')
        plt.ylabel('進度')
        plt.grid(True)
        
        if 'entropies' in self.training_history and len(self.training_history['entropies']) > 0:
            plt.subplot(4, 2, 7)
            plt.plot(episodes, self.training_history['entropies'])
            plt.title('策略熵')
            plt.xlabel('輪數')
            plt.ylabel('熵')
            plt.grid(True)
        
        if 'value_estimates' in self.training_history and len(self.training_history['value_estimates']) > 0:
            plt.subplot(4, 2, 8)
            plt.plot(episodes, self.training_history['value_estimates'])
            plt.title('值估計')
            plt.xlabel('輪數')
            plt.ylabel('值')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()
    
    
    def get_normalized_target(self, robot):
        """获取归一化的目标位置"""
        if robot.current_target_frontier is None:
            return np.zeros(2)
        return robot.current_target_frontier / np.array([float(robot.map_size[1]), 
                                                    float(robot.map_size[0])])

    def update_training_history(self, episode_reward, robot1_reward, robot2_reward, 
                            steps, progress):
        """更新训练历史记录"""
        self.training_history['episode_rewards'].append(float(episode_reward))
        self.training_history['robot1_rewards'].append(float(robot1_reward))
        self.training_history['robot2_rewards'].append(float(robot2_reward))
        self.training_history['episode_lengths'].append(steps)
        self.training_history['exploration_progress'].append(float(progress))