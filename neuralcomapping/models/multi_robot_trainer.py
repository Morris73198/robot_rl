import tensorflow as tf
import numpy as np
from collections import deque
import time
import os
import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 後端
import matplotlib.pyplot as plt

class PPOBuffer:
    """PPO 的經驗緩衝區"""
    def __init__(self, num_steps, num_robots, num_envs, obs_shape, max_frontiers):
        # 初始化緩衝區存儲
        self.states = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.float32)
        self.frontiers = np.zeros((num_steps, num_robots, max_frontiers, 2), dtype=np.float32)
        self.robot_poses = np.zeros((num_steps, num_robots, 2), dtype=np.float32)
        self.robot_targets = np.zeros((num_steps, num_robots, 2), dtype=np.float32)
        
        self.actions = np.zeros((num_steps, num_robots), dtype=np.int32)
        self.action_probs = np.zeros((num_steps, num_robots), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_robots), dtype=np.float32)
        self.values = np.zeros((num_steps, num_robots), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_robots), dtype=np.float32)
        
        self.num_steps = num_steps
        self.num_robots = num_robots
        self.num_envs = num_envs
        self.step = 0
        
    def store(self, state, frontiers, robot_poses, robot_targets, 
             actions, action_probs, rewards, values, dones):
        """存儲一個時間步的經驗"""
        # 處理狀態存儲
        if state.shape[0] == 1:
            state = np.repeat(state, self.num_envs, axis=0)
        self.states[self.step] = state
        
        # 確保所有數據都有正確的形狀
        values = np.squeeze(values)  # 去除多餘的維度
        if values.ndim == 1:
            values = values.reshape(-1)  # 確保是一維數組
            
        dones = np.array(dones).reshape(-1)  # 確保是一維數組
        
        # 存儲其他數據
        self.frontiers[self.step] = frontiers
        self.robot_poses[self.step] = robot_poses
        self.robot_targets[self.step] = robot_targets
        self.actions[self.step] = np.array(actions)
        self.action_probs[self.step] = np.array(action_probs)
        self.rewards[self.step] = np.array(rewards)
        self.values[self.step] = values
        self.dones[self.step] = dones
        
        self.step = (self.step + 1) % self.num_steps
        
    def get(self):
        """Get all stored experience"""
        return (self.states, self.frontiers, self.robot_poses, self.robot_targets,
                self.actions, self.action_probs, self.rewards, self.values, self.dones)
                
    def clear(self):
        """Clear the buffer"""
        self.step = 0
        
class MultiRobotTrainer:
    def __init__(self, network, robots, log_dir,
                 num_steps=128, num_envs=1,
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_ratio=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, num_epochs=10):
        
        self.network = network
        self.robots = robots
        self.num_robots = len(robots)
        self.num_steps = num_steps
        self.num_envs = num_envs
        
        # Training parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Initialize experience buffer
        obs_shape = robots[0].get_observation().shape
        self.buffer = PPOBuffer(
            num_steps=num_steps,
            num_robots=self.num_robots,
            num_envs=num_envs,
            obs_shape=obs_shape,
            max_frontiers=50
        )
        
        # Setup logging
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.episode_rewards = deque(maxlen=10)
        self.total_steps = 0
        
        # Setup visualization
        plt.ion()  # Enable interactive mode for visualization
        
    def compute_advantages(self, buffer_data, final_values):
        """Compute GAE advantages
        
        Args:
            buffer_data: Tuple of (states, frontiers, robot_poses, robot_targets,
                                actions, action_probs, rewards, values, dones)
            final_values: Final value estimates for each robot
            
        Returns:
            advantages: Array of shape [num_steps, num_robots]
            returns: Array of shape [num_steps, num_robots]
        """
        # Unpack buffer data
        _, _, _, _, _, _, rewards, values, dones = buffer_data
        
        # Initialize arrays
        advantages = np.zeros_like(rewards)  # [num_steps, num_robots]
        last_gae = np.zeros(self.num_robots)  # [num_robots]
        
        # Ensure proper shapes
        final_values = np.squeeze(final_values)  # Remove extra dimensions
        if final_values.ndim == 0:
            final_values = np.array([final_values])
        
        # GAE calculation
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = final_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae.copy()
        
        # Calculate returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns


    @tf.function
    def train_step(self, states, frontiers, robot_poses, robot_targets, 
               actions, old_probs, advantages, returns):
        """Execute one training step with proper shape handling
        
        Args:
            states: State tensor [batch_size, height, width, channels]
            frontiers: Frontiers tensor [batch_size, num_robots, max_frontiers, 2]
            robot_poses: Robot positions [batch_size, num_robots, 2]
            robot_targets: Target positions [batch_size, num_robots, 2]
            actions: Action indices [batch_size, num_robots]
            old_probs: Old action probabilities [batch_size, num_robots]
            advantages: Advantage estimates [batch_size, num_robots]
            returns: Return estimates [batch_size, num_robots]
        """
        with tf.GradientTape() as tape:
            # Forward pass
            policy_logits, values = self.network(
                [states, frontiers, robot_poses, robot_targets],
                training=True
            )
            
            total_loss = 0
            policy_loss = 0
            value_loss = 0
            entropy_loss = 0
            
            # Calculate losses for each robot
            for i in range(self.num_robots):
                # Get policy logits for current robot
                robot_logits = policy_logits[i]  # [batch_size, num_frontiers]
                num_frontiers = tf.shape(robot_logits)[1]
                
                # Ensure num_frontiers is not None and > 0
                num_frontiers = tf.maximum(num_frontiers, 1)
                
                # Handle actions for this robot
                robot_actions = actions[:, i]  # [batch_size]
                
                # Create one-hot actions with explicit depth
                action_one_hot = tf.one_hot(robot_actions, num_frontiers)  # [batch_size, num_frontiers]
                
                # Calculate probabilities
                probs = tf.nn.softmax(robot_logits)  # [batch_size, num_frontiers]
                
                # Calculate log probabilities for taken actions
                action_probs = tf.reduce_sum(probs * action_one_hot, axis=-1)  # [batch_size]
                log_probs = tf.math.log(action_probs + 1e-10)  # Add small epsilon to avoid log(0)
                
                # Calculate probability ratio
                ratio = tf.exp(log_probs - old_probs[:, i])
                
                # Calculate surrogate objectives
                surrogate1 = ratio * advantages[:, i]
                surrogate2 = tf.clip_by_value(
                    ratio, 
                    1.0 - self.clip_ratio, 
                    1.0 + self.clip_ratio
                ) * advantages[:, i]
                
                # Calculate policy loss
                policy_loss += -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                
                # Calculate value loss
                value_pred = values[i]
                value_loss += 0.5 * tf.reduce_mean(tf.square(returns[:, i] - value_pred))
                
                # Calculate entropy loss for exploration
                entropy_loss += -tf.reduce_mean(
                    tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)
                )
            
            # Combine losses with coefficients
            total_loss = (policy_loss + 
                        self.value_loss_coef * value_loss - 
                        self.entropy_coef * entropy_loss)
        
        # Compute and apply gradients
        grads = tape.gradient(total_loss, self.network.trainable_variables)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        
        return total_loss, policy_loss, value_loss, entropy_loss


    def train(self, num_episodes=1000):
        """基於完整地圖探索的訓練循環
        
        Args:
            num_episodes: 要訓練的地圖數量
        """
        try:
            print("開始訓練...")
            start_time = time.time()
            
            # 設置可視化間隔
            VIZ_INTERVAL = 5  # 每隔5步更新一次可視化
            PRINT_INTERVAL = 1  # 每完成一張地圖打印一次進度
            
            # 確保所有機器人的可視化設置正確
            for robot in self.robots:
                robot.plot = True
                if not hasattr(robot, 'fig'):
                    robot.initialize_visualization()
                    
            for episode in range(num_episodes):
                print(f"\n開始探索第 {episode + 1} 張地圖")
                
                # 初始化/重置環境
                state = self.robots[0].reset() if episode > 0 else self.robots[0].begin()
                if episode > 0:
                    self.robots[1].reset()
                else:
                    self.robots[1].begin()
                    
                episode_steps = 0
                episode_rewards = [0] * self.num_robots
                episode_buffer = []
                
                # 探索單張地圖直到完成
                while True:
                    # 前向傳遞和動作選擇
                    state_tensor = tf.convert_to_tensor(state[None], dtype=tf.float32)
                    frontiers = self.get_frontiers()
                    robot_poses = self.get_robot_poses()
                    robot_targets = self.get_robot_targets()
                    
                    frontiers_tensor = tf.convert_to_tensor(frontiers[None], dtype=tf.float32)
                    robot_poses_tensor = tf.convert_to_tensor(robot_poses[None], dtype=tf.float32)
                    robot_targets_tensor = tf.convert_to_tensor(robot_targets[None], dtype=tf.float32)
                    
                    policy_logits, values = self.network(
                        [state_tensor, frontiers_tensor, robot_poses_tensor, robot_targets_tensor],
                        training=False
                    )
                    
                    # 為每個機器人採樣動作
                    actions = []
                    action_probs = []
                    for i in range(self.num_robots):
                        probs = tf.nn.softmax(policy_logits[i][0]).numpy()
                        if len(probs) > 0:
                            action = np.random.choice(len(probs), p=probs)
                            actions.append(action)
                            action_probs.append(probs[action])
                        else:
                            actions.append(0)
                            action_probs.append(1.0)
                    
                    # 執行動作並獲取獎勵
                    rewards = []
                    dones = []
                    next_states = []
                    
                    for i, robot in enumerate(self.robots):
                        robot_frontiers = robot.get_frontiers()
                        if len(robot_frontiers) > 0 and actions[i] < len(robot_frontiers):
                            next_state, reward, done = robot.move_to_frontier(
                                robot_frontiers[actions[i]])
                        else:
                            next_state = state
                            reward = 0.0
                            done = True
                            
                        next_states.append(next_state)
                        rewards.append(reward)
                        dones.append(done)
                        episode_rewards[i] += reward
                    
                    # 更新機器人之間共享的地圖
                    self.robots[0].op_map = np.maximum(
                        self.robots[0].op_map, self.robots[1].op_map)
                    self.robots[1].op_map = self.robots[0].op_map.copy()
                    
                    # 存儲當前步驟的經驗
                    value_array = np.array([v[0].numpy() for v in values])
                    value_array = np.squeeze(value_array)
                    step_data = (state, frontiers, robot_poses, robot_targets,
                            np.array(actions), np.array(action_probs),
                            np.array(rewards), value_array, np.array(dones))
                    episode_buffer.append(step_data)
                    
                    # 更新狀態
                    state = next_states[0]
                    episode_steps += 1
                    self.total_steps += 1
                    
                    # 可視化更新
                    if episode_steps % VIZ_INTERVAL == 0:
                        for robot in self.robots:
                            if hasattr(robot, 'plot_env'):
                                robot.plot_env()
                        plt.pause(0.001)
                    
                    # 檢查是否完成當前地圖
                    exploration_progress = self.robots[0].get_exploration_progress()
                    if exploration_progress > self.robots[0].finish_percent:
                        print(f"地圖探索完成! 進度: {exploration_progress:.2%}")
                        break
                    
                    # 如果步數過多，也結束當前地圖
                    if episode_steps >= 1000:  # 可以調整這個閾值
                        print("達到最大步數限制，結束當前地圖")
                        break
                
                # 在每張地圖完成後進行策略更新
                if len(episode_buffer) > 0:
                    # 將 episode buffer 中的數據轉移到 PPO buffer
                    for step_data in episode_buffer:
                        self.buffer.store(*step_data)
                    
                    # 計算優勢和進行策略更新
                    buffer_data = self.buffer.get()
                    advantages, returns = self.compute_advantages(buffer_data, value_array)
                    
                    # 執行多次策略更新
                    total_loss = 0
                    policy_loss = 0
                    value_loss = 0
                    entropy_loss = 0
                    
                    # 準備訓練數據
                    states_tensor = tf.convert_to_tensor(buffer_data[0], dtype=tf.float32)
                    frontiers_tensor = tf.convert_to_tensor(buffer_data[1], dtype=tf.float32)
                    robot_poses_tensor = tf.convert_to_tensor(buffer_data[2], dtype=tf.float32)
                    robot_targets_tensor = tf.convert_to_tensor(buffer_data[3], dtype=tf.float32)
                    actions_tensor = tf.convert_to_tensor(buffer_data[4], dtype=tf.int32)
                    old_probs_tensor = tf.convert_to_tensor(buffer_data[5], dtype=tf.float32)
                    advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
                    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
                    
                    for epoch in range(self.num_epochs):
                        losses = self.train_step(
                            states_tensor,
                            frontiers_tensor,
                            robot_poses_tensor,
                            robot_targets_tensor,
                            actions_tensor,
                            old_probs_tensor,
                            advantages_tensor,
                            returns_tensor
                        )
                        
                        total_loss += losses[0]
                        policy_loss += losses[1]
                        value_loss += losses[2]
                        entropy_loss += losses[3]
                    
                    # 平均損失
                    total_loss /= self.num_epochs
                    policy_loss /= self.num_epochs
                    value_loss /= self.num_epochs
                    entropy_loss /= self.num_epochs
                
                # 打印每張地圖的完成情況
                if (episode + 1) % PRINT_INTERVAL == 0:
                    elapsed_time = time.time() - start_time
                    print(f"\n完成第 {episode + 1}/{num_episodes} 張地圖")
                    print(f"本張地圖步數: {episode_steps}")
                    print(f"本張地圖獎勵: {np.mean(episode_rewards):.2f}")
                    print(f"總損失: {total_loss:.4f}")
                    print(f"策略損失: {policy_loss:.4f}")
                    print(f"價值損失: {value_loss:.4f}")
                    print(f"熵損失: {entropy_loss:.4f}")
                    print(f"已用時間: {elapsed_time:.1f}秒")
                    
                    # 記錄到 TensorBoard
                    with self.summary_writer.as_default():
                        tf.summary.scalar('episode/steps', episode_steps, step=episode)
                        tf.summary.scalar('episode/mean_reward', np.mean(episode_rewards), step=episode)
                        tf.summary.scalar('loss/total', total_loss, step=episode)
                        tf.summary.scalar('loss/policy', policy_loss, step=episode)
                        tf.summary.scalar('loss/value', value_loss, step=episode)
                        tf.summary.scalar('loss/entropy', entropy_loss, step=episode)
                
                # 保存檢查點
                if (episode + 1) % 10 == 0:
                    self.save_model(f'model_checkpoint_{episode+1}.h5')
                
                # 清理緩衝區
                self.buffer.clear()
                
            print(f"\n訓練完成! 總共完成 {num_episodes} 張地圖")
            
        except KeyboardInterrupt:
            print("\n訓練被中斷!")
            self.save_model('model_interrupted.h5')
            
        except Exception as e:
            print(f"\n訓練過程中出現錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
            
        finally:
            # 最終清理
            plt.ioff()
            for robot in self.robots:
                if hasattr(robot, 'cleanup_visualization'):
                    robot.cleanup_visualization()
            
            # 最終保存
            self.save_model('model_final.h5')
            
    def __del__(self):
        """清理資源"""
        plt.ioff()
        for robot in self.robots:
            if hasattr(robot, 'cleanup_visualization'):
                robot.cleanup_visualization()
    
    def get_frontiers(self):
        """Get current frontiers from all robots with padding"""
        max_frontiers = 50  # Maximum number of frontiers to consider
        frontiers_array = np.zeros((len(self.robots), max_frontiers, 2), dtype=np.float32)
        
        for i, robot in enumerate(self.robots):
            frontiers = robot.get_frontiers()
            if len(frontiers) > 0:
                # Convert to numpy if not already
                frontiers = np.array(frontiers, dtype=np.float32)
                # Pad or truncate to max_frontiers
                n_frontiers = min(len(frontiers), max_frontiers)
                frontiers_array[i, :n_frontiers] = frontiers[:n_frontiers]
                
        return frontiers_array
    
    def get_robot_poses(self):
        """Get normalized robot positions"""
        return np.array([robot.get_normalized_position() for robot in self.robots])
    
    def get_robot_targets(self):
        """Get current robot targets"""
        targets = []
        for robot in self.robots:
            if robot.current_target_frontier is not None:
                targets.append(robot.current_target_frontier)
            else:
                targets.append(np.zeros(2))
        return np.array(targets)
    
    def save_model(self, filepath):
        """Save model weights"""
        print(f"\nSaving model to {filepath}")
        self.network.save_weights(filepath)
    
    def load_model(self, filepath):
        """Load model weights"""
        print(f"\nLoading model from {filepath}")
        self.network.load_weights(filepath)