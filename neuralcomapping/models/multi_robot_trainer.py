import tensorflow as tf
import numpy as np
from collections import deque
import time
import os
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
import random


class RobotVisualizer:
    def __init__(self, robot1, robot2):
        """初始化可視化器"""
        self.robot1 = robot1
        self.robot2 = robot2
        
        # 創建主圖和子圖
        self.fig = plt.figure(figsize=(15, 8))
        self.gs = self.fig.add_gridspec(2, 3)
        
        # 左側: 環境地圖
        self.ax_map = self.fig.add_subplot(self.gs[:, 0:2])
        
        # 右側上方: 探索進度條
        self.ax_progress = self.fig.add_subplot(self.gs[0, 2])
        
        # 右側下方: 路徑長度圖
        self.ax_path = self.fig.add_subplot(self.gs[1, 2])
        
        # 設置互動模式
        plt.ion()
        
        # 初始化顏色方案
        self.colors = {
            'robot1': '#800080',  # 紫色
            'robot2': '#FFA500',  # 橘色
            'frontier': '#FF0000',  # 紅色
            'unexplored': '#808080',  # 灰色
            'explored': '#FFFFFF',  # 白色
            'obstacle': '#000000',  # 黑色
            'path1': '#9370DB',  # 淺紫色
            'path2': '#FFB84D',  # 淺橘色
        }
        
        # 路徑記錄
        self.path_lengths = {'robot1': [], 'robot2': []}
        self.steps = []
        self.current_step = 0

    def update(self):
        """更新可視化"""
        self._update_map()
        self._update_progress()
        self._update_path_lengths()
        
        # 調整布局並刷新
        self.fig.tight_layout()
        plt.pause(0.01)

    def _update_map(self):
        """更新環境地圖視圖"""
        self.ax_map.clear()
        
        # 繪製基礎地圖
        map_data = self.robot1.op_map.copy()
        self.ax_map.imshow(map_data, cmap='gray', origin='upper')
        
        # 繪製frontier點
        frontiers = self.robot1.get_frontiers()
        if len(frontiers) > 0:
            self.ax_map.scatter(frontiers[:, 0], frontiers[:, 1],
                              c=self.colors['frontier'], marker='*',
                              s=100, label='Frontiers')
        
        # 繪製機器人1的路徑和位置
        self.ax_map.plot(self.robot1.xPoint, self.robot1.yPoint,
                        color=self.colors['path1'], linewidth=2,
                        label='Robot1 Path', alpha=0.7)
        self.ax_map.plot(self.robot1.robot_position[0], self.robot1.robot_position[1],
                        'o', color=self.colors['robot1'], markersize=10,
                        label='Robot1')
        
        if self.robot1.current_target_frontier is not None:
            self.ax_map.plot([self.robot1.robot_position[0], self.robot1.current_target_frontier[0]],
                           [self.robot1.robot_position[1], self.robot1.current_target_frontier[1]],
                           '--', color=self.colors['robot1'], alpha=0.5)
        
        # 繪製機器人2的路徑和位置
        self.ax_map.plot(self.robot2.xPoint, self.robot2.yPoint,
                        color=self.colors['path2'], linewidth=2,
                        label='Robot2 Path', alpha=0.7)
        self.ax_map.plot(self.robot2.robot_position[0], self.robot2.robot_position[1],
                        'o', color=self.colors['robot2'], markersize=10,
                        label='Robot2')
        
        if self.robot2.current_target_frontier is not None:
            self.ax_map.plot([self.robot2.robot_position[0], self.robot2.current_target_frontier[0]],
                           [self.robot2.robot_position[1], self.robot2.current_target_frontier[1]],
                           '--', color=self.colors['robot2'], alpha=0.5)
        
        self.ax_map.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
        self.ax_map.set_title('Environment Map')
        self.ax_map.axis('equal')

    def _update_progress(self):
        """更新探索進度條"""
        self.ax_progress.clear()
        
        # 計算探索進度
        progress = self.robot1.get_exploration_progress()
        
        # 繪製進度條
        self.ax_progress.barh(0, progress, color=self.colors['robot1'])
        self.ax_progress.barh(0, 1, color=self.colors['unexplored'], alpha=0.3)
        
        self.ax_progress.set_title('Exploration Progress')
        self.ax_progress.set_xlim(0, 1)
        self.ax_progress.set_ylim(-0.5, 0.5)
        self.ax_progress.text(0.5, 0, f'{progress:.1%}',
                            horizontalalignment='center',
                            verticalalignment='center')
        self.ax_progress.set_yticks([])

    def _update_path_lengths(self):
        """更新路徑長度圖"""
        self.ax_path.clear()
        
        # 計算並記錄路徑長度
        robot1_length = len(self.robot1.xPoint)
        robot2_length = len(self.robot2.xPoint)
        
        self.path_lengths['robot1'].append(robot1_length)
        self.path_lengths['robot2'].append(robot2_length)
        
        self.current_step += 1
        self.steps.append(self.current_step)
        
        self.ax_path.plot(self.steps, self.path_lengths['robot1'],
                         color=self.colors['robot1'], label='Robot1')
        self.ax_path.plot(self.steps, self.path_lengths['robot2'],
                         color=self.colors['robot2'], label='Robot2')
        
        self.ax_path.set_title('Path Length')
        self.ax_path.set_xlabel('Steps')
        self.ax_path.set_ylabel('Length')
        self.ax_path.legend()
        self.ax_path.grid(True)

    def save(self, filename='exploration_visualization.png'):
        """保存當前視圖"""
        self.fig.savefig(filename, bbox_inches='tight', dpi=300)

    def close(self):
        """清理資源"""
        plt.ioff()
        plt.close(self.fig)


class MultiRobotTrainer:
    def __init__(self, network, robots, log_dir,
                num_steps=128, num_envs=1,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                clip_ratio=0.2, value_loss_coef=0.5, entropy_coef=0.01,
                max_grad_norm=0.5, num_epochs=10, 
                memory_size=10000, batch_size=32):
        
        self.network = network
        self.robots = robots
        self.num_robots = len(robots)
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.max_frontiers = 50
        self.log_dir = log_dir  # Store the log_dir
        
        # Training parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Experience buffer
        self.memory = deque(maxlen=memory_size)
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # Setup logging
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.training_history = {
            'episode_rewards': [],
            'robot1_rewards': [],
            'robot2_rewards': [],
            'episode_lengths': [],
            'exploration_rates': [],
            'losses': [],
            'exploration_progress': []
        }
        
        # Initialize visualizer
        self.visualizer = RobotVisualizer(robots[0], robots[1])

    def pad_frontiers(self, frontiers):
        """Pad and normalize frontiers"""
        padded = np.zeros((self.max_frontiers, 2), dtype=np.float32)  # 明確指定float32類型
        if len(frontiers) > 0:
            frontiers = np.array(frontiers, dtype=np.float32)  # 轉換輸入為float32
            n_frontiers = min(len(frontiers), self.max_frontiers)
            # 正規化座標
            normalized_frontiers = frontiers[:n_frontiers].copy()
            normalized_frontiers[:, 0] = frontiers[:n_frontiers, 0] / float(self.robots[0].map_size[1])
            normalized_frontiers[:, 1] = frontiers[:n_frontiers, 1] / float(self.robots[0].map_size[0])
            padded[:n_frontiers] = normalized_frontiers
        return padded

    def get_normalized_target(self, target):
        """Normalize target coordinates"""
        if target is None:
            return np.array([0.0, 0.0])
        return np.array([
            target[0] / float(self.robots[0].map_size[1]),
            target[1] / float(self.robots[0].map_size[0])
        ])

    def choose_actions(self, state, frontiers, robot_poses, robot_targets):
        """Choose actions for both robots"""
        if len(frontiers) == 0:
            return (0, 0)  # Return default actions instead of None

        if np.random.random() < self.epsilon:
            valid_frontiers = list(range(min(self.max_frontiers, len(frontiers))))
            if not valid_frontiers:  # If no valid frontiers
                return (0, 0)
            return (np.random.choice(valid_frontiers), np.random.choice(valid_frontiers))

        # Use network for prediction
        state_tensor = tf.convert_to_tensor(state[None], dtype=tf.float32)
        frontiers_tensor = tf.convert_to_tensor(
            self.pad_frontiers(frontiers)[None], dtype=tf.float32)
        robot_poses_tensor = tf.convert_to_tensor(robot_poses[None], dtype=tf.float32)
        robot_targets_tensor = tf.convert_to_tensor(robot_targets[None], dtype=tf.float32)

        policy_logits, _ = self.network(
            [state_tensor, frontiers_tensor, robot_poses_tensor, robot_targets_tensor],
            training=False
        )

        valid_frontiers = min(self.max_frontiers, len(frontiers))
        robot1_logits = policy_logits[0][0, :valid_frontiers]
        robot2_logits = policy_logits[1][0, :valid_frontiers]

        robot1_action = tf.argmax(robot1_logits).numpy()
        robot2_action = tf.argmax(robot2_logits).numpy()

        return (robot1_action, robot2_action)

    def print_progress(self, episode, num_episodes, total_reward,
                      robot1_reward, robot2_reward, steps, episode_losses):
        """Print training progress"""
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"Map: {self.robots[0].li_map}")
        print(f"Steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Robot1 reward: {robot1_reward:.2f}")
        print(f"Robot2 reward: {robot2_reward:.2f}")
        print(f"Epsilon: {self.epsilon:.3f}")
        print(f"Average loss: {np.mean(episode_losses) if episode_losses else 0:.6f}")
        print(f"Exploration progress: {self.robots[0].get_exploration_progress():.1%}")
        print("-" * 50)

    def plot_training_progress(self):
        """绘制训练进度图"""
        fig, axs = plt.subplots(6, 1, figsize=(12, 20))
        
        episodes = range(1, len(self.training_history['episode_rewards']) + 1)
        
        # 绘制总奖励
        axs[0].plot(episodes, self.training_history['episode_rewards'], color='#4B0082')
        axs[0].set_title('Total Rewards')
        axs[0].set_xlabel('Episodes')
        axs[0].set_ylabel('Reward')
        axs[0].grid(True)
        
        # 绘制各机器人奖励
        axs[1].plot(episodes, self.training_history['robot1_rewards'], 
                    color='#800080', label='Robot1', alpha=0.8)
        axs[1].plot(episodes, self.training_history['robot2_rewards'], 
                    color='#FFA500', label='Robot2', alpha=0.8)
        axs[1].set_title('Robot Rewards')
        axs[1].set_xlabel('Episodes')
        axs[1].set_ylabel('Reward')
        axs[1].legend()
        axs[1].grid(True)
        
        # 绘制步数
        axs[2].plot(episodes, self.training_history['episode_lengths'], color='#4169E1')
        axs[2].set_title('Episode Lengths')
        axs[2].set_xlabel('Episodes')
        axs[2].set_ylabel('Steps')
        axs[2].grid(True)
        
        # 绘制探索率
        axs[3].plot(episodes, self.training_history['exploration_rates'], color='#228B22')
        axs[3].set_title('Exploration Rate')
        axs[3].set_xlabel('Episodes')
        axs[3].set_ylabel('Epsilon')
        axs[3].grid(True)
        
        # 绘制损失
        axs[4].plot(episodes, self.training_history['losses'], color='#B22222')
        axs[4].set_title('Training Loss')
        axs[4].set_xlabel('Episodes')
        axs[4].set_ylabel('Loss')
        axs[4].grid(True)
        
        # 绘制探索进度
        axs[5].plot(episodes, self.training_history['exploration_progress'], color='#2F4F4F')
        axs[5].set_title('Exploration Progress')
        axs[5].set_xlabel('Episodes')
        axs[5].set_ylabel('Progress')
        axs[5].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 额外绘制一个单独的机器人奖励对比图
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, self.training_history['robot1_rewards'],
                color='#800080', label='Robot1', alpha=0.7)
        plt.plot(episodes, self.training_history['robot2_rewards'],
                color='#FFA500', label='Robot2', alpha=0.7)
        plt.fill_between(episodes, self.training_history['robot1_rewards'],
                        alpha=0.3, color='#800080')
        plt.fill_between(episodes, self.training_history['robot2_rewards'],
                        alpha=0.3, color='#FFA500')
        plt.title('Robot Rewards Comparison')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'robots_rewards_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    @tf.function
    def _train_step(self, states, frontiers, robot_poses, robot_targets,
                actions, old_probs, advantages, returns):
        """Execute single training step with type consistency"""
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

            # Convert old_probs to float32 to match network output
            old_probs = tf.cast(old_probs, tf.float32)
            advantages = tf.cast(advantages, tf.float32)
            returns = tf.cast(returns, tf.float32)

            # Calculate losses for each robot
            for i in range(self.num_robots):
                # Get policy logits and probabilities
                logits = policy_logits[i]
                probs = tf.nn.softmax(logits)
                
                # Calculate action probabilities
                action_masks = tf.one_hot(actions[:, i], tf.shape(logits)[1], dtype=tf.float32)
                action_probs = tf.reduce_sum(probs * action_masks, axis=1)
                
                # Calculate ratio and surrogate objectives
                ratio = action_probs / (old_probs[:, i] + 1e-8)
                surrogate1 = ratio * advantages[:, i]
                surrogate2 = tf.clip_by_value(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio
                ) * advantages[:, i]
                
                # Policy loss
                policy_loss += -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                
                # Value loss
                value_pred = tf.cast(values[i], tf.float32)
                value_loss += 0.5 * tf.reduce_mean(
                    tf.square(returns[:, i] - value_pred))
                
                # Entropy loss
                entropy_loss += -tf.reduce_mean(
                    tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1))

            # Combine losses
            total_loss = (policy_loss + 
                        self.value_loss_coef * value_loss -
                        self.entropy_coef * entropy_loss)

        # Compute and apply gradients
        grads = tape.gradient(total_loss, self.network.trainable_variables)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        return total_loss, policy_loss, value_loss, entropy_loss

    def remember(self, state, frontiers, robot_poses, robot_targets,
                actions, rewards, next_state, next_frontiers,
                next_robot_poses, next_robot_targets, done):
        """Store experience in memory"""
        self.memory.append((
            state, frontiers, robot_poses, robot_targets,
            actions, rewards, next_state, next_frontiers,
            next_robot_poses, next_robot_targets, done
        ))

    def train_step(self):
        """Execute training on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = []
        frontiers_batch = []
        robot_poses_batch = []
        robot_targets_batch = []
        actions_batch = []
        rewards_batch = []
        next_states = []
        next_frontiers_batch = []
        next_robot_poses_batch = []
        next_robot_targets_batch = []
        dones = []
        
        for experience in batch:
            (state, frontiers, robot_poses, robot_targets,
             actions, rewards, next_state, next_frontiers,
             next_robot_poses, next_robot_targets, done) = experience
            
            states.append(state)
            frontiers_batch.append(self.pad_frontiers(frontiers))
            robot_poses_batch.append(robot_poses)
            robot_targets_batch.append(robot_targets)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states.append(next_state)
            next_frontiers_batch.append(self.pad_frontiers(next_frontiers))
            next_robot_poses_batch.append(next_robot_poses)
            next_robot_targets_batch.append(next_robot_targets)
            dones.append(done)
            
        # Convert to numpy arrays
        states = np.array(states)
        frontiers_batch = np.array(frontiers_batch)
        robot_poses_batch = np.array(robot_poses_batch)
        robot_targets_batch = np.array(robot_targets_batch)
        actions_batch = np.array(actions_batch)
        rewards_batch = np.array(rewards_batch)
        next_states = np.array(next_states)
        next_frontiers_batch = np.array(next_frontiers_batch)
        next_robot_poses_batch = np.array(next_robot_poses_batch)
        next_robot_targets_batch = np.array(next_robot_targets_batch)
        dones = np.array(dones)
        
        # Convert tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        frontiers_tensor = tf.convert_to_tensor(frontiers_batch, dtype=tf.float32)
        robot_poses_tensor = tf.convert_to_tensor(robot_poses_batch, dtype=tf.float32)
        robot_targets_tensor = tf.convert_to_tensor(robot_targets_batch, dtype=tf.float32)
        
        # Get current policy probabilities and values
        policy_logits, values = self.network(
            [states_tensor, frontiers_tensor, robot_poses_tensor, robot_targets_tensor],
            training=False
        )
        
        # Get next state values
        next_policy_logits, next_values = self.network(
            [
                tf.convert_to_tensor(next_states, dtype=tf.float32),
                tf.convert_to_tensor(next_frontiers_batch, dtype=tf.float32),
                tf.convert_to_tensor(next_robot_poses_batch, dtype=tf.float32),
                tf.convert_to_tensor(next_robot_targets_batch, dtype=tf.float32)
            ],
            training=False
        )
        
        # Calculate advantages and returns
        advantages = np.zeros((self.batch_size, self.num_robots))
        returns = np.zeros((self.batch_size, self.num_robots))
        
        # Calculate for each robot
        for i in range(self.num_robots):
            current_values = values[i].numpy().flatten()
            next_state_values = next_values[i].numpy().flatten()
            robot_rewards = rewards_batch[:, i]
            
            # Calculate TD error and advantages
            delta = (robot_rewards + 
                    self.gamma * next_state_values * (1 - dones) - 
                    current_values)
            advantages[:, i] = delta
            returns[:, i] = current_values + advantages[:, i]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old action probabilities
        old_probs = np.zeros((self.batch_size, self.num_robots))
        for i in range(self.num_robots):
            probs = tf.nn.softmax(policy_logits[i]).numpy()
            for j in range(self.batch_size):
                if actions_batch[j, i] < probs.shape[1]:
                    old_probs[j, i] = probs[j, actions_batch[j, i]]
                else:
                    old_probs[j, i] = 1e-8
        
        # Execute training step
        loss = self._train_step(
            states_tensor,
            frontiers_tensor,
            robot_poses_tensor,
            robot_targets_tensor,
            actions_batch,
            old_probs,
            advantages,
            returns
        )
        
        return loss

    def train(self, num_episodes=1000):
        """Execute training loop"""
        try:
            print("Starting training...")
            start_time = time.time()
            
            for episode in range(num_episodes):
                # Initialize environment
                state = self.robots[0].begin()
                self.robots[1].begin()
                
                # Episode statistics
                total_reward = 0
                robot1_total_reward = 0
                robot2_total_reward = 0
                steps = 0
                episode_losses = []
                
                # Main training loop
                while True:
                    # Get current state info
                    frontiers = self.robots[0].get_frontiers()
                    if len(frontiers) == 0:
                        break
                        
                    robot_poses = np.array([
                        robot.get_normalized_position() for robot in self.robots
                    ])
                    robot_targets = np.array([
                        self.get_normalized_target(robot.current_target_frontier)
                        for robot in self.robots
                    ])
                    
                    # Choose and execute actions
                    robot1_action, robot2_action = self.choose_actions(
                        state, frontiers, robot_poses, robot_targets)
                    
                    # Execute actions and get rewards
                    if len(frontiers) > robot1_action:
                        next_state1, r1, d1 = self.robots[0].move_to_frontier(
                            frontiers[robot1_action])
                        robot1_reward = r1
                    else:
                        next_state1 = state
                        robot1_reward = 0
                        d1 = True
                        
                    # Update shared map
                    self.robots[1].op_map = self.robots[0].op_map.copy()
                    
                    if len(frontiers) > robot2_action:
                        next_state2, r2, d2 = self.robots[1].move_to_frontier(
                            frontiers[robot2_action])
                        robot2_reward = r2
                    else:
                        next_state2 = state
                        robot2_reward = 0
                        d2 = True
                    
                    # Sync maps between robots
                    self.robots[0].op_map = self.robots[1].op_map.copy()
                    
                    # Get next state info
                    next_frontiers = self.robots[0].get_frontiers()
                    next_robot_poses = np.array([
                        robot.get_normalized_position() for robot in self.robots
                    ])
                    next_robot_targets = np.array([
                        self.get_normalized_target(robot.current_target_frontier)
                        for robot in self.robots
                    ])
                    
                    # Store experience
                    if d1 or d2:
                        self.remember(
                            state, frontiers, robot_poses, robot_targets,
                            [robot1_action, robot2_action],
                            [robot1_reward, robot2_reward],
                            next_state1,
                            next_frontiers,
                            next_robot_poses,
                            next_robot_targets,
                            d1 or d2
                        )
                        
                        # Train on batch
                        loss = self.train_step()
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    # Update state and rewards
                    state = next_state1
                    total_reward += (robot1_reward + robot2_reward)
                    robot1_total_reward += robot1_reward
                    robot2_total_reward += robot2_reward
                    steps += 1
                    
                    # Update visualization
                    if steps % 5 == 0:  # 每5步更新一次
                        self.visualizer.update()
                    
                    # Check if exploration is complete
                    if (self.robots[0].get_exploration_progress() > 
                        self.robots[0].finish_percent):
                        # Save final state visualization
                        self.visualizer.save(f'exploration_final_ep{episode+1}.png')
                        break
                
                # End of episode updates
                self.update_training_history(
                    total_reward, robot1_total_reward, robot2_total_reward,
                    steps, episode_losses
                )
                
                # Decay epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                
                # Print progress
                self.print_progress(episode, num_episodes, total_reward,
                                  robot1_total_reward, robot2_total_reward,
                                  steps, episode_losses)
                
                # Save periodic checkpoints and visualizations
                if (episode + 1) % 20 == 0:
                    self.save_checkpoint(episode + 1)
                    self.plot_training_progress()
                    self.visualizer.save(f'exploration_ep{episode+1}.png')
                
                # Reset for next episode
                state = self.robots[0].reset()
                self.robots[1].reset()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted!")
            self.save_checkpoint('interrupted')
            self.visualizer.save('exploration_interrupted.png')
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Clean up
            self.visualizer.close()
            for robot in self.robots:
                if hasattr(robot, 'cleanup_visualization'):
                    robot.cleanup_visualization()
            
            # Save final model and results
            self.save_checkpoint('final')

    def update_training_history(self, total_reward, robot1_reward, robot2_reward,
                              steps, episode_losses):
        """Update training history with episode results"""
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['robot1_rewards'].append(robot1_reward)
        self.training_history['robot2_rewards'].append(robot2_reward)
        self.training_history['episode_lengths'].append(steps)
        self.training_history['exploration_rates'].append(self.epsilon)
        self.training_history['losses'].append(
            np.mean(episode_losses) if episode_losses else 0
        )
        self.training_history['exploration_progress'].append(
            self.robots[0].get_exploration_progress()
        )
        
        
        
    def save_checkpoint(self, identifier):
        """Save model checkpoint and training history"""
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model weights
        model_path = os.path.join(checkpoint_dir, f'model_checkpoint_{identifier}.h5')
        self.network.save(model_path)
        print(f"\nSaved model to: {model_path}")

        # Save training history
        history_path = os.path.join(checkpoint_dir, f'training_history_{identifier}.npz')
        np.savez(
            history_path,
            episode_rewards=self.training_history['episode_rewards'],
            robot1_rewards=self.training_history['robot1_rewards'],
            robot2_rewards=self.training_history['robot2_rewards'],
            episode_lengths=self.training_history['episode_lengths'],
            exploration_rates=self.training_history['exploration_rates'],
            losses=self.training_history['losses'],
            exploration_progress=self.training_history['exploration_progress']
        )
        print(f"Saved training history to: {history_path}")

    def load_checkpoint(self, identifier):
        """Load model checkpoint and training history"""
        checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        
        # Load model weights
        model_path = os.path.join(checkpoint_dir, f'model_checkpoint_{identifier}.h5')
        if os.path.exists(model_path):
            self.network.load(model_path)
            print(f"\nLoaded model from: {model_path}")
        
        # Load training history
        history_path = os.path.join(checkpoint_dir, f'training_history_{identifier}.npz')
        if os.path.exists(history_path):
            data = np.load(history_path)
            self.training_history = {
                'episode_rewards': data['episode_rewards'].tolist(),
                'robot1_rewards': data['robot1_rewards'].tolist(),
                'robot2_rewards': data['robot2_rewards'].tolist(),
                'episode_lengths': data['episode_lengths'].tolist(),
                'exploration_rates': data['exploration_rates'].tolist(),
                'losses': data['losses'].tolist(),
                'exploration_progress': data['exploration_progress'].tolist()
            }
            print(f"Loaded training history from: {history_path}")
    
    
    def save_model(self, path):
        """Save model to specified path"""
        try:
            # Save network weights
            self.network.save(path)
            print(f"\nSaved model to: {path}")
            
            # Save training history
            history_path = path.replace('.h5', '_history.npz')
            np.savez(
                history_path,
                episode_rewards=self.training_history['episode_rewards'],
                robot1_rewards=self.training_history['robot1_rewards'],
                robot2_rewards=self.training_history['robot2_rewards'],
                episode_lengths=self.training_history['episode_lengths'],
                exploration_rates=self.training_history['exploration_rates'],
                losses=self.training_history['losses'],
                exploration_progress=self.training_history['exploration_progress']
            )
            print(f"Saved training history to: {history_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, path):
        """Load model from specified path"""
        try:
            # Load network weights
            self.network.load(path)
            print(f"\nLoaded model from: {path}")
            
            # Try to load training history
            history_path = path.replace('.h5', '_history.npz')
            if os.path.exists(history_path):
                data = np.load(history_path)
                self.training_history = {
                    'episode_rewards': data['episode_rewards'].tolist(),
                    'robot1_rewards': data['robot1_rewards'].tolist(),
                    'robot2_rewards': data['robot2_rewards'].tolist(),
                    'episode_lengths': data['episode_lengths'].tolist(),
                    'exploration_rates': data['exploration_rates'].tolist(),
                    'losses': data['losses'].tolist(),
                    'exploration_progress': data['exploration_progress'].tolist()
                }
                print(f"Loaded training history from: {history_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")