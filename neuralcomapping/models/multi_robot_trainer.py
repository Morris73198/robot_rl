import tensorflow as tf
import numpy as np
from collections import deque
import time
import os

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
                 num_steps=128, num_envs=8,
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
            max_frontiers=50  # Adjust based on your needs
        )
        
        # Logging
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.episode_rewards = deque(maxlen=10)
        self.total_steps = 0
        
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


    def train(self, num_updates=1000):
        """Main training loop for multi-robot exploration
        
        Args:
            num_updates: Number of training updates to perform
            
        Returns:
            None
        """
        try:
            # Initialize environment
            state = self.robots[0].begin()
            self.robots[1].begin()
            
            print("Starting training...")
            start_time = time.time()
            
            for update in range(num_updates):
                # Collect experience
                for step in range(self.num_steps):
                    # Convert state to tensor and add batch dimension
                    state_tensor = tf.convert_to_tensor(state[None], dtype=tf.float32)
                    
                    # Get current frontiers and robot information
                    frontiers = self.get_frontiers()  # Shape: [num_robots, max_frontiers, 2]
                    robot_poses = self.get_robot_poses()  # Shape: [num_robots, 2]
                    robot_targets = self.get_robot_targets()  # Shape: [num_robots, 2]
                    
                    # Convert to tensors with batch dimension
                    frontiers_tensor = tf.convert_to_tensor(frontiers[None], dtype=tf.float32)
                    robot_poses_tensor = tf.convert_to_tensor(robot_poses[None], dtype=tf.float32)
                    robot_targets_tensor = tf.convert_to_tensor(robot_targets[None], dtype=tf.float32)
                    
                    # Forward pass through network
                    policy_logits, values = self.network(
                        [state_tensor, frontiers_tensor, robot_poses_tensor, robot_targets_tensor],
                        training=False
                    )
                    
                    # Sample actions for each robot
                    actions = []
                    action_probs = []
                    for i in range(self.num_robots):
                        probs = tf.nn.softmax(policy_logits[i][0]).numpy()
                        if len(probs) > 0:  # If there are available frontiers
                            action = np.random.choice(len(probs), p=probs)
                            actions.append(action)
                            action_probs.append(probs[action])
                        else:  # No frontiers available
                            actions.append(0)
                            action_probs.append(1.0)
                    
                    # Execute actions and get rewards
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
                    
                    # Update shared map between robots
                    self.robots[0].op_map = np.maximum(
                        self.robots[0].op_map, self.robots[1].op_map)
                    self.robots[1].op_map = self.robots[0].op_map.copy()
                    
                    # Process values with proper shape handling
                    value_array = np.array([v.numpy() for v in values])  # [num_robots, 1]
                    value_array = np.squeeze(value_array)  # Remove extra dimensions
                    
                    # Store experience
                    self.buffer.store(
                        state, frontiers, robot_poses, robot_targets,
                        np.array(actions), np.array(action_probs),
                        np.array(rewards), value_array, np.array(dones)
                    )
                    
                    # Update state
                    state = next_states[0]  # Use first robot's state
                    self.total_steps += 1
                    
                    # Visualization update (every 10 steps)
                    if self.total_steps % 10 == 0:
                        if self.robots[0].plot:
                            self.robots[0].plot_env()
                        if self.robots[1].plot:
                            self.robots[1].plot_env()
                
                # Get final values for advantage calculation
                final_state_tensor = tf.convert_to_tensor(state[None], dtype=tf.float32)
                final_frontiers = tf.convert_to_tensor(self.get_frontiers()[None], dtype=tf.float32)
                final_poses = tf.convert_to_tensor(self.get_robot_poses()[None], dtype=tf.float32)
                final_targets = tf.convert_to_tensor(self.get_robot_targets()[None], dtype=tf.float32)
                
                _, final_values = self.network(
                    [final_state_tensor, final_frontiers, final_poses, final_targets],
                    training=False
                )
                
                # Process final values with proper shape handling
                final_value_array = np.array([v.numpy() for v in final_values])
                final_value_array = np.squeeze(final_value_array)
                
                # Get stored experience
                buffer_data = self.buffer.get()
                
                # Compute advantages and returns
                advantages, returns = self.compute_advantages(
                    buffer_data, final_value_array
                )
                
                # Perform PPO updates
                total_loss = 0
                policy_loss = 0
                value_loss = 0
                entropy_loss = 0
                
                # Convert data to tensors for training
                states_tensor = tf.convert_to_tensor(buffer_data[0], dtype=tf.float32)
                frontiers_tensor = tf.convert_to_tensor(buffer_data[1], dtype=tf.float32)
                robot_poses_tensor = tf.convert_to_tensor(buffer_data[2], dtype=tf.float32)
                robot_targets_tensor = tf.convert_to_tensor(buffer_data[3], dtype=tf.float32)
                actions_tensor = tf.convert_to_tensor(buffer_data[4], dtype=tf.int32)
                old_probs_tensor = tf.convert_to_tensor(buffer_data[5], dtype=tf.float32)
                advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
                returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
                
                # Multiple epochs of updating
                for epoch in range(self.num_epochs):
                    total_loss_epoch, policy_loss_epoch, value_loss_epoch, entropy_loss_epoch = \
                        self.train_step(
                            states_tensor,
                            frontiers_tensor,
                            robot_poses_tensor,
                            robot_targets_tensor,
                            actions_tensor,
                            old_probs_tensor,
                            advantages_tensor,
                            returns_tensor
                        )
                    
                    total_loss += total_loss_epoch
                    policy_loss += policy_loss_epoch
                    value_loss += value_loss_epoch
                    entropy_loss += entropy_loss_epoch
                
                # Average losses over epochs
                total_loss /= self.num_epochs
                policy_loss /= self.num_epochs
                value_loss /= self.num_epochs
                entropy_loss /= self.num_epochs
                
                # Logging
                if update % 10 == 0:
                    exploration_progress = self.robots[0].get_exploration_progress()
                    
                    print(f"\nUpdate {update}")
                    print(f"Total Loss: {total_loss:.4f}")
                    print(f"Policy Loss: {policy_loss:.4f}")
                    print(f"Value Loss: {value_loss:.4f}")
                    print(f"Entropy Loss: {entropy_loss:.4f}")
                    print(f"Exploration Progress: {exploration_progress:.2%}")
                    
                    with self.summary_writer.as_default():
                        tf.summary.scalar('loss/total', total_loss, step=update)
                        tf.summary.scalar('loss/policy', policy_loss, step=update)
                        tf.summary.scalar('loss/value', value_loss, step=update)
                        tf.summary.scalar('loss/entropy', entropy_loss, step=update)
                        tf.summary.scalar('exploration/progress', exploration_progress, step=update)
                
                # Save model checkpoint
                if update % 100 == 0:
                    self.save_model(f'model_checkpoint_{update}.h5')
                
                # Clear buffer for next update
                self.buffer.clear()
            
            print("\nTraining completed!")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted!")
            self.save_model('model_interrupted.h5')
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise e
            
        finally:
            # Final save
            self.save_model('model_final.h5')
    
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