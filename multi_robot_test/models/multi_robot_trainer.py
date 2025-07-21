import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from multi_robot_test.models.multi_robot_network import MultiRobotDQNNetwork

# from .network import MultiRobotDQNNetwork





# 訓練配置 - 如果config文件不存在，使用默認值
try:
    from ..config import TRAINING_CONFIG, REWARD_CONFIG, ROBOT_CONFIG
except ImportError:
    TRAINING_CONFIG = {
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'gamma': 0.99,
        'target_update_freq': 1000,
        'memory_start_size': 10000
    }
    REWARD_CONFIG = {
        'collision_penalty': -10
    }
    ROBOT_CONFIG = {
        'sensor_range': 30
    }


class MultiRobotDQNTrainer:
    def __init__(self, num_robots=2, state_size=(84, 84, 1), num_actions=8, 
                 learning_rate=0.0001, buffer_size=50000, batch_size=32):
        """
        多機器人DQN訓練器
        
        Args:
            num_robots: 機器人數量
            state_size: 狀態空間大小
            num_actions: 動作空間大小
            learning_rate: 學習率
            buffer_size: 經驗回放緩衝區大小
            batch_size: 批次大小
        """
        self.num_robots = num_robots
        self.state_size = state_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # 訓練參數
        self.epsilon = TRAINING_CONFIG['epsilon_start']
        self.epsilon_min = TRAINING_CONFIG['epsilon_min']
        self.epsilon_decay = TRAINING_CONFIG['epsilon_decay']
        self.gamma = TRAINING_CONFIG['gamma']
        self.target_update_freq = TRAINING_CONFIG['target_update_freq']
        self.memory_start_size = TRAINING_CONFIG['memory_start_size']
        
        # 創建神經網絡
        self.main_network = MultiRobotDQNNetwork(
            num_robots=num_robots,
            state_size=state_size,
            num_actions=num_actions,
            learning_rate=learning_rate
        )
        
        self.target_network = MultiRobotDQNNetwork(
            num_robots=num_robots,
            state_size=state_size,
            num_actions=num_actions,
            learning_rate=learning_rate
        )
        
        # 初始化目標網絡
        self.update_target_network()
        
        # 經驗回放緩衝區
        self.memory = deque(maxlen=buffer_size)
        
        # 訓練統計
        self.training_step = 0
        self.episode_count = 0
        self.losses = []
        
        # 多機器人協調相關
        self.coordination_weight = 0.1
        self.exploration_bonus = 0.05

    def remember(self, states, actions, rewards, next_states, dones):
        """
        存儲經驗到回放緩衝區
        
        Args:
            states: 所有機器人的當前狀態 [num_robots, height, width, channels]
            actions: 所有機器人的動作 [num_robots]
            rewards: 所有機器人的獎勵 [num_robots]
            next_states: 所有機器人的下一狀態 [num_robots, height, width, channels]
            dones: 所有機器人的完成狀態 [num_robots]
        """
        experience = {
            'states': states.copy(),
            'actions': actions.copy(),
            'rewards': rewards.copy(),
            'next_states': next_states.copy(),
            'dones': dones.copy()
        }
        self.memory.append(experience)

    def act(self, states, robot_positions=None, use_epsilon_greedy=True):
        """
        根據當前狀態選擇動作
        
        Args:
            states: 所有機器人的狀態 [num_robots, height, width, channels]
            robot_positions: 機器人位置信息，用於協調 [num_robots, 2]
            use_epsilon_greedy: 是否使用epsilon-greedy探索
            
        Returns:
            actions: 所有機器人的動作 [num_robots]
        """
        if use_epsilon_greedy and np.random.random() <= self.epsilon:
            # 隨機探索
            actions = np.random.randint(0, self.num_actions, size=self.num_robots)
        else:
            # 使用神經網絡預測
            q_values = self.main_network.predict(states, robot_positions)
            actions = np.argmax(q_values, axis=1)
        
        return actions

    def replay(self):
        """執行經驗回放訓練"""
        if len(self.memory) < self.memory_start_size:
            return None
            
        if len(self.memory) < self.batch_size:
            return None
            
        # 從經驗回放緩衝區採樣
        batch = random.sample(self.memory, self.batch_size)
        
        # 準備批次數據
        batch_states = np.array([exp['states'] for exp in batch])
        batch_actions = np.array([exp['actions'] for exp in batch])
        batch_rewards = np.array([exp['rewards'] for exp in batch])
        batch_next_states = np.array([exp['next_states'] for exp in batch])
        batch_dones = np.array([exp['dones'] for exp in batch])
        
        # 重塑數據維度: [batch_size, num_robots, height, width, channels]
        batch_size, num_robots = batch_states.shape[0], batch_states.shape[1]
        
        # 計算目標Q值
        next_q_values = self.target_network.predict_batch(batch_next_states)
        target_q_values = self.main_network.predict_batch(batch_states)
        
        # 更新Q值
        for i in range(batch_size):
            for robot_id in range(num_robots):
                action = batch_actions[i][robot_id]
                reward = batch_rewards[i][robot_id]
                done = batch_dones[i][robot_id]
                
                if done:
                    target_q_values[i][robot_id][action] = reward
                else:
                    target_q = reward + self.gamma * np.max(next_q_values[i][robot_id])
                    target_q_values[i][robot_id][action] = target_q
        
        # 訓練網絡
        loss = self.main_network.train_step(batch_states, target_q_values)
        self.losses.append(loss)
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.training_step += 1
        
        # 定期更新目標網絡
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
            
        return loss

    def update_target_network(self):
        """更新目標網絡權重"""
        self.target_network.model.set_weights(self.main_network.model.get_weights())

    def save_model(self, filepath):
        """保存模型"""
        self.main_network.model.save_weights(f"{filepath}_main.h5")
        self.target_network.model.save_weights(f"{filepath}_target.h5")
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """加載模型"""
        try:
            self.main_network.model.load_weights(f"{filepath}_main.h5")
            self.target_network.model.load_weights(f"{filepath}_target.h5")
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def calculate_coordination_reward(self, robot_positions, robot_actions):
        """
        計算協調獎勵，鼓勵機器人分散探索
        
        Args:
            robot_positions: 機器人位置 [num_robots, 2]
            robot_actions: 機器人動作 [num_robots]
            
        Returns:
            coordination_rewards: 協調獎勵 [num_robots]
        """
        coordination_rewards = np.zeros(self.num_robots)
        
        if robot_positions is None or len(robot_positions) < 2:
            return coordination_rewards
            
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                distance = np.linalg.norm(robot_positions[i] - robot_positions[j])
                
                # 鼓勵適當的距離 - 使用較保守的參數
                optimal_distance = 40.0  # 使用固定值而不是依賴ROBOT_CONFIG
                distance_penalty = abs(distance - optimal_distance) / optimal_distance
                
                coordination_reward = self.coordination_weight * (1 - distance_penalty)
                coordination_rewards[i] += coordination_reward
                coordination_rewards[j] += coordination_reward
        
        return coordination_rewards

    def get_training_stats(self):
        """獲取訓練統計信息"""
        stats = {
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'average_loss': np.mean(self.losses[-100:]) if self.losses else 0
        }
        return stats

    def reset_episode(self):
        """重置回合統計"""
        self.episode_count += 1

    def should_train(self):
        """檢查是否應該開始訓練"""
        return len(self.memory) >= self.memory_start_size

    def get_exploration_action(self, states, robot_positions, exploration_map):
        """
        基於探索地圖的動作選擇
        
        Args:
            states: 機器人狀態
            robot_positions: 機器人位置
            exploration_map: 探索地圖
            
        Returns:
            actions: 動作選擇
        """
        actions = self.act(states, robot_positions, use_epsilon_greedy=True)
        
        # 基於探索獎勵調整動作
        for robot_id in range(self.num_robots):
            if np.random.random() < self.exploration_bonus:
                # 選擇朝向未探索區域的動作
                pos = robot_positions[robot_id]
                best_action = self._find_exploration_action(pos, exploration_map)
                if best_action is not None:
                    actions[robot_id] = best_action
                    
        return actions

    def _find_exploration_action(self, position, exploration_map):
        """
        尋找朝向未探索區域的最佳動作
        
        Args:
            position: 當前位置 [2]
            exploration_map: 探索地圖
            
        Returns:
            best_action: 最佳動作，如果沒有則返回None
        """
        # 8個方向的偏移量
        directions = [
            (0, -1),   # 上
            (1, -1),   # 右上
            (1, 0),    # 右
            (1, 1),    # 右下
            (0, 1),    # 下
            (-1, 1),   # 左下
            (-1, 0),   # 左
            (-1, -1)   # 左上
        ]
        
        best_score = -1
        best_action = None
        movement_step = 5  # 使用固定的移動步長
        
        for action, (dx, dy) in enumerate(directions):
            new_x = int(position[0] + dx * movement_step)
            new_y = int(position[1] + dy * movement_step)
            
            # 檢查邊界
            if (0 <= new_x < exploration_map.shape[1] and 
                0 <= new_y < exploration_map.shape[0]):
                
                # 計算周圍未探索區域的比例
                window_size = 10
                x_start = max(0, new_x - window_size)
                x_end = min(exploration_map.shape[1], new_x + window_size)
                y_start = max(0, new_y - window_size)
                y_end = min(exploration_map.shape[0], new_y + window_size)
                
                window = exploration_map[y_start:y_end, x_start:x_end]
                unexplored_ratio = np.sum(window == 127) / window.size
                
                if unexplored_ratio > best_score:
                    best_score = unexplored_ratio
                    best_action = action
                    
        return best_action if best_score > 0.1 else None