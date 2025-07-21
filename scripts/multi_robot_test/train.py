import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from collections import deque
import json

from multi_robot_test.environment.multi_robot_no_unknown import Robot
from multi_robot_test.models.multi_robot_trainer import MultiRobotDQNTrainer

# from trainer import MultiRobotDQNTrainer

# 配置 - 如果config文件不存在，使用默認值
try:
    from config import TRAINING_CONFIG, ROBOT_CONFIG
except ImportError:
    TRAINING_CONFIG = {
        'state_size': (84, 84, 1),
        'num_actions': 8,
        'learning_rate': 0.0001,
        'buffer_size': 50000,
        'batch_size': 32,
        'max_episode_steps': 2000
    }
    ROBOT_CONFIG = {
        'movement_step': 5
    }


class MultiRobotTraining:
    def __init__(self, num_robots=2, num_episodes=1000, save_interval=100, 
                 model_save_path='./models', use_attention=True, plot_training=True):
        """
        多機器人訓練系統
        
        Args:
            num_robots: 機器人數量
            num_episodes: 訓練回合數
            save_interval: 模型保存間隔
            model_save_path: 模型保存路徑
            use_attention: 是否使用注意力機制
            plot_training: 是否繪製訓練過程
        """
        self.num_robots = num_robots
        self.num_episodes = num_episodes
        self.save_interval = save_interval
        self.model_save_path = model_save_path
        self.use_attention = use_attention
        self.plot_training = plot_training
        
        # 創建模型保存目錄
        os.makedirs(model_save_path, exist_ok=True)
        
        # 初始化訓練器
        self.trainer = MultiRobotDQNTrainer(
            num_robots=num_robots,
            state_size=TRAINING_CONFIG.get('state_size', (84, 84, 1)),
            num_actions=TRAINING_CONFIG.get('num_actions', 8),
            learning_rate=TRAINING_CONFIG.get('learning_rate', 0.0001),
            buffer_size=TRAINING_CONFIG.get('buffer_size', 50000),
            batch_size=TRAINING_CONFIG.get('batch_size', 32)
        )
        
        # 訓練歷史記錄
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'exploration_progress': [],
            'losses': [],
            'epsilon_values': [],
            'coordination_scores': []
        }
        
        # 為每個機器人單獨記錄
        for i in range(num_robots):
            self.training_history[f'robot{i}_rewards'] = []
            self.training_history[f'robot{i}_steps'] = []
            self.training_history[f'robot{i}_exploration'] = []
        
        # 機器人環境
        self.robots = None
        
        # 動作映射 (8個方向) - 使用固定移動步長
        self.movement_step = 5  # 設定固定移動步長，避免依賴ROBOT_CONFIG
        self.action_map = {
            0: np.array([0, -self.movement_step]),    # 上
            1: np.array([self.movement_step, -self.movement_step]),  # 右上
            2: np.array([self.movement_step, 0]),     # 右
            3: np.array([self.movement_step, self.movement_step]),   # 右下
            4: np.array([0, self.movement_step]),     # 下
            5: np.array([-self.movement_step, self.movement_step]),  # 左下
            6: np.array([-self.movement_step, 0]),    # 左
            7: np.array([-self.movement_step, -self.movement_step]) # 左上
        }

    def initialize_environment(self, map_index=0, train=True, plot=False):
        """初始化多機器人環境"""
        print(f"Initializing environment with {self.num_robots} robots...")
        
        # 創建共享環境的機器人
        self.robots = Robot.create_shared_robots(
            index_map=map_index,
            num_robots=self.num_robots,
            train=train,
            plot=plot
        )
        
        print(f"Environment initialized with {len(self.robots)} robots")

    def get_states(self):
        """獲取所有機器人的狀態"""
        states = []
        for robot in self.robots:
            state = robot.get_observation()
            states.append(state)
        return np.array(states)

    def get_positions(self):
        """獲取所有機器人的位置"""
        positions = []
        for robot in self.robots:
            positions.append(robot.robot_position.copy())
        return np.array(positions)

    def execute_actions(self, actions):
        """執行所有機器人的動作並返回結果"""
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        # 更新機器人位置信息（用於協調）
        current_positions = self.get_positions()
        for i, robot in enumerate(self.robots):
            # 更新其他機器人的位置
            other_positions = [current_positions[j] for j in range(self.num_robots) if j != i]
            robot.other_robots_positions = other_positions
        
        # 執行動作
        for i, (robot, action) in enumerate(zip(self.robots, actions)):
            move_vector = self.action_map[action]
            
            # 檢查是否正在執行frontier移動任務
            if hasattr(robot, 'current_target_frontier') and robot.current_target_frontier is not None:
                # 繼續執行frontier移動
                state, reward, done = robot.move_to_frontier(robot.current_target_frontier)
            else:
                # 執行普通移動
                state, reward, done = robot.execute_movement(move_vector)
            
            next_states.append(state)
            rewards.append(reward)
            
            # 檢查是否完成探索
            exploration_done = robot.check_done()
            dones.append(done or exploration_done)
            
            infos.append({
                'position': robot.robot_position.copy(),
                'exploration_progress': robot.get_exploration_progress(),
                'target_frontier': getattr(robot, 'current_target_frontier', None)
            })
        
        return np.array(next_states), np.array(rewards), np.array(dones), infos

    def calculate_coordination_score(self, positions, rewards):
        """計算協調分數"""
        if len(positions) < 2:
            return 0.0
            
        # 基於距離的協調分數
        total_distance = 0
        count = 0
        optimal_distance = 40.0  # 使用固定值
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                # 計算與最佳距離的偏差
                deviation = abs(distance - optimal_distance) / optimal_distance
                total_distance += 1 - deviation  # 偏差越小分數越高
                count += 1
        
        distance_score = total_distance / count if count > 0 else 0
        
        # 基於獎勵分佈的協調分數
        reward_std = np.std(rewards) if len(rewards) > 1 else 0
        reward_score = max(0, 1 - reward_std / 10.0)  # 獎勵差異越小協調越好
        
        return (distance_score + reward_score) / 2

    def should_select_frontier(self, robot, step_count):
        """決定是否應該選擇新的frontier目標"""
        # 如果沒有當前目標，或者已經到達目標，則選擇新目標
        if not hasattr(robot, 'current_target_frontier') or robot.current_target_frontier is None:
            return True
            
        # 如果卡住了太長時間，重新選擇目標
        if step_count % 100 == 0:  # 每100步檢查一次
            return True
            
        return False

    def select_frontiers_for_robots(self):
        """為所有機器人智能分配frontier目標"""
        robot_targets = {}
        
        for i, robot in enumerate(self.robots):
            # 獲取frontier點 - 直接調用環境的frontier方法
            frontiers = robot.frontier(robot.op_map, robot.map_size, robot.t)
            
            if len(frontiers) == 0:
                robot_targets[i] = None
                continue
            
            # 計算到每個frontier的距離
            distances = np.linalg.norm(frontiers - robot.robot_position, axis=1)
            
            # 考慮其他機器人的目標，避免衝突
            conflict_penalties = np.zeros(len(frontiers))
            for j, other_robot in enumerate(self.robots):
                if i == j:
                    continue
                    
                if hasattr(other_robot, 'current_target_frontier') and other_robot.current_target_frontier is not None:
                    other_distances = np.linalg.norm(frontiers - other_robot.current_target_frontier, axis=1)
                    conflict_penalties += np.maximum(0, 50 - other_distances)  # 距離其他目標太近的懲罰
            
            # 選擇最佳frontier
            scores = -distances - conflict_penalties  # 距離越近分數越高，衝突懲罰
            best_idx = np.argmax(scores)
            robot_targets[i] = frontiers[best_idx]
        
        return robot_targets

    def train_episode(self, episode):
        """執行一個訓練回合"""
        # 重置環境
        if episode == 0:
            self.initialize_environment(map_index=episode % 10, train=True, plot=self.plot_training)
            # 獲取初始狀態
            states = []
            for robot in self.robots:
                state = robot.begin()
                states.append(state)
            states = np.array(states)
        else:
            # 重置到新地圖，但只有主要機器人執行reset
            states = []
            for i, robot in enumerate(self.robots):
                if i == 0:  # 主要機器人
                    state = robot.reset()
                else:  # 次要機器人共享環境
                    state = robot.begin()
                states.append(state)
            states = np.array(states)
        
        positions = self.get_positions()
        
        episode_rewards = np.zeros(self.num_robots)
        episode_steps = np.zeros(self.num_robots)
        step_count = 0
        max_steps = TRAINING_CONFIG.get('max_episode_steps', 2000)
        
        print(f"Episode {episode + 1}/{self.num_episodes}")
        
        while step_count < max_steps:
            # 為機器人分配frontier目標（每隔一定步數）
            if step_count % 50 == 0:  # 每50步重新分配一次目標
                targets = self.select_frontiers_for_robots()
                for i, target in targets.items():
                    if target is not None:
                        self.robots[i].current_target_frontier = target
                        self.robots[i].current_path = None  # 重新規劃路徑
            
            # 選擇動作
            actions = self.trainer.act(states, positions, use_epsilon_greedy=True)
            
            # 執行動作
            next_states, rewards, dones, infos = self.execute_actions(actions)
            next_positions = self.get_positions()
            
            # 添加協調獎勵
            coordination_rewards = self.trainer.calculate_coordination_reward(positions, actions)
            total_rewards = rewards + coordination_rewards
            
            # 存儲經驗
            self.trainer.remember(states, actions, total_rewards, next_states, dones)
            
            # 更新統計
            episode_rewards += total_rewards
            episode_steps += 1
            
            # 更新狀態
            states = next_states
            positions = next_positions
            step_count += 1
            
            # 檢查是否所有機器人都完成了
            if all(dones):
                break
            
            # 訓練網絡
            if self.trainer.should_train() and step_count % 4 == 0:
                loss = self.trainer.replay()
                if loss is not None:
                    self.training_history['losses'].append(loss)
        
        # 記錄回合統計
        total_reward = np.sum(episode_rewards)
        coordination_score = self.calculate_coordination_score(positions, episode_rewards)
        exploration_progress = np.mean([robot.get_exploration_progress() for robot in self.robots])
        
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['episode_lengths'].append(step_count)
        self.training_history['exploration_progress'].append(exploration_progress)
        self.training_history['epsilon_values'].append(self.trainer.epsilon)
        self.training_history['coordination_scores'].append(coordination_score)
        
        # 記錄每個機器人的統計
        for i in range(self.num_robots):
            self.training_history[f'robot{i}_rewards'].append(episode_rewards[i])
            self.training_history[f'robot{i}_steps'].append(episode_steps[i])
            self.training_history[f'robot{i}_exploration'].append(infos[i]['exploration_progress'])
        
        self.trainer.reset_episode()
        
        # 打印統計信息
        if (episode + 1) % 10 == 0:
            recent_rewards = self.training_history['episode_rewards'][-10:]
            recent_progress = self.training_history['exploration_progress'][-10:]
            print(f"Episodes {episode-8}-{episode+1}: "
                  f"Avg Reward: {np.mean(recent_rewards):.2f}, "
                  f"Avg Progress: {np.mean(recent_progress):.2%}, "
                  f"Epsilon: {self.trainer.epsilon:.3f}, "
                  f"Coordination: {coordination_score:.3f}")

    def save_model_and_history(self, episode):
        """保存模型和訓練歷史"""
        # 保存模型
        model_path = os.path.join(self.model_save_path, f'model_episode_{episode+1}')
        self.trainer.save_model(model_path)
        
        # 保存訓練歷史
        history_path = os.path.join(self.model_save_path, f'training_history_{episode+1}.json')
        with open(history_path, 'w') as f:
            # 轉換numpy數組為列表以便JSON序列化
            history_to_save = {}
            for key, value in self.training_history.items():
                if isinstance(value, (list, np.ndarray)):
                    history_to_save[key] = np.array(value).tolist()
                else:
                    history_to_save[key] = value
            json.dump(history_to_save, f, indent=2)
        
        print(f"Model and history saved at episode {episode + 1}")

    def plot_training_progress(self):
        """繪製訓練進度"""
        if len(self.training_history['episode_rewards']) == 0:
            return
            
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{self.num_robots}-Robot Training Progress', fontsize=16)
        
        episodes = range(1, len(self.training_history['episode_rewards']) + 1)
        
        # 總獎勵
        axes[0, 0].plot(episodes, self.training_history['episode_rewards'], 'b-', alpha=0.7)
        axes[0, 0].plot(episodes, self._smooth(self.training_history['episode_rewards']), 'r-', linewidth=2)
        axes[0, 0].set_title('Total Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # 每個機器人的獎勵
        colors = ['purple', 'orange', 'red', 'blue', 'green', 'gold', 'pink', 'cyan']
        for i in range(self.num_robots):
            color = colors[i % len(colors)]
            axes[0, 1].plot(episodes, self.training_history[f'robot{i}_rewards'], 
                           color=color, alpha=0.7, label=f'Robot {i}')
        axes[0, 1].set_title('Individual Robot Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 探索進度
        axes[1, 0].plot(episodes, self.training_history['exploration_progress'], 'g-', alpha=0.7)
        axes[1, 0].plot(episodes, self._smooth(self.training_history['exploration_progress']), 'darkgreen', linewidth=2)
        axes[1, 0].set_title('Exploration Progress')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Progress (%)')
        axes[1, 0].grid(True)
        
        # 協調分數
        axes[1, 1].plot(episodes, self.training_history['coordination_scores'], 'm-', alpha=0.7)
        axes[1, 1].plot(episodes, self._smooth(self.training_history['coordination_scores']), 'darkmagenta', linewidth=2)
        axes[1, 1].set_title('Coordination Score')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True)
        
        # Epsilon值
        axes[2, 0].plot(episodes, self.training_history['epsilon_values'], 'orange')
        axes[2, 0].set_title('Exploration Rate (Epsilon)')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Epsilon')
        axes[2, 0].grid(True)
        
        # 訓練損失
        if self.training_history['losses']:
            loss_episodes = np.linspace(1, len(episodes), len(self.training_history['losses']))
            axes[2, 1].plot(loss_episodes, self.training_history['losses'], 'red', alpha=0.7)
            axes[2, 1].plot(loss_episodes, self._smooth(self.training_history['losses']), 'darkred', linewidth=2)
        axes[2, 1].set_title('Training Loss')
        axes[2, 1].set_xlabel('Training Step')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_progress.png'), dpi=300, bbox_inches='tight')
        if self.plot_training:
            plt.show()
        plt.close()

    def _smooth(self, data, window_size=20):
        """平滑數據用於繪圖"""
        if len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        return smoothed

    def train(self):
        """執行完整的訓練過程"""
        print(f"Starting training with {self.num_robots} robots for {self.num_episodes} episodes")
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            try:
                self.train_episode(episode)
                
                # 保存模型和歷史
                if (episode + 1) % self.save_interval == 0:
                    self.save_model_and_history(episode)
                    self.plot_training_progress()
                
            except KeyboardInterrupt:
                print("Training interrupted by user")
                break
            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
                continue
        
        # 最終保存
        self.save_model_and_history(self.num_episodes - 1)
        self.plot_training_progress()
        
        end_time = time.time()
        print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Multi-Robot DQN Training')
    parser.add_argument('--num_robots', type=int, default=2, help='Number of robots')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--save_interval', type=int, default=100, help='Model save interval')
    parser.add_argument('--model_save_path', type=str, default='./models', help='Model save path')
    parser.add_argument('--use_attention', action='store_true', help='Use attention mechanism')
    parser.add_argument('--plot_training', action='store_true', help='Plot training progress')
    parser.add_argument('--load_model', type=str, default=None, help='Load pre-trained model')
    
    args = parser.parse_args()
    
    # 創建訓練系統
    training_system = MultiRobotTraining(
        num_robots=args.num_robots,
        num_episodes=args.num_episodes,
        save_interval=args.save_interval,
        model_save_path=args.model_save_path,
        use_attention=args.use_attention,
        plot_training=args.plot_training
    )
    
    # 如果指定了預訓練模型，則加載
    if args.load_model:
        training_system.trainer.load_model(args.load_model)
        print(f"Loaded pre-trained model from {args.load_model}")
    
    # 開始訓練
    training_system.train()


if __name__ == '__main__':
    main()