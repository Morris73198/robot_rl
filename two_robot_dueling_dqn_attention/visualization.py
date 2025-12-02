import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

class Visualizer:
    def __init__(self):
        """初始化可视化器"""
        plt.style.use('seaborn')
        self.training_fig = None
        self.frontier_fig = None
        self.current_episode = 0
        
    def plot_training_progress(self, history):
        """绘制训练进度图
        
        Args:
            history: 包含训练历史数据的字典:
                - episode_rewards: 每个episode的总奖励
                - episode_lengths: 每个episode的步数
                - exploration_rates: 探索率
                - losses: 训练损失
        """
        if self.training_fig is None:
            self.training_fig, self.axs = plt.subplots(4, 1, figsize=(12, 15))
            plt.tight_layout(pad=3.0)
        else:
            for ax in self.axs:
                ax.clear()
                
        # 设置风格
        colors = sns.color_palette("husl", 4)
        
        # Plot rewards
        episodes = range(1, len(history['episode_rewards']) + 1)
        self.axs[0].plot(episodes, history['episode_rewards'], color=colors[0])
        self.axs[0].set_title('Episode Rewards', fontsize=12, pad=10)
        self.axs[0].set_xlabel('Episode')
        self.axs[0].set_ylabel('Total Reward')
        self.axs[0].grid(True, alpha=0.3)
        
        # Plot episode lengths
        self.axs[1].plot(episodes, history['episode_lengths'], color=colors[1])
        self.axs[1].set_title('Episode Lengths', fontsize=12, pad=10)
        self.axs[1].set_xlabel('Episode')
        self.axs[1].set_ylabel('Steps')
        self.axs[1].grid(True, alpha=0.3)
        
        # Plot exploration rate
        self.axs[2].plot(episodes, history['exploration_rates'], color=colors[2])
        self.axs[2].set_title('Exploration Rate', fontsize=12, pad=10)
        self.axs[2].set_xlabel('Episode')
        self.axs[2].set_ylabel('Epsilon')
        self.axs[2].grid(True, alpha=0.3)
        
        # Plot losses
        self.axs[3].plot(episodes, history['losses'], color=colors[3])
        self.axs[3].set_title('Training Loss', fontsize=12, pad=10)
        self.axs[3].set_xlabel('Episode')
        self.axs[3].set_ylabel('Loss')
        self.axs[3].grid(True, alpha=0.3)
        
        plt.savefig(f'training_progress_ep{len(episodes)}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_frontiers(self, robot, save=False):
        """可视化当前地图上的所有frontier点
        
        Args:
            robot: Robot类实例
            save: 是否保存图像
        """
        if self.frontier_fig is None:
            self.frontier_fig, self.frontier_ax = plt.subplots(figsize=(10, 10))
        else:
            self.frontier_ax.clear()
            
        # 创建自定义colormap
        colors = ['darkgrey', 'white', 'black']  # unknown, free, obstacle
        n_bins = 3
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
        
        # 绘制地图
        im = self.frontier_ax.imshow(robot.op_map, cmap=cmap, vmin=0, vmax=255)
        
        # 获取frontiers
        frontiers = robot.get_frontiers()
        
        # 绘制机器人位置
        robot_circle = Circle((robot.robot_position[0], robot.robot_position[1]), 
                            radius=5, color='blue', label='Robot')
        self.frontier_ax.add_patch(robot_circle)
        
        # 绘制所有frontier点和路径
        if len(frontiers) > 0:
            frontiers = np.array(frontiers)
            
            # 绘制frontier点
            self.frontier_ax.scatter(frontiers[:, 0], frontiers[:, 1], 
                                   c='red', s=30, label='Frontiers')
            
            # 绘制从机器人到各个frontier的路径
            for frontier in frontiers:
                self.frontier_ax.plot([robot.robot_position[0], frontier[0]], 
                                    [robot.robot_position[1], frontier[1]], 
                                    'g--', alpha=0.2)
        
        # 添加colorbar
        cbar = plt.colorbar(im)
        cbar.set_ticks([0, 127, 255])
        cbar.set_ticklabels(['Unknown', 'Free', 'Obstacle'])
        
        # 设置标题和图例
        self.frontier_ax.set_title(f'Map Exploration - {len(frontiers)} Frontiers', 
                                 pad=20)
        self.frontier_ax.legend(loc='upper right')
        
        # 修正坐标轴方向
        self.frontier_ax.axis((0, robot.map_size[1], robot.map_size[0], 0))
        
        # 添加探索进度
        explored_ratio = np.sum(robot.op_map == 255) / np.sum(robot.global_map == 255)
        self.frontier_ax.text(0.02, 0.02, f'Explored: {explored_ratio:.1%}', 
                            transform=self.frontier_ax.transAxes,
                            bbox=dict(facecolor='white', alpha=0.7))
        
        if save:
            plt.savefig(f'frontier_map_ep{self.current_episode}.png', 
                       dpi=300, bbox_inches='tight')
            self.current_episode += 1
        else:
            plt.pause(0.1)
        
        plt.close()

    def visualize_path(self, robot, path, selected_frontier=None):
        """可视化机器人路径和选定的frontier
        
        Args:
            robot: Robot类实例
            path: 路径点列表
            selected_frontier: 选定的frontier点
        """
        plt.figure(figsize=(10, 10))
        
        # 绘制地图
        plt.imshow(robot.op_map, cmap='gray')
        
        # 绘制路径
        if path is not None and len(path) > 0:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
        
        # 绘制机器人位置
        plt.plot(robot.robot_position[0], robot.robot_position[1], 
                'bo', markersize=10, label='Robot')
        
        # 绘制选定的frontier
        if selected_frontier is not None:
            plt.plot(selected_frontier[0], selected_frontier[1], 
                    'ro', markersize=10, label='Target Frontier')
        
        plt.title('Robot Path Planning')
        plt.legend()
        plt.axis((0, robot.map_size[1], robot.map_size[0], 0))
        plt.pause(0.1)
        plt.close()

    def create_exploration_animation(self, robot, history, output_file='exploration.gif'):
        """创建探索过程的动画
        
        Args:
            robot: Robot类实例
            history: 包含历史位置和地图状态的列表
            output_file: 输出文件名
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            ax.clear()
            state = history[frame]
            
            # 绘制地图状态
            ax.imshow(state['map'], cmap='gray')
            
            # 绘制机器人位置
            ax.plot(state['robot_pos'][0], state['robot_pos'][1], 
                   'bo', markersize=10)
            
            # 绘制frontier点
            if 'frontiers' in state and len(state['frontiers']) > 0:
                frontiers = np.array(state['frontiers'])
                ax.scatter(frontiers[:, 0], frontiers[:, 1], 
                          c='red', s=30)
            
            ax.set_title(f'Step {frame}')
            ax.axis((0, robot.map_size[1], robot.map_size[0], 0))
            
        anim = FuncAnimation(fig, update, frames=len(history), 
                           interval=200, blit=False)
        anim.save(output_file, writer='pillow')
        plt.close()