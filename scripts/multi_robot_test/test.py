#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-time Multi-Robot Exploration Visualization
Displays robot positions, map exploration, and performance metrics
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches
import time
from collections import deque

# ========== 配置区域 ==========
DEFAULT_MODEL_PATH = '/home/airlab2/robot_rl/saved_models/multi_robot_model_episode_4940.h5'  # 在这里指定默认模型
DEFAULT_NUM_ROBOTS = 3
DEFAULT_NUM_EPISODES = 5
DEFAULT_MAX_STEPS = 1500
DEFAULT_MAP_INDEX = 0
DEFAULT_UPDATE_INTERVAL = 10
# =============================

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_robot_test.models.multi_robot_network import MultiRobotNetworkModel
from multi_robot_test.environment.multi_robot_no_unknown import Robot
from multi_robot_test.config import MODEL_CONFIG, MODEL_DIR


def load_model_weights(model_path, max_robots=10):
    """Load model weights only"""
    print(f"Loading model weights: {model_path}")
    
    model = MultiRobotNetworkModel(
        input_shape=MODEL_CONFIG['input_shape'],
        max_frontiers=MODEL_CONFIG['max_frontiers'],
        max_robots=max_robots
    )
    
    try:
        print("Attempting to load model weights...")
        model.model.load_weights(model_path)
        model.target_model.load_weights(model_path)
        print("✓ Weights loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load weights: {str(e)}")
        return None


class RealTimeMapVisualizer:
    """Real-time visualization with map and robot positions"""
    
    def __init__(self, num_robots, max_data_points=500):
        """Initialize visualization
        
        Args:
            num_robots: Number of robots
            max_data_points: Maximum data points to display
        """
        self.num_robots = num_robots
        self.max_data_points = max_data_points
        
        # Robot colors
        self.robot_colors = ['#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#00FFFF'][:num_robots]
        
        # Data storage
        self.steps = deque(maxlen=max_data_points)
        self.exploration = deque(maxlen=max_data_points)
        self.total_rewards = deque(maxlen=max_data_points)
        self.robot_rewards = [deque(maxlen=max_data_points) for _ in range(num_robots)]
        self.path_lengths = [deque(maxlen=max_data_points) for _ in range(num_robots)]
        
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.manager.set_window_title('Multi-Robot Exploration Monitor')
        
        # Create subplots with GridSpec
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # 1. Map Visualization (Large - left side)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.map_img = None
        self.robot_markers = []
        self.robot_paths = [[] for _ in range(num_robots)]
        self.path_lines = []
        self.target_markers = []
        
        self.ax_map.set_title('Environment Map & Robot Positions', 
                             fontsize=14, fontweight='bold', pad=10)
        self.ax_map.set_xlabel('X Position (pixels)', fontsize=11)
        self.ax_map.set_ylabel('Y Position (pixels)', fontsize=11)
        
        # 2. Exploration Progress (top right)
        self.ax_exploration = self.fig.add_subplot(gs[0, 1:])
        self.line_exploration, = self.ax_exploration.plot([], [], 'b-', linewidth=2.5, 
                                                          label='Exploration Progress')
        self.target_line = self.ax_exploration.axhline(y=99.5, color='r', linestyle='--', 
                                                       alpha=0.7, linewidth=2, label='Target (99.5%)')
        self.ax_exploration.fill_between([], [], alpha=0.2, color='blue')
        self.ax_exploration.set_xlabel('Steps', fontsize=11)
        self.ax_exploration.set_ylabel('Coverage (%)', fontsize=11)
        self.ax_exploration.set_title('Real-Time Exploration Progress', 
                                     fontsize=13, fontweight='bold')
        self.ax_exploration.grid(True, alpha=0.3, linestyle='--')
        self.ax_exploration.legend(loc='lower right', fontsize=10)
        self.ax_exploration.set_ylim([0, 105])
        
        # 3. Cumulative Rewards (middle right)
        self.ax_rewards = self.fig.add_subplot(gs[1, 1:])
        self.lines_robot_rewards = []
        for i in range(num_robots):
            line, = self.ax_rewards.plot([], [], linewidth=2, 
                                        label=f'Robot {i}', 
                                        color=self.robot_colors[i], 
                                        alpha=0.8)
            self.lines_robot_rewards.append(line)
        self.line_total_reward, = self.ax_rewards.plot([], [], 'k--', linewidth=2.5, 
                                                       label='Total', alpha=0.9)
        self.ax_rewards.set_xlabel('Steps', fontsize=11)
        self.ax_rewards.set_ylabel('Cumulative Reward', fontsize=11)
        self.ax_rewards.set_title('Cumulative Rewards Over Time', 
                                 fontsize=13, fontweight='bold')
        self.ax_rewards.grid(True, alpha=0.3, linestyle='--')
        self.ax_rewards.legend(loc='lower right', fontsize=10, ncol=2)
        
        # 4. Statistics Panel (bottom right)
        self.ax_stats = self.fig.add_subplot(gs[2, 1:])
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.95, '', 
                                            transform=self.ax_stats.transAxes,
                                            fontsize=10, 
                                            verticalalignment='top', 
                                            fontfamily='monospace',
                                            bbox=dict(boxstyle='round', 
                                                    facecolor='lightblue', 
                                                    alpha=0.3, 
                                                    edgecolor='gray'))
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show(block=False)
        
    def update(self, step, exploration_pct, total_reward, robot_rewards_list, 
               path_lengths_list, robots_info=None):
        """Update visualization data
        
        Args:
            step: Current step number
            exploration_pct: Exploration progress percentage
            total_reward: Total cumulative reward
            robot_rewards_list: List of cumulative rewards per robot
            path_lengths_list: List of cumulative path lengths per robot
            robots_info: Dictionary containing robot positions, map, targets, etc.
        """
        # Add data points
        self.steps.append(step)
        self.exploration.append(exploration_pct)
        self.total_rewards.append(total_reward)
        
        for i in range(self.num_robots):
            self.robot_rewards[i].append(robot_rewards_list[i])
            self.path_lengths[i].append(path_lengths_list[i])
        
        # Update exploration progress curve
        self.line_exploration.set_data(list(self.steps), list(self.exploration))
        if len(self.steps) > 0:
            self.ax_exploration.set_xlim([0, max(self.steps[-1] * 1.1, 100)])
        
        # Update rewards curves
        for i, line in enumerate(self.lines_robot_rewards):
            line.set_data(list(self.steps), list(self.robot_rewards[i]))
        self.line_total_reward.set_data(list(self.steps), list(self.total_rewards))
        
        if len(self.total_rewards) > 0:
            self.ax_rewards.set_xlim([0, max(self.steps[-1] * 1.1, 100)])
            all_rewards = [val for r in self.robot_rewards for val in r] + list(self.total_rewards)
            min_reward = min(all_rewards)
            max_reward = max(all_rewards)
            margin = (max_reward - min_reward) * 0.1 if max_reward != min_reward else 100
            self.ax_rewards.set_ylim([min_reward - margin, max_reward + margin])
        
        # Update map with robot positions
        if robots_info is not None:
            robot_map = robots_info.get('map')
            robot_positions = robots_info.get('positions', [])
            robot_targets = robots_info.get('targets', [])
            frontiers = robots_info.get('frontiers', [])
            
            if robot_map is not None:
                # Display map
                if self.map_img is None:
                    # Create color map: obstacles=black, explored=white, unexplored=gray
                    display_map = np.zeros((*robot_map.shape, 3))
                    display_map[robot_map == 1] = [0, 0, 0]        # Obstacles - black
                    display_map[robot_map == 255] = [1, 1, 1]      # Explored - white
                    display_map[robot_map == 127] = [0.5, 0.5, 0.5]  # Unexplored - gray
                    
                    self.map_img = self.ax_map.imshow(display_map, origin='lower')
                    self.ax_map.set_xlim([0, robot_map.shape[1]])
                    self.ax_map.set_ylim([0, robot_map.shape[0]])
                else:
                    # Update map
                    display_map = np.zeros((*robot_map.shape, 3))
                    display_map[robot_map == 1] = [0, 0, 0]
                    display_map[robot_map == 255] = [1, 1, 1]
                    display_map[robot_map == 127] = [0.5, 0.5, 0.5]
                    self.map_img.set_data(display_map)
                
                # Clear old markers
                for marker in self.robot_markers:
                    marker.remove()
                for marker in self.target_markers:
                    marker.remove()
                for line in self.path_lines:
                    line.remove()
                
                self.robot_markers = []
                self.target_markers = []
                self.path_lines = []
                
                # Draw robot positions and paths
                for i, pos in enumerate(robot_positions):
                    if pos is not None:
                        # Add current position to path history
                        self.robot_paths[i].append(pos.copy())
                        
                        # Keep only recent path points (last 100)
                        if len(self.robot_paths[i]) > 100:
                            self.robot_paths[i].pop(0)
                        
                        # Draw path trail
                        if len(self.robot_paths[i]) > 1:
                            path_array = np.array(self.robot_paths[i])
                            line, = self.ax_map.plot(path_array[:, 0], path_array[:, 1], 
                                                    color=self.robot_colors[i], 
                                                    alpha=0.3, linewidth=2, linestyle='-')
                            self.path_lines.append(line)
                        
                        # Draw robot position (circle)
                        circle = Circle((pos[0], pos[1]), radius=8, 
                                      color=self.robot_colors[i], 
                                      alpha=0.8, zorder=10)
                        self.ax_map.add_patch(circle)
                        self.robot_markers.append(circle)
                        
                        # Add robot label
                        text = self.ax_map.text(pos[0], pos[1], f'R{i}', 
                                              ha='center', va='center', 
                                              fontsize=9, fontweight='bold',
                                              color='white', zorder=11)
                        self.robot_markers.append(text)
                        
                        # Draw target if exists
                        if i < len(robot_targets) and robot_targets[i] is not None:
                            target = robot_targets[i]
                            # Target marker (X)
                            marker = self.ax_map.plot(target[0], target[1], 'x', 
                                                     color=self.robot_colors[i], 
                                                     markersize=12, markeredgewidth=3,
                                                     zorder=9)[0]
                            self.target_markers.append(marker)
                            
                            # Draw line from robot to target
                            line, = self.ax_map.plot([pos[0], target[0]], 
                                                    [pos[1], target[1]], 
                                                    color=self.robot_colors[i], 
                                                    linestyle='--', linewidth=1.5, 
                                                    alpha=0.5, zorder=8)
                            self.target_markers.append(line)
                
                # Draw frontiers
                if len(frontiers) > 0:
                    frontier_array = np.array(frontiers)
                    scatter = self.ax_map.scatter(frontier_array[:, 0], 
                                                 frontier_array[:, 1], 
                                                 c='yellow', s=20, alpha=0.6, 
                                                 marker='o', edgecolors='orange',
                                                 linewidths=0.5, zorder=5)
                    self.target_markers.append(scatter)
                
                # Create legend for map
                legend_elements = [
                    mpatches.Patch(facecolor='black', edgecolor='black', label='Obstacles'),
                    mpatches.Patch(facecolor='white', edgecolor='gray', label='Explored'),
                    mpatches.Patch(facecolor='gray', edgecolor='gray', label='Unexplored'),
                    mpatches.Patch(facecolor='yellow', edgecolor='orange', label='Frontiers')
                ]
                for i in range(self.num_robots):
                    legend_elements.append(
                        mpatches.Patch(facecolor=self.robot_colors[i], 
                                     label=f'Robot {i}')
                    )
                
                self.ax_map.legend(handles=legend_elements, loc='upper right', 
                                  fontsize=9, framealpha=0.8)
        
        # Update statistics text
        total_path = sum([path_lengths_list[i] for i in range(self.num_robots)])
        stats_str = f"""Current Status (Step: {step})
{'='*55}

Exploration Progress:  {exploration_pct:6.2f}%
Total Reward:          {total_reward:10.2f}
Total Path Length:     {total_path:10.2f} px

Individual Robot Status:
"""
        for i in range(self.num_robots):
            stats_str += f"  Robot {i}: "
            stats_str += f"Reward={robot_rewards_list[i]:8.2f}  "
            stats_str += f"Path={path_lengths_list[i]:8.2f} px\n"
        
        if total_path > 0:
            efficiency = (exploration_pct / 100.0) / (total_path / 1000.0)
            stats_str += f"\nExploration Efficiency:  {efficiency:.4f} (%/km)"
        
        self.stats_text.set_text(stats_str)
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def close(self):
        """Close visualization"""
        plt.close(self.fig)


def test_single_episode_with_map_visualization(model, robots, episode_idx=0, max_steps=1500, 
                                                update_interval=10, save_dir=None):
    """Run single test episode with real-time map visualization
    
    Args:
        model: Trained model
        robots: List of robots
        episode_idx: Episode index
        max_steps: Maximum steps
        update_interval: Visualization update interval (steps)
        save_dir: Save directory
    
    Returns:
        dict: Episode statistics
    """
    print(f"\n{'='*60}")
    print(f"Test Episode {episode_idx + 1}")
    print(f"{'='*60}")
    
    # Reset environment
    state = robots[0].reset()
    for robot in robots[1:]:
        robot.reset()
    
    state = robots[0].begin()
    for robot in robots[1:]:
        robot.begin()
    
    # Statistics
    total_reward = 0
    robot_rewards = [0] * len(robots)
    steps = 0
    path_lengths = [0] * len(robots)
    previous_positions = [robot.robot_position.copy() for robot in robots]
    
    # Create visualizer
    visualizer = RealTimeMapVisualizer(len(robots))
    
    # Pad frontiers function
    def pad_frontiers(frontiers):
        max_frontiers = MODEL_CONFIG['max_frontiers']
        padded = np.zeros((max_frontiers, 2))
        if len(frontiers) > 0:
            num_to_copy = min(len(frontiers), max_frontiers)
            padded[:num_to_copy] = frontiers[:num_to_copy]
        return padded
    
    start_time = time.time()
    print("\nStarting exploration...")
    
    # Main loop
    try:
        while not any(robot.check_done() for robot in robots) and steps < max_steps:
            frontiers = robots[0].get_frontiers()
            
            if len(frontiers) == 0:
                print("  ⚠ No available frontiers")
                break
            
            # Collect robot states
            map_dims = np.array([robots[0].map_size[1], robots[0].map_size[0]])
            robots_poses = [robot.robot_position / map_dims for robot in robots]
            robots_targets = []
            
            for robot in robots:
                if robot.current_target_frontier is None:
                    robots_targets.append(np.zeros(2))
                else:
                    robots_targets.append(robot.current_target_frontier / map_dims)
            
            # Predict actions (greedy policy)
            predictions = model.predict(
                np.expand_dims(state, 0),
                np.expand_dims(pad_frontiers(frontiers), 0),
                robots_poses,
                robots_targets,
                len(robots)
            )
            
            # Select actions
            actions = []
            for i in range(len(robots)):
                robot_q_values = predictions[f'robot{i}'][0, :len(frontiers)]
                actions.append(np.argmax(robot_q_values))
            
            # Execute actions
            next_states = []
            rewards = []
            
            for i, robot in enumerate(robots):
                action = min(actions[i], len(frontiers) - 1)
                next_state, reward, done = robot.move_to_frontier(frontiers[action])
                
                # Synchronize map state to other robots
                for other_robot in robots:
                    if other_robot != robot:
                        other_robot.op_map = robot.op_map.copy()
                
                next_states.append(next_state)
                rewards.append(reward)
                robot_rewards[i] += reward
                
                # Calculate path length
                path_length = np.linalg.norm(robot.robot_position - previous_positions[i])
                path_lengths[i] += path_length
                previous_positions[i] = robot.robot_position.copy()
            
            # Update state
            state = next_states[0]
            total_reward += sum(rewards)
            steps += 1
            
            # Update visualization
            if steps % update_interval == 0:
                exploration_pct = robots[0].get_exploration_progress() * 100
                
                # Prepare robot information for visualization
                robots_info = {
                    'map': robots[0].op_map.copy(),
                    'positions': [robot.robot_position.copy() for robot in robots],
                    'targets': [robot.current_target_frontier.copy() 
                               if robot.current_target_frontier is not None else None 
                               for robot in robots],
                    'frontiers': frontiers
                }
                
                visualizer.update(
                    steps, 
                    exploration_pct,
                    total_reward,
                    robot_rewards.copy(),
                    path_lengths.copy(),
                    robots_info
                )
                
                # Terminal output
                elapsed = time.time() - start_time
                speed = steps / elapsed if elapsed > 0 else 0
                print(f"  Step {steps:4d} | Coverage: {exploration_pct:6.2f}% | "
                      f"Reward: {total_reward:9.2f} | Speed: {speed:5.1f} steps/s")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
    
    # Final statistics
    elapsed_time = time.time() - start_time
    final_exploration = robots[0].get_exploration_progress()
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx + 1} Complete")
    print(f"{'='*60}")
    print(f"Total Steps:           {steps}")
    print(f"Execution Time:        {elapsed_time:.2f} seconds")
    print(f"Total Reward:          {total_reward:.2f}")
    print(f"Final Exploration:     {final_exploration:.1%}")
    
    total_path = sum(path_lengths)
    print(f"\nIndividual Robot Performance:")
    for i in range(len(robots)):
        print(f"  Robot {i}:")
        print(f"    - Reward:          {robot_rewards[i]:8.2f}")
        print(f"    - Path Length:     {path_lengths[i]:8.2f} px")
        print(f"    - Path Ratio:      {path_lengths[i]/total_path*100:6.1f}%")
    
    if total_path > 0:
        efficiency = (final_exploration * 100) / (total_path / 1000.0)
        print(f"\nExploration Efficiency: {efficiency:.4f} (%/km)")
    
    # Save final visualization
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'episode_{episode_idx + 1}_final.png')
        visualizer.fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"\n✓ Saved visualization: {save_path}")
    
    # Pause for user to view
    print("\nPress Enter to continue or Ctrl+C to exit...")
    try:
        input()
    except KeyboardInterrupt:
        pass
    
    # Close visualizer
    visualizer.close()
    
    return {
        'steps': steps,
        'time': elapsed_time,
        'total_reward': total_reward,
        'exploration': final_exploration,
        'robot_rewards': robot_rewards,
        'path_lengths': path_lengths,
        'efficiency': efficiency if total_path > 0 else 0
    }


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Robot DQN Test (Real-Time Map Visualization)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Model file path')
    parser.add_argument('--num_robots', type=int, default=DEFAULT_NUM_ROBOTS,
                        help='Number of robots')
    parser.add_argument('--num_episodes', type=int, default=DEFAULT_NUM_EPISODES,
                        help='Number of test episodes')
    parser.add_argument('--max_steps', type=int, default=DEFAULT_MAX_STEPS,
                        help='Maximum steps per episode')
    parser.add_argument('--map_index', type=int, default=DEFAULT_MAP_INDEX,
                        help='Map index')
    parser.add_argument('--update_interval', type=int, default=DEFAULT_UPDATE_INTERVAL,
                        help='Visualization update interval (steps)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Multi-Robot DQN Test System (Real-Time Visualization)")
    print("="*60)
    
    # 使用指定的模型路径
    model_path = args.model
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"\nError: Model file not found: {model_path}")
        print(f"Please check the path in DEFAULT_MODEL_PATH configuration.")
        return
    
    print(f"\nUsing model: {os.path.basename(model_path)}")
    print(f"Full path: {model_path}")
    print(f"Number of Robots:  {args.num_robots}")
    print(f"Test Episodes:     {args.num_episodes}")
    print(f"Max Steps:         {args.max_steps}")
    print(f"Update Interval:   {args.update_interval} steps")
    
    # Load model
    print("\n" + "-"*60)
    model = load_model_weights(model_path, max_robots=10)
    if model is None:
        print("Model loading failed, exiting")
        return
    
    # Create test environment
    print("\n" + "-"*60)
    print(f"Creating test environment with {args.num_robots} robots...")
    try:
        robots = Robot.create_shared_robots(
            index_map=args.map_index,
            num_robots=args.num_robots,
            train=False,
            plot=False  # Disable Robot's own plotting
        )
        
        if robots is None or len(robots) == 0:
            print("Failed to create robot environment")
            return
        
        print(f"✓ Successfully created {len(robots)} robots")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create save directory
    save_dir = os.path.join(MODEL_DIR, 'test_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # Run tests
    print("\n" + "="*60)
    print("Starting Test (Real-time visualization window will display)")
    print("="*60)
    print("\nTips:")
    print("  - Visualization window updates in real-time")
    print("  - Press Enter after each episode to continue")
    print("  - Press Ctrl+C anytime to exit")
    
    all_stats = []
    
    for episode in range(args.num_episodes):
        try:
            stats = test_single_episode_with_map_visualization(
                model, robots, episode,
                max_steps=args.max_steps,
                update_interval=args.update_interval,
                save_dir=save_dir
            )
            all_stats.append(stats)
        except KeyboardInterrupt:
            print("\n\n⚠ Test interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Episode {episode + 1} error: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall statistics
    if all_stats:
        print("\n" + "="*60)
        print("Overall Test Statistics")
        print("="*60)
        
        avg_steps = np.mean([s['steps'] for s in all_stats])
        avg_time = np.mean([s['time'] for s in all_stats])
        avg_reward = np.mean([s['total_reward'] for s in all_stats])
        avg_exploration = np.mean([s['exploration'] for s in all_stats])
        avg_efficiency = np.mean([s['efficiency'] for s in all_stats])
        
        print(f"\nCompleted Episodes: {len(all_stats)}/{args.num_episodes}")
        print(f"\nAverage Performance:")
        print(f"  Steps:             {avg_steps:.2f}")
        print(f"  Time:              {avg_time:.2f} seconds")
        print(f"  Total Reward:      {avg_reward:.2f}")
        print(f"  Exploration:       {avg_exploration:.1%}")
        print(f"  Efficiency:        {avg_efficiency:.4f} (%/km)")
        
        # Save statistics
        stats_file = os.path.join(save_dir, 'test_statistics_map_viz.txt')
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Real-Time Map Visualization Test Statistics\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Model: {os.path.basename(model_path)}\n")
            f.write(f"Number of Robots: {args.num_robots}\n")
            f.write(f"Completed Episodes: {len(all_stats)}/{args.num_episodes}\n\n")
            
            f.write("Average Performance:\n")
            f.write(f"  Steps:       {avg_steps:.2f}\n")
            f.write(f"  Time:        {avg_time:.2f} seconds\n")
            f.write(f"  Reward:      {avg_reward:.2f}\n")
            f.write(f"  Exploration: {avg_exploration:.1%}\n")
            f.write(f"  Efficiency:  {avg_efficiency:.4f}\n\n")
            
            for i, stats in enumerate(all_stats):
                f.write(f"\nEpisode {i + 1}:\n")
                f.write(f"  Steps:       {stats['steps']}\n")
                f.write(f"  Time:        {stats['time']:.2f}s\n")
                f.write(f"  Exploration: {stats['exploration']:.1%}\n")
                f.write(f"  Efficiency:  {stats['efficiency']:.4f}\n")
        
        print(f"\n✓ Statistics saved: {stats_file}")
        print("="*60)
    
    # Cleanup
    plt.close('all')
    for robot in robots:
        if hasattr(robot, 'cleanup_visualization'):
            robot.cleanup_visualization()
    
    print("\n✓ Test Complete!\n")


if __name__ == '__main__':
    main()
