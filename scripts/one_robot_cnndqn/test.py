import os
import numpy as np
import tensorflow as tf
from time import sleep
import matplotlib.pyplot as plt
from one_robot_cnndqn.models.network import FrontierNetworkModel
from one_robot_cnndqn.environment.robot import Robot
from one_robot_cnndqn.config import MODEL_CONFIG, MODEL_DIR
from one_robot_cnndqn.visualization import Visualizer

def save_frame(robot, visualizer, save_path):
    """保存当前帧为图像"""
    # 创建新的图形
    plt.figure(figsize=(10, 10))
    
    # 使用visualizer绘制当前状态
    visualizer.visualize_frontiers(robot, save=False)
    
    # 保存图形
    plt.savefig(save_path, bbox_inches='tight', dpi=100, facecolor='white', edgecolor='none')
    
    # 关闭图形，释放内存
    plt.close('all')

def pad_frontiers(frontiers, max_frontiers, map_size):
    """Pad and normalize frontiers"""
    padded = np.zeros((max_frontiers, 2))
    if len(frontiers) > 0:
        normalized_frontiers = frontiers.copy()
        normalized_frontiers[:, 0] = frontiers[:, 0] / float(map_size[1])
        normalized_frontiers[:, 1] = frontiers[:, 1] / float(map_size[0])
        n_frontiers = min(len(frontiers), max_frontiers)
        padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
    return padded

def test_model(model_path, num_episodes=5, visualize=True, save_results=True, debug=True):
    """Test the trained frontier exploration model with debugging"""
    # Initialize model
    model = FrontierNetworkModel(
        input_shape=MODEL_CONFIG['input_shape'],
        max_frontiers=MODEL_CONFIG['max_frontiers']
    )
    model.load(model_path)
    
    # Initialize visualizer if needed
    visualizer = Visualizer() if visualize else None
    
    # Create test_file directory if it doesn't exist
    base_dir = "test_file"
    os.makedirs(base_dir, exist_ok=True)
    
    # Test statistics
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'exploration_ratios': [],
        'success_rate': 0
    }
    
    successful_episodes = 0
    
    for episode in range(num_episodes):
        print(f"\nStarting test episode {episode + 1}/{num_episodes}")
        
        # Create episode directory
        episode_dir = os.path.join(base_dir, f"episode_{episode+1}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Create test environment
        robot = Robot(episode, train=False, plot=visualize)
        
        # Start testing
        state = robot.begin()
        frame_count = 0
        
        # Save initial frame
        if visualize:
            save_frame(robot, visualizer, 
                      os.path.join(episode_dir, f"pic{frame_count:04d}.png"))
            frame_count += 1
        
        total_reward = 0
        steps = 0
        episode_done = False
        last_position = robot.robot_position.copy()
        stuck_counter = 0
        
        while not episode_done:
            # Get frontiers
            frontiers = robot.get_frontiers()
            current_position = robot.robot_position.copy()
            exploration_ratio = robot.get_exploration_progress()
            
            # Debug prints
            if debug:
                print(f"\nStep {steps + 1}:")
                print(f"Current position: {current_position}")
                print(f"Number of frontiers: {len(frontiers)}")
                print(f"Current exploration: {exploration_ratio:.1%}")
            
            # Check completion conditions
            if len(frontiers) == 0:
                print("No more frontiers available")
                episode_done = True
                break
                
            if exploration_ratio >= robot.finish_percent:
                print("Reached target exploration ratio")
                episode_done = True
                break
            
            # Get normalized robot position
            robot_pos = robot.get_normalized_position()
            
            # Prepare input for model prediction
            state_batch = np.expand_dims(state, 0)
            frontiers_batch = np.expand_dims(
                pad_frontiers(frontiers, MODEL_CONFIG['max_frontiers'], robot.map_size),
                0
            )
            robot_pos_batch = np.expand_dims(robot_pos, 0)
            
            try:
                # Predict best frontier
                q_values = model.predict(state_batch, frontiers_batch, robot_pos_batch)[0]
                action = np.argmax(q_values[:len(frontiers)])
                selected_frontier = frontiers[action]
                
                if debug:
                    print(f"Selected frontier: {selected_frontier}")
                    print(f"Q-value for selected frontier: {q_values[action]:.2f}")
                
                # Move to selected frontier
                next_state, reward, move_done = robot.move_to_frontier(selected_frontier)
                
                # Save frame after movement
                if visualize:
                    save_frame(robot, visualizer, 
                             os.path.join(episode_dir, f"pic{frame_count:04d}.png"))
                    frame_count += 1
                
                # Check if robot is stuck
                position_diff = np.linalg.norm(current_position - last_position)
                if position_diff < 0.1:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                    
                if stuck_counter >= 5:
                    print("Robot appears to be stuck, trying different frontier")
                    sorted_actions = np.argsort(q_values[:len(frontiers)])[::-1]
                    for alt_action in sorted_actions[1:]:
                        alt_frontier = frontiers[alt_action]
                        if np.linalg.norm(alt_frontier - selected_frontier) > 20:
                            next_state, reward, move_done = robot.move_to_frontier(alt_frontier)
                            # Save frame for alternative movement
                            if visualize:
                                save_frame(robot, visualizer, 
                                         os.path.join(episode_dir, f"pic{frame_count:04d}.png"))
                                frame_count += 1
                            break
                    stuck_counter = 0
                
                last_position = current_position.copy()
                
                # Update state and metrics
                total_reward += reward
                steps += 1
                state = next_state
                
                if debug:
                    print(f"Step reward: {reward:.2f}")
                    print(f"Move done: {move_done}")
                    print(f"Frames saved: {frame_count}")
                
                # Check if we need to end this episode
                if move_done and exploration_ratio < robot.finish_percent:
                    print("Movement failed, but exploration incomplete. Retrying...")
                    continue
                
                if steps >= 1000:
                    print("Reached maximum steps limit")
                    episode_done = True
                    
            except Exception as e:
                print(f"Error during step execution: {str(e)}")
                episode_done = True
        
        # Calculate final exploration ratio
        final_exploration = robot.get_exploration_progress()
        
        # Save final frame
        if visualize:
            save_frame(robot, visualizer, 
                      os.path.join(episode_dir, f"pic{frame_count:04d}.png"))
        
        # Record episode results
        results['episode_rewards'].append(total_reward)
        results['episode_lengths'].append(steps)
        results['exploration_ratios'].append(final_exploration)
        
        if final_exploration >= robot.finish_percent:
            successful_episodes += 1
        
        print(f"\nEpisode {episode + 1} Results:")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final exploration progress: {final_exploration:.1%}")
        print(f"Total frames saved: {frame_count + 1}")
        print("-" * 50)
    
    # Calculate overall statistics
    results['success_rate'] = successful_episodes / num_episodes
    
    print("\nOverall Test Results:")
    print(f"Average steps: {np.mean(results['episode_lengths']):.1f}")
    print(f"Average reward: {np.mean(results['episode_rewards']):.2f}")
    print(f"Average exploration ratio: {np.mean(results['exploration_ratios']):.1%}")
    print(f"Success rate: {results['success_rate']:.1%}")
    
    return results

if __name__ == '__main__':
    # Set up model path
    model_path = os.path.join(MODEL_DIR, 'frontier_model_ep000740.h5')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)
    
    # Enable GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Run tests with debugging enabled
    test_results = test_model(
        model_path=model_path,
        num_episodes=5,
        visualize=True,
        save_results=True,
        debug=True
    )
