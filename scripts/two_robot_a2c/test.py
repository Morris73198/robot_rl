import os
import sys
import numpy as np
import matplotlib
# Set the backend to Agg before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from two_robot_a2c.models.multi_robot_network import MultiRobotACModel
from two_robot_a2c.environment.multi_robot_no_unknown import Robot
from two_robot_a2c.config import MODEL_CONFIG, MODEL_DIR
from two_robot_a2c.environment.robot_local_map_tracker import RobotIndividualMapTracker

def save_plot(robot, step, output_path):
    """Save a single robot's plot
    
    Args:
        robot: Robot instance
        step: Current step number
        output_path: Path to save the plot
    """
    # Create a new figure for each plot
    plt.figure(figsize=(10, 10))
    robot.plot_env()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')  # Close all figures to free memory

def test_model(model_path, num_episodes=5):
    """Test the A2C model and save exploration visualizations
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to run
    """
    robot1, robot2 = None, None
    map_tracker = None
    
    try:
        # Create base output directory
        base_output_dir = 'result_a2c'
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
            
        # Load model
        print("Loading model from:", model_path)
        model = MultiRobotACModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        model.load(model_path)
        
        # Try to verify model, but continue even if verification fails
        try:
            print("驗證模型...")
            model.verify_model()
            print("模型驗證通過")
        except Exception as e:
            print(f"模型驗證警告 (將繼續執行): {str(e)}")
        
        # Create shared environment
        print("Initializing test environment...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0,
            train=False,
            plot=True
        )
        
        # Create map tracker
        print("Creating robot individual map tracker...")
        map_tracker = RobotIndividualMapTracker(
            robot1=robot1,
            robot2=robot2,
            save_dir=os.path.join(base_output_dir, 'robot_individual_maps')
        )
        
        # Track statistics
        episode_stats = {
            'exploration_progress': [],
            'steps': [],
            'robot1_path_length': [],
            'robot2_path_length': [],
            'overlap_ratios': []
        }
        
        for episode in range(num_episodes):
            print(f"\nStarting episode {episode + 1}/{num_episodes}")
            
            # Create episode directory
            episode_dir = os.path.join(base_output_dir, f'episode_{episode+1:02d}')
            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir)
            
            # Reset environment and initialize episode
            state = robot1.begin()
            robot2.begin()
            
            # Start tracking maps
            map_tracker.start_tracking()
            
            # Save initial state
            save_plot(robot1, 0, os.path.join(episode_dir, f'robot1_step_0000.png'))
            save_plot(robot2, 0, os.path.join(episode_dir, f'robot2_step_0000.png'))
            
            steps = 0
            robot1_path_length = 0
            robot2_path_length = 0
            
            while not (robot1.check_done() or robot2.check_done()):
                frontiers = robot1.get_frontiers()
                if len(frontiers) == 0:
                    break
                    
                # Get current positions and normalize
                robot1_pos = robot1.get_normalized_position()
                robot2_pos = robot2.get_normalized_position()
                old_robot1_pos = robot1.robot_position.copy()
                old_robot2_pos = robot2.robot_position.copy()
                
                # Normalize targets - implementing the target normalization directly
                # Define normalization function
                def normalize_target(target, map_size):
                    if target is None:
                        return np.array([0.0, 0.0])
                    normalized = np.array([
                        target[0] / float(map_size[1]),
                        target[1] / float(map_size[0])
                    ])
                    return normalized
                
                robot1_target = normalize_target(robot1.current_target_frontier, robot1.map_size)
                robot2_target = normalize_target(robot2.current_target_frontier, robot2.map_size)
                
                # Pad frontiers to fixed length and normalize
                def pad_frontiers(frontiers, max_frontiers, map_size):
                    """Fill frontier points to fixed length and normalize coordinates"""
                    padded = np.zeros((max_frontiers, 2))
                    
                    if len(frontiers) > 0:
                        frontiers_arr = np.array(frontiers)
                        
                        # Normalize coordinates
                        normalized_frontiers = frontiers_arr.copy()
                        normalized_frontiers[:, 0] = frontiers_arr[:, 0] / float(map_size[1])
                        normalized_frontiers[:, 1] = frontiers_arr[:, 1] / float(map_size[0])
                        
                        n_frontiers = min(len(frontiers), max_frontiers)
                        padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
                    
                    return padded
                
                # Create state batch
                state_batch = np.expand_dims(state, 0)
                frontiers_batch = np.expand_dims(pad_frontiers(frontiers, MODEL_CONFIG['max_frontiers'], robot1.map_size), 0)
                robot1_pos_batch = np.expand_dims(robot1_pos, 0)
                robot2_pos_batch = np.expand_dims(robot2_pos, 0)
                robot1_target_batch = np.expand_dims(robot1_target, 0)
                robot2_target_batch = np.expand_dims(robot2_target, 0)
                
                # Get action probabilities from the actor network
                policy_dict = model.predict_policy(
                    state_batch, frontiers_batch,
                    robot1_pos_batch, robot2_pos_batch,
                    robot1_target_batch, robot2_target_batch
                )
                
                # Select best actions (no exploration in testing)
                valid_frontiers = min(MODEL_CONFIG['max_frontiers'], len(frontiers))
                robot1_action = np.argmax(policy_dict['robot1_policy'][0, :valid_frontiers])
                robot2_action = np.argmax(policy_dict['robot2_policy'][0, :valid_frontiers])
                
                # Get target positions for both robots
                robot1_target_pos = frontiers[robot1_action]
                robot2_target_pos = frontiers[robot2_action]
                
                # Move robots
                next_state1, r1, d1 = robot1.move_to_frontier(robot1_target_pos)
                robot2.op_map = robot1.op_map.copy()
                
                next_state2, r2, d2 = robot2.move_to_frontier(robot2_target_pos)
                robot1.op_map = robot2.op_map.copy()
                
                # Update positions
                robot1.other_robot_position = robot2.robot_position.copy()
                robot2.other_robot_position = robot1.robot_position.copy()
                
                # Update map tracker
                map_tracker.update()
                
                # Update path lengths
                robot1_path_length += np.linalg.norm(robot1.robot_position - old_robot1_pos)
                robot2_path_length += np.linalg.norm(robot2.robot_position - old_robot2_pos)
                
                state = next_state1
                steps += 1
                
                # Save plots every 10 steps
                if steps % 10 == 0:
                    save_plot(robot1, steps, 
                            os.path.join(episode_dir, f'robot1_step_{steps:04d}.png'))
                    save_plot(robot2, steps, 
                            os.path.join(episode_dir, f'robot2_step_{steps:04d}.png'))
                    
                    # Save current individual maps
                    map_tracker.save_current_maps(steps)
            
            # Save final state
            save_plot(robot1, steps, 
                     os.path.join(episode_dir, f'robot1_final_step_{steps:04d}.png'))
            save_plot(robot2, steps, 
                     os.path.join(episode_dir, f'robot2_final_step_{steps:04d}.png'))
            
            # Calculate overlap ratio
            overlap_ratio = map_tracker.calculate_overlap()
            
            # Stop tracking and generate overlap graph
            map_tracker.stop_tracking()
            map_tracker.plot_coverage_over_time()
            
            # Record statistics
            final_progress = robot1.get_exploration_progress()
            episode_stats['exploration_progress'].append(final_progress)
            episode_stats['steps'].append(steps)
            episode_stats['robot1_path_length'].append(robot1_path_length)
            episode_stats['robot2_path_length'].append(robot2_path_length)
            episode_stats['overlap_ratios'].append(float(overlap_ratio))
            
            print(f"Episode {episode + 1} Results:")
            print(f"Steps taken: {steps}")
            print(f"Final exploration progress: {final_progress:.1%}")
            print(f"Robot1 path length: {robot1_path_length:.2f}")
            print(f"Robot2 path length: {robot2_path_length:.2f}")
            print(f"Map overlap ratio: {overlap_ratio:.2%}")
            
            # Reset for next episode
            state = robot1.reset()
            robot2.reset()
        
        # Print and save overall results
        print("\nOverall Test Results:")
        results = {
            'Average steps': f"{np.mean(episode_stats['steps']):.2f}",
            'Average exploration progress': f"{np.mean(episode_stats['exploration_progress']):.1%}",
            'Average Robot1 path length': f"{np.mean(episode_stats['robot1_path_length']):.2f}",
            'Average Robot2 path length': f"{np.mean(episode_stats['robot2_path_length']):.2f}",
            'Average map overlap ratio': f"{np.mean(episode_stats['overlap_ratios']):.2%}"
        }
        
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # Save statistics
        with open(os.path.join(base_output_dir, 'test_statistics.txt'), 'w') as f:
            f.write("Overall Test Results:\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        plt.close('all')  # Close any remaining figures
        if map_tracker is not None:
            map_tracker.cleanup()
        for robot in [robot1, robot2]:
            if robot is not None and hasattr(robot, 'cleanup_visualization'):
                robot.cleanup_visualization()

def find_latest_model():
    """Find the latest A2C model in the MODEL_DIR"""
    # Look for model files with the pattern 'multi_robot_model_ac_ep'
    try:
        model_files = []
        for f in os.listdir(MODEL_DIR):
            if f.startswith('multi_robot_model_ac_ep'):
                if '_actor' in f or '_critic' in f:
                    base_name = f.split('_actor')[0].split('_critic')[0]
                    model_files.append(base_name)
                else:
                    model_files.append(f)
        
        # Remove duplicates
        model_files = list(set(model_files))
        
        if not model_files:
            print(f"Error: No A2C model files found in {MODEL_DIR}")
            return None
        
        # Extract episode numbers
        episode_nums = []
        for f in model_files:
            # Extract episode number after 'ep'
            try:
                ep_str = f.split('ep')[1]
                episode_nums.append(int(ep_str))
            except (IndexError, ValueError):
                continue
        
        if not episode_nums:
            return None
        
        # Find the highest episode number
        latest_ep = max(episode_nums)
        
        # Construct the base model path
        model_path = os.path.join(MODEL_DIR, f'multi_robot_model_ac_ep{str(latest_ep).zfill(6)}')
        
        return model_path
    except Exception as e:
        print(f"Error finding model: {str(e)}")
        return None

def main():
    # Find latest model file
    model_path = find_latest_model()
    
    if model_path is None:
        # Alternative: Try to use the base model
        model_path = os.path.join(MODEL_DIR, 'multi_robot_model_ac')
        if not os.path.exists(model_path + '_actor'):
            print(f"Error: No model files found in {MODEL_DIR}")
            return
    
    # Create fake config.json if it doesn't exist
    config_path = model_path + '_config.json'
    if not os.path.exists(config_path):
        print(f"Config file not found, creating default config: {config_path}")
        import json
        default_config = {
            'input_shape': MODEL_CONFIG['input_shape'],
            'max_frontiers': MODEL_CONFIG['max_frontiers'],
            'd_model': 128,
            'num_heads': 4,
            'dff': 256,
            'dropout_rate': 0.1
        }
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not create config file: {str(e)}")
    
    # Check if both actor and critic models exist
    if not (os.path.exists(model_path + '_actor') and os.path.exists(model_path + '_critic')):
        print(f"Error: Incomplete model at {model_path}")
        return
    
    print(f"Using model: {os.path.basename(model_path)}")
    print(f"Model path: {model_path}")
    
    # Specify model path for testing
    test_model(model_path, num_episodes=5)

if __name__ == '__main__':
    main()