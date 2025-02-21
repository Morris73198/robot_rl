import os
import sys
import numpy as np
import matplotlib
# Set the backend to Agg before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from two_robot_cnndqn_attention.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_cnndqn_attention.environment.multi_robot import Robot
from two_robot_cnndqn_attention.config import MODEL_CONFIG, MODEL_DIR

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
    """Test the model and save exploration visualizations
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to run
    """
    robot1, robot2 = None, None
    
    try:
        # Create base output directory
        base_output_dir = 'result_attention'
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
            
        # Load model
        print("Loading model from:", model_path)
        model = MultiRobotNetworkModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        model.load(model_path)
        
        # Create shared environment
        print("Initializing test environment...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0,
            train=False,
            plot=True
        )
        
        # Track statistics
        episode_stats = {
            'exploration_progress': [],
            'steps': [],
            'robot1_path_length': [],
            'robot2_path_length': []
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
                
                # Normalize targets
                map_dims = np.array([float(robot1.map_size[1]), float(robot1.map_size[0])])
                robot1_target = (np.zeros(2) if robot1.current_target_frontier is None 
                               else robot1.current_target_frontier / map_dims)
                robot2_target = (np.zeros(2) if robot2.current_target_frontier is None 
                               else robot2.current_target_frontier / map_dims)
                
                # Prepare model inputs and get predictions
                predictions = model.predict(
                    np.expand_dims(state, 0),
                    np.expand_dims(model.pad_frontiers(frontiers), 0),
                    np.expand_dims(robot1_pos, 0),
                    np.expand_dims(robot2_pos, 0),
                    np.expand_dims(robot1_target, 0),
                    np.expand_dims(robot2_target, 0)
                )
                
                # Select and execute actions
                valid_frontiers = min(MODEL_CONFIG['max_frontiers'], len(frontiers))
                robot1_action = np.argmax(predictions['robot1'][0, :valid_frontiers])
                robot2_action = np.argmax(predictions['robot2'][0, :valid_frontiers])
                
                robot1_target = frontiers[robot1_action]
                robot2_target = frontiers[robot2_action]
                
                # Move robots
                next_state1, r1, d1 = robot1.move_to_frontier(robot1_target)
                robot2.op_map = robot1.op_map.copy()
                
                next_state2, r2, d2 = robot2.move_to_frontier(robot2_target)
                robot1.op_map = robot2.op_map.copy()
                
                # Update positions
                robot1.other_robot_position = robot2.robot_position.copy()
                robot2.other_robot_position = robot1.robot_position.copy()
                
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
            
            # Save final state
            save_plot(robot1, steps, 
                     os.path.join(episode_dir, f'robot1_final_step_{steps:04d}.png'))
            save_plot(robot2, steps, 
                     os.path.join(episode_dir, f'robot2_final_step_{steps:04d}.png'))
            
            # Record statistics
            final_progress = robot1.get_exploration_progress()
            episode_stats['exploration_progress'].append(final_progress)
            episode_stats['steps'].append(steps)
            episode_stats['robot1_path_length'].append(robot1_path_length)
            episode_stats['robot2_path_length'].append(robot2_path_length)
            
            print(f"Episode {episode + 1} Results:")
            print(f"Steps taken: {steps}")
            print(f"Final exploration progress: {final_progress:.1%}")
            print(f"Robot1 path length: {robot1_path_length:.2f}")
            print(f"Robot2 path length: {robot2_path_length:.2f}")
            
            # Reset for next episode
            state = robot1.reset()
            robot2.reset()
        
        # Print and save overall results
        print("\nOverall Test Results:")
        results = {
            'Average steps': f"{np.mean(episode_stats['steps']):.2f}",
            'Average exploration progress': f"{np.mean(episode_stats['exploration_progress']):.1%}",
            'Average Robot1 path length': f"{np.mean(episode_stats['robot1_path_length']):.2f}",
            'Average Robot2 path length': f"{np.mean(episode_stats['robot2_path_length']):.2f}"
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
        for robot in [robot1, robot2]:
            if robot is not None and hasattr(robot, 'cleanup_visualization'):
                robot.cleanup_visualization()

def main():
    # Get latest model file
    # model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    # if not model_files:
    #     print(f"Error: No model files found in {MODEL_DIR}")
    #     return
        
    # latest_model = sorted(model_files)[-1]
    # model_path = os.path.join(MODEL_DIR, latest_model)
    
    model_path = os.path.join(MODEL_DIR, 'multi_robot_model_attention_ep000420.h5')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)
    
    # print(f"\nUsing model: {latest_model}")
    print(f"Model path: {model_path}")
    
    # Run test
    test_model(model_path, num_episodes=5)

if __name__ == '__main__':
    main()