import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from two_robot_cnndqn_attention.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_cnndqn_attention.environment.multi_robot import Robot
from two_robot_cnndqn_attention.config import MODEL_CONFIG, MODEL_DIR

def test_model(model_path, num_episodes=5, plot=True):
    # Initialize robots to None for proper cleanup
    robot1, robot2 = None, None
    
    try:
        # Create output directory
        base_output_dir = 'exploration_steps'
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
        
        # Initialize and load model
        print("Loading model from:", model_path)
        model = MultiRobotNetworkModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        model.load(model_path)
        
        # Create shared environment for testing
        print("Initializing test environment...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0,
            train=False,
            plot=plot
        )
        
        # Track testing statistics
        episode_stats = {
            'exploration_progress': [],
            'steps': [],
            'robot1_path_length': [],
            'robot2_path_length': [],
            'completion_time': []
        }
        
        for episode in range(num_episodes):
            print(f"\nStarting episode {episode + 1}/{num_episodes}")
            
            # Create episode directory
            episode_dir = os.path.join(base_output_dir, f'episode_{episode+1:03d}')
            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir)
            
            # Reset environment
            state = robot1.begin()
            robot2.begin()
            
            # Save initial state
            if plot:
                robot1.plot_env()
                plt.savefig(os.path.join(episode_dir, f'robot1_step_000.png'))
                robot2.plot_env()
                plt.savefig(os.path.join(episode_dir, f'robot2_step_000.png'))
            
            steps = 0
            robot1_path_length = 0
            robot2_path_length = 0
            
            while not (robot1.check_done() or robot2.check_done()):
                frontiers = robot1.get_frontiers()
                if len(frontiers) == 0:
                    break
                    
                # Get current positions and targets
                robot1_pos = robot1.get_normalized_position()
                robot2_pos = robot2.get_normalized_position()
                
                # Normalize robot targets using map dimensions
                map_width = float(robot1.map_size[1])
                map_height = float(robot1.map_size[0])
                
                robot1_target = np.zeros(2) if robot1.current_target_frontier is None else \
                              robot1.current_target_frontier / np.array([map_width, map_height])
                robot2_target = np.zeros(2) if robot2.current_target_frontier is None else \
                              robot2.current_target_frontier / np.array([map_width, map_height])
                
                # Prepare model inputs
                state_batch = np.expand_dims(state, 0)
                frontiers_batch = np.expand_dims(model.pad_frontiers(frontiers), 0)
                robot1_pos_batch = np.expand_dims(robot1_pos, 0)
                robot2_pos_batch = np.expand_dims(robot2_pos, 0)
                robot1_target_batch = np.expand_dims(robot1_target, 0)
                robot2_target_batch = np.expand_dims(robot2_target, 0)
                
                # Get model predictions
                predictions = model.predict(
                    state_batch, frontiers_batch,
                    robot1_pos_batch, robot2_pos_batch,
                    robot1_target_batch, robot2_target_batch
                )
                
                # Select actions
                valid_frontiers = min(MODEL_CONFIG['max_frontiers'], len(frontiers))
                robot1_action = np.argmax(predictions['robot1'][0, :valid_frontiers])
                robot2_action = np.argmax(predictions['robot2'][0, :valid_frontiers])
                
                # Execute actions
                robot1_target = frontiers[robot1_action]
                robot2_target = frontiers[robot2_action]
                
                # Move robots
                old_robot1_pos = robot1.robot_position.copy()
                old_robot2_pos = robot2.robot_position.copy()
                
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
                
                # Save visualization
                if plot and steps % 10 == 0:
                    robot1.plot_env()
                    plt.savefig(os.path.join(episode_dir, f'robot1_step_{steps:03d}.png'))
                    robot2.plot_env()
                    plt.savefig(os.path.join(episode_dir, f'robot2_step_{steps:03d}.png'))
                    plt.pause(0.001)
            
            # Save final state
            if plot:
                robot1.plot_env()
                plt.savefig(os.path.join(episode_dir, f'robot1_final.png'))
                robot2.plot_env()
                plt.savefig(os.path.join(episode_dir, f'robot2_final.png'))
            
            # Record episode statistics
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
        
        # Print overall results
        print("\nOverall Test Results:")
        print(f"Average steps: {np.mean(episode_stats['steps']):.2f}")
        print(f"Average exploration progress: {np.mean(episode_stats['exploration_progress']):.1%}")
        print(f"Average Robot1 path length: {np.mean(episode_stats['robot1_path_length']):.2f}")
        print(f"Average Robot2 path length: {np.mean(episode_stats['robot2_path_length']):.2f}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if plot:
            for robot in [robot1, robot2]:
                if robot is not None and hasattr(robot, 'cleanup_visualization'):
                    robot.cleanup_visualization()

def main():
    # Specify model file
    model_file = 'multi_robot_model_ep000420.h5'
    model_path = os.path.join(MODEL_DIR, model_file)
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        print(f"Available files in {MODEL_DIR}:")
        for file in os.listdir(MODEL_DIR):
            print(f"  - {file}")
        return
    
    print(f"\nUsing model: {model_file}")
    print(f"Model path: {model_path}")
    
    # Run test
    num_episodes = 5
    test_model(model_path, num_episodes=num_episodes, plot=True)

if __name__ == '__main__':
    main()