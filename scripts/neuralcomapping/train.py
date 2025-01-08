import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from neuralcomapping.models.multi_robot_network import MultiRobotNetwork
from neuralcomapping.models.multi_robot_trainer import MultiRobotTrainer
from neuralcomapping.environment.multi_robot import Robot


def setup_gpu():
    """Setup GPU memory growth to avoid taking all memory"""
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"Found {len(physical_devices)} GPU(s), enabled memory growth")
        else:
            print("No GPU found, using CPU")
    except Exception as e:
        print(f"Error setting up GPU: {str(e)}")

def setup_logging():
    """Setup logging directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'multi_robot_gnn_{timestamp}')
    models_dir = os.path.join(log_dir, 'models')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Logging to: {log_dir}")
    return log_dir, models_dir

def create_model_and_trainer(log_dir, robots):
    """Create and initialize network and trainer"""
    # Network hyperparameters
    NETWORK_CONFIG = {
        'input_shape': (84, 84, 1),
        'num_robots': 2,
        'feature_dim': 128,
        'num_gnn_layers': 3
    }
    
    # Training hyperparameters
    TRAINER_CONFIG = {
        'num_steps': 128,
        'num_envs': 8,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'num_epochs': 10
    }
    
    # Create network
    network = MultiRobotNetwork(**NETWORK_CONFIG)
    
    # Build network by calling it once with dummy data
    batch_size = 1
    dummy_state = np.zeros((batch_size, *NETWORK_CONFIG['input_shape']), dtype=np.float32)
    dummy_frontiers = np.zeros((batch_size, NETWORK_CONFIG['num_robots'], 50, 2), dtype=np.float32)  # Updated shape
    dummy_robot_poses = np.zeros((batch_size, NETWORK_CONFIG['num_robots'], 2), dtype=np.float32)
    dummy_robot_targets = np.zeros((batch_size, NETWORK_CONFIG['num_robots'], 2), dtype=np.float32)
    
    
    
    
    print("Dummy input shapes:")
    print("- dummy_state:", dummy_state.shape)
    print("- dummy_frontiers:", dummy_frontiers.shape) 
    print("- dummy_robot_poses:", dummy_robot_poses.shape)
    print("- dummy_robot_targets:", dummy_robot_targets.shape)

    
    # Call model once to build it
    network([
        dummy_state,
        dummy_frontiers,
        dummy_robot_poses,
        dummy_robot_targets
    ], training=False)
    
    # Create trainer
    trainer = MultiRobotTrainer(
        network=network,
        robots=robots,
        log_dir=log_dir,
        **TRAINER_CONFIG
    )
    
    return network, trainer

def try_load_checkpoint(trainer, models_dir):
    """Attempt to load latest checkpoint if available"""
    checkpoints = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    if checkpoints:
        latest_checkpoint = max(checkpoints)
        checkpoint_path = os.path.join(models_dir, latest_checkpoint)
        print(f"\nLoading checkpoint: {checkpoint_path}")
        trainer.load_model(checkpoint_path)
        return True
    return False

def main():
    # Basic setup
    setup_gpu()
    log_dir, models_dir = setup_logging()
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    try:
        # Create shared environment and robots
        print("\nInitializing robots...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0,
            train=True,
            plot=True
        )
        robots = [robot1, robot2]
        
        # Create model and trainer
        print("\nInitializing network and trainer...")
        network, trainer = create_model_and_trainer(log_dir, robots)
        
        # Try to load checkpoint
        loaded = try_load_checkpoint(trainer, models_dir)
        if not loaded:
            print("Starting training from scratch")
        
        # Start training (updated parameter name)
        print("\nStarting training loop...")
        print("Press Ctrl+C to stop training\n")
        trainer.train(num_episodes=1000)  # Changed from num_updates to num_episodes
        
        # Save final model
        final_path = os.path.join(models_dir, 'final_model.h5')
        trainer.save_model(final_path)
        print(f"\nTraining completed! Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_path = os.path.join(models_dir, 'interrupted_model.h5')
        trainer.save_model(interrupted_path)
        print(f"Saved interrupted model to: {interrupted_path}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up resources
        print("\nCleaning up...")
        for robot in robots:
            if hasattr(robot, 'cleanup_visualization'):
                robot.cleanup_visualization()
        
        print("\nExiting...")

if __name__ == '__main__':
    main()