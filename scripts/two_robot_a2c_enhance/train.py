import os
import tensorflow as tf

import sys
# Fix the imports to match the actual class names in the modules
from two_robot_a2c_enhance.models.multi_robot_network import EnhancedMultiRobotA2CModel
from two_robot_a2c_enhance.models.multi_robot_trainer import EnhancedMultiRobotA2CTrainer
from two_robot_a2c_enhance.environment.multi_robot_no_unknown import Robot
from two_robot_a2c_enhance.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

def main():
    try:
        # Specify model path
        model_path = os.path.join(MODEL_DIR, 'enhanced_multi_robot_model_a2c_latest.h5')
        
        if os.path.exists(model_path):
            print(f"Loading A2C model: {model_path}")
            # Create model and load weights
            model = EnhancedMultiRobotA2CModel(
                input_shape=MODEL_CONFIG['input_shape'],
                max_frontiers=MODEL_CONFIG['max_frontiers']
            )
            model.load(model_path)
            
            # Create shared environment with two robots
            print("Creating robots...")
            robot1, robot2 = Robot.create_shared_robots(
                index_map=0, 
                train=True, 
                plot=True
            )
            
            # Create trainer
            print("Creating A2C trainer...")
            trainer = EnhancedMultiRobotA2CTrainer(  # Fixed class name
                model=model,
                robot1=robot1,
                robot2=robot2,
                memory_size=MODEL_CONFIG['memory_size'],
                batch_size=MODEL_CONFIG['batch_size'],
                gamma=MODEL_CONFIG['gamma']
            )
            
            # Adjust exploration parameters
            trainer.epsilon = 0.35          # Current epsilon value
            trainer.epsilon_min = 0.05     # Minimum epsilon value
            trainer.epsilon_decay = 0.99995  # Epsilon decay rate
            
            print(f"Starting training... (Current epsilon: {trainer.epsilon})")
            trainer.train(
                episodes=TRAIN_CONFIG['episodes'],
                save_freq=TRAIN_CONFIG['save_freq']
            )
        else:
            print(f"Model file not found at {model_path}")
            print("Starting new training...")
            
            print("Creating enhanced A2C model...")
            model = EnhancedMultiRobotA2CModel(  # Fixed class name
                input_shape=MODEL_CONFIG['input_shape'],
                max_frontiers=MODEL_CONFIG['max_frontiers']
            )
        
            # Create shared environment with two robots
            print("Creating robots...")
            robot1, robot2 = Robot.create_shared_robots(
                index_map=0, 
                train=True, 
                plot=True
            )
            
            # Create trainer
            print("Creating A2C trainer...")
            trainer = EnhancedMultiRobotA2CTrainer(  # Fixed class name
                model=model,
                robot1=robot1,
                robot2=robot2,
                memory_size=MODEL_CONFIG['memory_size'],
                batch_size=MODEL_CONFIG['batch_size'],
                gamma=MODEL_CONFIG['gamma']
            )
            
            # Set epsilon parameters
            trainer.epsilon = 1.0           # Initial epsilon value (exploration rate)
            trainer.epsilon_min = 0.075     # Minimum epsilon value
            trainer.epsilon_decay = 0.9985  # Epsilon decay rate
            
            # Ensure model save directory exists
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
                
            print(f"Starting training... (Current epsilon: {trainer.epsilon})")
            trainer.train(
                episodes=TRAIN_CONFIG['episodes'],
                save_freq=TRAIN_CONFIG['save_freq']
            )
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()