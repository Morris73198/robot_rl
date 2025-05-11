import os
import tensorflow as tf

import sys
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
        # Specify model path for H5 format
        model_path = os.path.join(MODEL_DIR, 'multi_robot_model_a2c_latest')
        
        # Check if H5 files exist
        actor_h5_exists = os.path.exists(model_path + '_actor.h5')
        critic_h5_exists = os.path.exists(model_path + '_critic.h5')
        config_exists = os.path.exists(model_path + '_config.json')
        
        if actor_h5_exists and critic_h5_exists and config_exists:
            print(f"Loading A2C model from: {model_path}")
            # Create model and load weights
            model = EnhancedMultiRobotA2CModel(
                input_shape=MODEL_CONFIG['input_shape'],
                max_frontiers=MODEL_CONFIG['max_frontiers']
            )
            model.load(model_path)
            
            # Create shared environment robots
            print("Creating robots...")
            robot1, robot2 = Robot.create_shared_robots(
                index_map=0, 
                train=True, 
                plot=True
            )
            
            # Create trainer
            print("Creating A2C trainer...")
            trainer = EnhancedMultiRobotA2CTrainer(
                model=model,
                robot1=robot1,
                robot2=robot2,
                memory_size=MODEL_CONFIG['memory_size'],
                batch_size=MODEL_CONFIG['batch_size'],
                gamma=MODEL_CONFIG['gamma']
            )
            
            # Adjust exploration parameters
            trainer.epsilon = 0.35          # Set current epsilon value
            trainer.epsilon_min = 0.05      # Set minimum epsilon value
            trainer.epsilon_decay = 0.99995 # Set epsilon decay rate
            
            print(f"Starting training... (current epsilon: {trainer.epsilon})")
            trainer.train(
                episodes=TRAIN_CONFIG['episodes'],
                save_freq=TRAIN_CONFIG['save_freq']
            )
        else:
            print(f"No model files found at {model_path}")
            print("Starting new training...")
            
            print("Creating A2C model...")
            model = EnhancedMultiRobotA2CModel(
                input_shape=MODEL_CONFIG['input_shape'],
                max_frontiers=MODEL_CONFIG['max_frontiers']
            )
        
            # Create shared environment robots
            print("Creating robots...")
            robot1, robot2 = Robot.create_shared_robots(
                index_map=0, 
                train=True, 
                plot=True
            )
            
            # Create trainer
            print("Creating A2C trainer...")
            trainer = EnhancedMultiRobotA2CTrainer(
                model=model,
                robot1=robot1,
                robot2=robot2,
                memory_size=MODEL_CONFIG['memory_size'],
                batch_size=MODEL_CONFIG['batch_size'],
                gamma=MODEL_CONFIG['gamma']
            )
            
            # Set epsilon related parameters
            trainer.epsilon = 0.0           # Set current epsilon value (exploration rate)
            trainer.epsilon_min = 0.075     # Set minimum epsilon value
            trainer.epsilon_decay = 0.9985  # Set epsilon decay rate
            
            # Ensure model save directory exists
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
                
            print(f"Starting training... (current epsilon: {trainer.epsilon})")
            trainer.train(
                episodes=TRAIN_CONFIG['episodes'],
                save_freq=TRAIN_CONFIG['save_freq']
            )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()