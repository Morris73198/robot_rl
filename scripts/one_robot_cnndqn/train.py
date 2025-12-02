import os
import sys
from one_robot_cnndqn.models.network import FrontierNetworkModel
from one_robot_cnndqn.models.trainer import FrontierTrainer
from one_robot_cnndqn.environment.robot import Robot
from one_robot_cnndqn.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

def main():
    # model
    model = FrontierNetworkModel(
        input_shape=MODEL_CONFIG['input_shape'],
        max_frontiers=MODEL_CONFIG['max_frontiers']
    )
    
    # robot env
    robot = Robot(0, train=True, plot=True)
    
    # trainer
    trainer = FrontierTrainer(
        model=model,
        robot=robot,
        memory_size=MODEL_CONFIG['memory_size'],
        batch_size=MODEL_CONFIG['batch_size'],
        gamma=MODEL_CONFIG['gamma']
    )
    
    # train
    trainer.train(
        episodes=TRAIN_CONFIG['episodes'],
        target_update_freq=TRAIN_CONFIG['target_update_freq'],
        save_freq=TRAIN_CONFIG['save_freq'],
    )
    
    # save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, 'frontier_model_final.h5'))

if __name__ == '__main__':
    main()
