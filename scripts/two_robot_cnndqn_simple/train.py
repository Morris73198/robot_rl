import os
import sys
from two_robot_cnndqn_simple.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_cnndqn_simple.models.multi_robot_trainer import MultiRobotTrainer
from two_robot_cnndqn_simple.environment.multi_robot import Robot
from two_robot_cnndqn_simple.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()  # 開啟交互模式

def main():
    try:
        # 創建模型
        print("Creating model...")
        model = MultiRobotNetworkModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        
        # 創建共享環境的兩個機器人
        print("Creating robots...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0, 
            train=True, 
            plot=True
        )
        
        # 創建訓練器
        print("Creating trainer...")
        trainer = MultiRobotTrainer(
            model=model,
            robot1=robot1,
            robot2=robot2,
            memory_size=MODEL_CONFIG['memory_size'],
            batch_size=MODEL_CONFIG['batch_size'],
            gamma=MODEL_CONFIG['gamma']
        )
        
        # 確保模型保存目錄存在
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        print("Starting training...")
        # 開始訓練
        trainer.train(
            episodes=TRAIN_CONFIG['episodes'],
            target_update_freq=TRAIN_CONFIG['target_update_freq'],
            save_freq=TRAIN_CONFIG['save_freq']
        )
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
