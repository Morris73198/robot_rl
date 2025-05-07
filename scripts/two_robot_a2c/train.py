import os
import tensorflow as tf

import sys
from two_robot_a2c.models.multi_robot_network import MultiRobotA2CModel
from two_robot_a2c.models.multi_robot_trainer import MultiRobotA2CTrainer
from two_robot_a2c.environment.multi_robot_no_unknown import Robot
from two_robot_a2c.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

def main():
    try:
        # 指定模型路徑
        model_path = os.path.join(MODEL_DIR, 'multi_robot_model_a2c_latest.h5')
        
        if os.path.exists(model_path):
            print(f"正在載入A2C模型: {model_path}")
            # 創建模型並載入權重
            model = MultiRobotA2CModel(
                input_shape=MODEL_CONFIG['input_shape'],
                max_frontiers=MODEL_CONFIG['max_frontiers']
            )
            model.load(model_path)
            
            # 創建共享環境的兩個機器人
            print("正在創建機器人...")
            robot1, robot2 = Robot.create_shared_robots(
                index_map=0, 
                train=True, 
                plot=True
            )
            
            # 創建訓練器
            print("正在創建A2C訓練器...")
            trainer = MultiRobotA2CTrainer(
                model=model,
                robot1=robot1,
                robot2=robot2,
                memory_size=MODEL_CONFIG['memory_size'],
                batch_size=MODEL_CONFIG['batch_size'],
                gamma=MODEL_CONFIG['gamma']
            )
            
            # 調整探索參數
            trainer.epsilon = 0.35          # 設置當前的 epsilon 值
            trainer.epsilon_min = 0.05     # 設置最小 epsilon 值
            trainer.epsilon_decay = 0.99995  # 設置 epsilon 衰減率
            
            print(f"開始訓練... (當前 epsilon: {trainer.epsilon})")
            trainer.train(
                episodes=TRAIN_CONFIG['episodes'],
                save_freq=TRAIN_CONFIG['save_freq']
            )
        else:
            print(f"在 {model_path} 未找到模型檔案")
            print("將開始全新訓練...")
            
            print("正在創建A2C模型...")
            model = MultiRobotA2CModel(
                input_shape=MODEL_CONFIG['input_shape'],
                max_frontiers=MODEL_CONFIG['max_frontiers']
            )
        
            # 創建共享環境的兩個機器人
            print("正在創建機器人...")
            robot1, robot2 = Robot.create_shared_robots(
                index_map=0, 
                train=True, 
                plot=True
            )
            
            # 創建訓練器
            print("正在創建A2C訓練器...")
            trainer = MultiRobotA2CTrainer(
                model=model,
                robot1=robot1,
                robot2=robot2,
                memory_size=MODEL_CONFIG['memory_size'],
                batch_size=MODEL_CONFIG['batch_size'],
                gamma=MODEL_CONFIG['gamma']
            )
            
            # 設置 epsilon 相關參數
            trainer.epsilon = 1.0           # 設置當前的 epsilon 值 (探索率)
            trainer.epsilon_min = 0.075     # 設置最小 epsilon 值
            trainer.epsilon_decay = 0.9989985  # 設置 epsilon 衰減率
            
            # 確保模型保存目錄存在
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
                
            print(f"開始訓練... (當前 epsilon: {trainer.epsilon})")
            trainer.train(
                episodes=TRAIN_CONFIG['episodes'],
                save_freq=TRAIN_CONFIG['save_freq']
            )
        
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()