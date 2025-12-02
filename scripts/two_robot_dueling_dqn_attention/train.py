import os
import tensorflow as tf

import sys
from two_robot_dueling_dqn_attention.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_dueling_dqn_attention.models.multi_robot_trainer import MultiRobotTrainer
from two_robot_dueling_dqn_attention.environment.multi_robot_no_unknown import Robot
from two_robot_dueling_dqn_attention.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

def fine_tune_without_architecture_change(pretrained_path):
    """在不改變架構的情況下微調模型"""
    # 載入完全相同架構的模型
    model = MultiRobotNetworkModel(
        input_shape=MODEL_CONFIG['input_shape'],
        max_frontiers=MODEL_CONFIG['max_frontiers']
    )
    
    # 載入預訓練權重
    model.load(pretrained_path)
    
    # 只修改學習率和優化器參數
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,  # 使用較小的學習率
        decay_steps=1000,
        decay_rate=0.95
    )
    
    # 重新編譯模型，使用更適合的優化器設置
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=0.5,  # 更小的梯度裁剪值，穩定訓練
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss={
            'robot1': model._huber_loss,
            'robot2': model._huber_loss
        }
    )
    
    return model

def main():
    try:
        # 指定模型路徑
        # model_path = os.path.join(MODEL_DIR, 'xxx.h5')
        model_path = os.path.join(MODEL_DIR, 'best0111.h5')

        
        if os.path.exists(model_path):
            print(f"正在載入並微調模型: {model_path}")
            # 使用微調函數而不是直接載入
            model = fine_tune_without_architecture_change(model_path)
            
            # 創建共享環境的兩個機器人
            print("正在創建機器人...")
            robot1, robot2 = Robot.create_shared_robots(
                index_map=0, 
                train=True, 
                plot=True
            )
            
            # 創建訓練器
            print("正在創建訓練器...")
            trainer = MultiRobotTrainer(
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
                target_update_freq=TRAIN_CONFIG['target_update_freq'],
                save_freq=TRAIN_CONFIG['save_freq']
            )
        else:
            print(f"在 {model_path} 未找到模型檔案")
            print("將開始全新訓練...")
            
            print("正在創建模型...")
            model = MultiRobotNetworkModel(
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
            
            # 創建訓練器並手動設置 epsilon 相關參數
            print("正在創建訓練器...")
            trainer = MultiRobotTrainer(
                model=model,
                robot1=robot1,
                robot2=robot2,
                memory_size=MODEL_CONFIG['memory_size'],
                batch_size=MODEL_CONFIG['batch_size'],
                gamma=MODEL_CONFIG['gamma']
            )
            
            # 手動調整 epsilon 相關參數
            trainer.epsilon = 1.0          # 設置當前的 epsilon 值 (探索率)
            trainer.epsilon_min = 0.075     # 設置最小 epsilon 值
            trainer.epsilon_decay = 0.9985 # 設置 epsilon 衰減率
            
            # 確保模型保存目錄存在
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
                
            print(f"開始訓練... (當前 epsilon: {trainer.epsilon})")
            trainer.train(
                episodes=TRAIN_CONFIG['episodes'],
                target_update_freq=TRAIN_CONFIG['target_update_freq'],
                save_freq=TRAIN_CONFIG['save_freq']
            )
        
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
