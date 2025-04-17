import os
import sys
from two_robot_cnndqn_attention.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_cnndqn_attention.models.multi_robot_trainer import MultiRobotTrainer
from two_robot_cnndqn_attention.environment.multi_robot_no_unknown import Robot
from two_robot_cnndqn_attention.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

def main():
    try:
        # 指定模型路徑
        model_path = os.path.join(MODEL_DIR, 'best0111.h5')
        # model_path = os.path.join(MODEL_DIR, 'xxx')
        
        # 創建模型
        print("正在創建模型...")
        model = MultiRobotNetworkModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        
        start_episode = 0
        
        # 載入指定的模型
        if os.path.exists(model_path):
            print(f"正在載入模型: {model_path}")
            model.load(model_path)
            start_episode = 0
            print(f"將從第 {start_episode} 輪繼續訓練")
        else:
            print(f"在 {model_path} 未找到模型檔案")
            print("將開始全新訓練...")
        
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
        trainer.epsilon = 0.35          # 設置當前的 epsilon 值 (探索率)
        trainer.epsilon_min = 0.075     # 設置最小 epsilon 值
        trainer.epsilon_decay = 0.9985 # 設置 epsilon 衰減率
        
        # 確保模型保存目錄存在
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        print(f"開始訓練... (當前 epsilon: {trainer.epsilon})")
        remaining_episodes = max(0, TRAIN_CONFIG['episodes'] - start_episode)
        trainer.train(
            episodes=remaining_episodes,
            target_update_freq=TRAIN_CONFIG['target_update_freq'],
            save_freq=TRAIN_CONFIG['save_freq']
        )
        
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
