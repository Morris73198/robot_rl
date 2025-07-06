import os
import sys
from two_robot_a2c_enhance2.models.multi_robot_network import MultiRobotACModel
from two_robot_a2c_enhance2.models.multi_robot_trainer import MultiRobotACTrainer
from two_robot_a2c_enhance2.environment.multi_robot_no_unknown import Robot
from two_robot_a2c_enhance2.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

def main():
    try:
        # 指定模型路徑（actor和critic分開保存）
        model_path = os.path.join(MODEL_DIR, 'multi_robot_model_ac')
        
        # 創建模型
        print("Creating Actor-Critic model...")
        model = MultiRobotACModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        
        start_episode = 0
        
        # 載入已有的模型（如果存在）
        if os.path.exists(model_path + '_actor') and os.path.exists(model_path + '_critic'):
            print(f"Loading existing model from: {model_path}")
            model.load(model_path)
            # 獲取起始episode（從文件名解析）
            existing_models = [f for f in os.listdir(MODEL_DIR) if f.startswith('multi_robot_model_ac_ep')]
            if existing_models:
                latest_ep = max([int(f.split('ep')[-1].split('_')[0]) for f in existing_models])
                start_episode = latest_ep
            print(f"Continuing training from episode {start_episode}")
        else:
            print(f"No existing model found at {model_path}")
            print("Starting fresh training...")
        
        # 創建共享環境的兩個機器人
        print("Creating robots with shared environment...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0, 
            train=True, 
            plot=False
        )
        
        # 創建Actor-Critic訓練器
        print("Creating Actor-Critic trainer...")
        trainer = MultiRobotACTrainer(
            model=model,
            robot1=robot1,
            robot2=robot2,
            gamma=MODEL_CONFIG['gamma'],
            gae_lambda=0.95  # GAE lambda parameter
        )
        
        # 確保模型保存目錄存在
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        print("Starting Actor-Critic training...")
        remaining_episodes = max(0, TRAIN_CONFIG['episodes'] - start_episode)
        
        # 開始訓練
        trainer.train(
            episodes=remaining_episodes,
            save_freq=TRAIN_CONFIG['save_freq']
        )
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        plt.ioff()
        plt.close('all')

if __name__ == '__main__':
    main()
