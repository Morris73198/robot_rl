import os
import sys
from two_robot_a2c_enhance.models.multi_robot_network import MultiRobotACModel
from two_robot_a2c_enhance.models.multi_robot_trainer import MultiRobotACTrainer
from two_robot_a2c_enhance.environment.multi_robot_no_unknown import Robot
from two_robot_a2c_enhance.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR
# 引入 RobotIndividualMapTracker
from two_robot_a2c_enhance.environment.robot_local_map_tracker import RobotIndividualMapTracker

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

def main():
    try:
        # 指定模型路徑（actor和critic分開保存，使用.h5格式）
        model_path = os.path.join(MODEL_DIR, 'multi_robot_model_ac')
        
        # 創建模型
        print("Creating Actor-Critic model...")
        model = MultiRobotACModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        
        start_episode = 0
        
        # 載入已有的模型（如果存在）- 檢查.h5文件
        if os.path.exists(model_path + '_actor.h5') and os.path.exists(model_path + '_critic.h5'):
            print(f"Loading existing model from: {model_path}")
            if model.load(model_path):
                # 獲取起始episode（從文件名解析）
                existing_models = [f for f in os.listdir(MODEL_DIR) if f.startswith('multi_robot_model_ac_ep') and f.endswith('_actor.h5')]
                if existing_models:
                    latest_ep = max([int(f.split('ep')[-1].split('_')[0]) for f in existing_models])
                    start_episode = latest_ep
                print(f"Continuing training from episode {start_episode}")
            else:
                print("Failed to load existing model, starting fresh training...")
        else:
            print(f"No existing .h5 model found at {model_path}")
            print("Starting fresh training...")
        
        # 創建共享環境的兩個機器人
        print("Creating robots with shared environment...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0, 
            train=True, 
            plot=False
        )
        
        # 創建機器人個人地圖追蹤器
        print("Creating robot individual map tracker...")
        map_tracker = RobotIndividualMapTracker(
            robot1=robot1,
            robot2=robot2,
            save_dir='robot_individual_maps'
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
        
        # 將地圖追蹤器添加到訓練器中
        trainer.map_tracker = map_tracker
        
        # 修改訓練器的 train 方法以使用地圖追蹤器
        original_train = trainer.train
        
        def train_with_tracker(*args, **kwargs):
            # 開始追蹤
            trainer.map_tracker.start_tracking()
            
            # 保存原始的 train_on_episode 方法
            original_train_on_episode = trainer.train_on_episode
            
            def train_on_episode_with_tracker():
                # 獲取交集百分比
                overlap_ratio = trainer.map_tracker.calculate_overlap()
                print(f"Episode overlap ratio: {overlap_ratio:.2%}")
                
                # 記錄覆蓋率圖表
                if hasattr(trainer, 'training_history'):
                    if 'overlap_ratios' not in trainer.training_history:
                        trainer.training_history['overlap_ratios'] = []
                    trainer.training_history['overlap_ratios'].append(float(overlap_ratio))
                
                # 調用原始方法
                return original_train_on_episode()
            
            # 替換方法
            trainer.train_on_episode = train_on_episode_with_tracker
            
            # 調用原始的 train 方法
            result = original_train(*args, **kwargs)
            
            # 停止追蹤並生成覆蓋率圖表
            trainer.map_tracker.stop_tracking()
            trainer.map_tracker.plot_coverage_over_time()
            
            # 恢復原始方法
            trainer.train_on_episode = original_train_on_episode
            
            return result
        
        # 替換訓練方法
        trainer.train = train_with_tracker
        
        # 修改訓練循環中的一部分，添加地圖追蹤更新
        original_reset_episode_buffer = trainer.reset_episode_buffer
        
        def reset_episode_buffer_with_tracker():
            original_reset_episode_buffer()
            # 初始化或重置地圖追蹤器
            if hasattr(trainer, 'map_tracker'):
                trainer.map_tracker.start_tracking()
        
        # 替換方法
        trainer.reset_episode_buffer = reset_episode_buffer_with_tracker
        
        # 確保模型保存目錄存在
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        print("Starting Actor-Critic training with map tracking...")
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
        # 確保清理地圖追蹤器資源
        if 'map_tracker' in locals() and map_tracker is not None:
            map_tracker.cleanup()

if __name__ == '__main__':
    main()