import os
import sys
import types
from two_robot_cnndqn_attention.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_cnndqn_attention.models.multi_robot_trainer import MultiRobotTrainer
from two_robot_cnndqn_attention.environment.multi_robot import Robot
from two_robot_cnndqn_attention.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR
from two_robot_cnndqn_attention.environment.robot_local_map_tracker import RobotIndividualMapTracker

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

def main():
    try:
        # 指定模型路徑
        # model_path = os.path.join(MODEL_DIR, 'multi_robot_model_attention_ep000340.h5')
        model_path = os.path.join(MODEL_DIR, 'xxx')
        
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
            plot=False
        )
        
        # 創建機器人地圖追蹤器
        print("正在創建地圖追蹤器...")
        map_tracker = RobotIndividualMapTracker(robot1, robot2)
        
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
        trainer.epsilon_min = 0.1     # 設置最小 epsilon 值
        trainer.epsilon_decay = 0.9975 # 設置 epsilon 衰減率
        
        # 確保模型保存目錄存在
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        # 添加保存重疊比例的列表
        overlap_percentages = []
        
        # 保存原始的 train 方法
        original_train = trainer.train
        
        # 定義修改後的 train 方法，添加重疊比例計算
        def train_with_overlap_tracking(self, episodes, target_update_freq, save_freq):
            # 開始追蹤地圖
            map_tracker.start_tracking()
            
            # 保存原始的 _train_episode 方法
            original_train_episode = self._train_episode
            
            # 定義修改後的 _train_episode 方法，增加對重疊比例的計算
            def _train_episode_with_overlap(self, episode):
                # 執行原始的 _train_episode 方法
                result = original_train_episode(episode)
                
                # 在每個 episode 結束時計算重疊比例
                overlap_ratio = map_tracker.calculate_overlap()
                overlap_percentage = overlap_ratio * 100
                
                # 打印重疊比例
                print(f"Episode {episode} - 機器人探索區域重疊: {overlap_percentage:.2f}%")
                
                # 保存重疊比例
                overlap_percentages.append(overlap_percentage)
                
                # 重置追蹤器，開始下一輪
                map_tracker.stop_tracking()
                map_tracker.start_tracking()
                
                return result
            
            # 替換 _train_episode 方法
            self._train_episode = types.MethodType(_train_episode_with_overlap, self)
            
            # 調用原始的 train 方法
            result = original_train(episodes, target_update_freq, save_freq)
            
            # 恢復原始的 _train_episode 方法
            self._train_episode = original_train_episode
            
            return result
        
        # 替換訓練方法
        trainer.train = types.MethodType(train_with_overlap_tracking, trainer)
            
        print(f"開始訓練... (當前 epsilon: {trainer.epsilon})")
        remaining_episodes = max(0, TRAIN_CONFIG['episodes'] - start_episode)
        trainer.train(
            episodes=remaining_episodes,
            target_update_freq=TRAIN_CONFIG['target_update_freq'],
            save_freq=TRAIN_CONFIG['save_freq']
        )
        
        # 訓練結束後，繪製重疊比例圖表
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(overlap_percentages) + 1), overlap_percentages, 'b-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Overlap Percentage (%)')
        plt.title('Robots Exploration Overlap Percentage per Episode')
        plt.grid(True)
        plt.savefig('robots_overlap_percentage.png', dpi=300)
        plt.close()
        
        # 清理資源
        map_tracker.cleanup()
        
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
