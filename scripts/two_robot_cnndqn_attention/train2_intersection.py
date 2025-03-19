import os
import sys
import types
from two_robot_cnndqn_attention.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_cnndqn_attention.models.multi_robot_trainer import MultiRobotTrainer
from two_robot_cnndqn_attention.environment.multi_robot_no_unknown import Robot
from two_robot_cnndqn_attention.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR
from two_robot_cnndqn_attention.environment.robot_local_map_tracker import RobotIndividualMapTracker

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

def main():
    try:
        # 指定模型路徑
        model_path = os.path.join(MODEL_DIR, 'multi_robot_model_attention_ep000420.h5')
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
        trainer.epsilon = 0.4          # 設置當前的 epsilon 值 (探索率)
        trainer.epsilon_min = 0.075     # 設置最小 epsilon 值
        trainer.epsilon_decay = 0.9975 # 設置 epsilon 衰減率
        
        # 確保模型保存目錄存在
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            
        print(f"開始訓練... (當前 epsilon: {trainer.epsilon})")
        remaining_episodes = max(0, TRAIN_CONFIG['episodes'] - start_episode)
        
        # 添加保存重疊比例的列表
        overlap_percentages = []
        
        # 獲取 MultiRobotTrainer 類的 train 方法源碼
        # 因為我們不能直接看到源碼，所以這裡只能嘗試自己實現一個類似的 train 方法
        # 根據原始 train 方法的參數和行為進行修改
        
        # 開始訓練，並手動追蹤重疊比例
        map_tracker.start_tracking()
        
        for episode in range(1, remaining_episodes + 1):
            # 重置環境
            state1 = robot1.reset()
            state2 = robot2.reset()
            
            episode_reward1 = 0
            episode_reward2 = 0
            
            # 更新地圖追蹤
            map_tracker.update()
            
            done = False
            while not done:
                # 假設 trainer 有方法來執行單步訓練
                # 這裡只是示例，實際情況取決於 trainer 的實現
                # 我們假設訓練器有一個 step 方法來執行單步訓練
                # 如果 trainer 的接口不同，請調整以下代碼
                
                # 執行 trainer 的正常訓練（這裡使用原始的 train 方法）
                trainer.train(1, TRAIN_CONFIG['target_update_freq'], TRAIN_CONFIG['save_freq'])
                
                # 更新地圖追蹤
                map_tracker.update()
                
                # 檢查是否完成
                # 由於我們無法真正看到 trainer 的內部狀態，這裡我們假設一個 episode 只訓練一次
                done = True
            
            # 計算重疊比例
            overlap_ratio = map_tracker.calculate_overlap()
            overlap_percentage = overlap_ratio * 100
            
            # 打印重疊比例
            print(f"Episode {episode} - 機器人探索區域重疊: {overlap_percentage:.2f}%")
            
            # 保存重疊比例
            overlap_percentages.append(overlap_percentage)
            
            # 重置追蹤器，開始下一輪
            map_tracker.stop_tracking()
            if episode < remaining_episodes:
                map_tracker.start_tracking()
        
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
