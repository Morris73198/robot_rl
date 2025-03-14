import os
# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 禁用 GPU
import sys
import numpy as np
import matplotlib
# 在導入 pyplot 之前設置後端為 Agg
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial
from two_robot_cnndqn_attention.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_cnndqn_attention.environment.multi_robot_with_unknown import Robot
from two_robot_cnndqn_attention.config import MODEL_CONFIG, MODEL_DIR

def save_plot(robot, step, output_path):
    """儲存單個機器人的繪圖
    
    參數:
        robot: 機器人實例
        step: 當前步驟數
        output_path: 儲存繪圖的路徑
    """
    # 為每個繪圖創建新的圖形
    plt.figure(figsize=(10, 10))
    robot.plot_env()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')  # 關閉所有圖形以釋放記憶體

def create_robots_with_specific_map(map_file_path, train=False, plot=True):
    """創建使用特定地圖檔案的機器人
    
    參數:
        map_file_path: 指定地圖檔案的完整路徑
        train: 是否處於訓練模式
        plot: 是否繪製可視化
    
    返回:
        robot1, robot2: 兩個共享環境的機器人實例
    """
    # 創建一個自定義的 Robot 類繼承原始的 Robot 類
    class CustomMapRobot(Robot):
        @classmethod
        def create_shared_robots_with_map(cls, map_file_path, train=False, plot=True):
            """創建共享環境的機器人實例，使用指定的地圖檔案"""
            print(f"使用指定地圖創建共享環境的機器人: {map_file_path}")
            
            # 創建第一個機器人，它會載入和初始化地圖
            robot1 = cls(0, train, plot, is_primary=True)
            
            # 重寫地圖載入
            robot1.global_map, robot1.initial_positions = robot1.map_setup(map_file_path)
            
            # 設置機器人位置
            robot1.robot_position = robot1.initial_positions[0].astype(np.int64)
            robot1.other_robot_position = robot1.initial_positions[1].astype(np.int64)
            
            # 重新初始化其他必要的屬性
            robot1.op_map = np.ones(robot1.global_map.shape) * 127
            robot1.map_size = np.shape(robot1.global_map)
            robot1.t = robot1.map_points(robot1.global_map)
            robot1.free_tree = spatial.KDTree(robot1.free_points(robot1.global_map).tolist())
            
            # 創建第二個機器人，共享第一個機器人的地圖和相關資源
            robot2 = cls(0, train, plot, is_primary=False, shared_env=robot1)
            
            robot1.other_robot = robot2
            
            # 重新初始化路徑紀錄
            if plot:
                # 清理舊的可視化
                if hasattr(robot1, 'fig'):
                    plt.close(robot1.fig)
                if hasattr(robot2, 'fig'):
                    plt.close(robot2.fig)
                
                # 初始化可視化
                robot1.initialize_visualization()
                robot2.initialize_visualization()
            
            return robot1, robot2
    
    # 使用自定義機器人類創建機器人
    return CustomMapRobot.create_shared_robots_with_map(map_file_path, train, plot)

def test_model(model_path, map_file_path=None, num_episodes=5):
    """測試模型並儲存探索視覺化
    
    參數:
        model_path: 已訓練模型的路徑
        map_file_path: 要使用的特定地圖檔案的路徑（如果為None，將使用預設地圖）
        num_episodes: 要運行的回合數
    """
    robot1, robot2 = None, None
    
    try:
        # 創建基本輸出目錄
        base_output_dir = 'result_attention'
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
            
        # 載入模型
        print("正在載入模型:", model_path)
        model = MultiRobotNetworkModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        model.load(model_path)
        
        # 創建共享環境 - 使用特定地圖或預設地圖
        print("正在初始化測試環境...")
        if map_file_path:
            print(f"使用特定地圖檔案: {map_file_path}")
            robot1, robot2 = create_robots_with_specific_map(
                map_file_path,
                train=False,
                plot=True
            )
        else:
            print("使用預設測試地圖")
            robot1, robot2 = Robot.create_shared_robots(
                index_map=0,
                train=False,
                plot=True
            )
        
        # 追蹤統計信息
        episode_stats = {
            'exploration_progress': [],
            'steps': [],
            'robot1_path_length': [],
            'robot2_path_length': []
        }
        
        for episode in range(num_episodes):
            print(f"\n開始第 {episode + 1}/{num_episodes} 回合")
            
            # 創建回合目錄
            episode_dir = os.path.join(base_output_dir, f'episode_{episode+1:02d}')
            if not os.path.exists(episode_dir):
                os.makedirs(episode_dir)
            
            # 重置環境並初始化回合
            state = robot1.begin()
            robot2.begin()
            
            # 儲存初始狀態
            save_plot(robot1, 0, os.path.join(episode_dir, f'robot1_step_0000.png'))
            save_plot(robot2, 0, os.path.join(episode_dir, f'robot2_step_0000.png'))
            
            steps = 0
            robot1_path_length = 0
            robot2_path_length = 0
            
            while not (robot1.check_done() or robot2.check_done()):
                frontiers = robot1.get_frontiers()
                if len(frontiers) == 0:
                    break
                    
                # 獲取當前位置並正規化
                robot1_pos = robot1.get_normalized_position()
                robot2_pos = robot2.get_normalized_position()
                old_robot1_pos = robot1.robot_position.copy()
                old_robot2_pos = robot2.robot_position.copy()
                
                # 正規化目標
                map_dims = np.array([float(robot1.map_size[1]), float(robot1.map_size[0])])
                robot1_target = (np.zeros(2) if robot1.current_target_frontier is None 
                               else robot1.current_target_frontier / map_dims)
                robot2_target = (np.zeros(2) if robot2.current_target_frontier is None 
                               else robot2.current_target_frontier / map_dims)
                
                # 準備模型輸入並獲取預測
                predictions = model.predict(
                    np.expand_dims(state, 0),
                    np.expand_dims(model.pad_frontiers(frontiers), 0),
                    np.expand_dims(robot1_pos, 0),
                    np.expand_dims(robot2_pos, 0),
                    np.expand_dims(robot1_target, 0),
                    np.expand_dims(robot2_target, 0)
                )
                
                # 選擇並執行動作
                valid_frontiers = min(MODEL_CONFIG['max_frontiers'], len(frontiers))
                robot1_action = np.argmax(predictions['robot1'][0, :valid_frontiers])
                robot2_action = np.argmax(predictions['robot2'][0, :valid_frontiers])
                
                robot1_target = frontiers[robot1_action]
                robot2_target = frontiers[robot2_action]
                
                # 移動機器人
                next_state1, r1, d1 = robot1.move_to_frontier(robot1_target)
                robot2.op_map = robot1.op_map.copy()
                
                next_state2, r2, d2 = robot2.move_to_frontier(robot2_target)
                robot1.op_map = robot2.op_map.copy()
                
                # 更新位置
                robot1.other_robot_position = robot2.robot_position.copy()
                robot2.other_robot_position = robot1.robot_position.copy()
                
                # 更新路徑長度
                robot1_path_length += np.linalg.norm(robot1.robot_position - old_robot1_pos)
                robot2_path_length += np.linalg.norm(robot2.robot_position - old_robot2_pos)
                
                state = next_state1
                steps += 1
                
                # 每 10 步儲存繪圖
                if steps % 10 == 0:
                    save_plot(robot1, steps, 
                            os.path.join(episode_dir, f'robot1_step_{steps:04d}.png'))
                    save_plot(robot2, steps, 
                            os.path.join(episode_dir, f'robot2_step_{steps:04d}.png'))
            
            # 儲存最終狀態
            save_plot(robot1, steps, 
                     os.path.join(episode_dir, f'robot1_final_step_{steps:04d}.png'))
            save_plot(robot2, steps, 
                     os.path.join(episode_dir, f'robot2_final_step_{steps:04d}.png'))
            
            # 記錄統計信息
            final_progress = robot1.get_exploration_progress()
            episode_stats['exploration_progress'].append(final_progress)
            episode_stats['steps'].append(steps)
            episode_stats['robot1_path_length'].append(robot1_path_length)
            episode_stats['robot2_path_length'].append(robot2_path_length)
            
            print(f"第 {episode + 1} 回合結果:")
            print(f"步數: {steps}")
            print(f"最終探索進度: {final_progress:.1%}")
            print(f"Robot1 路徑長度: {robot1_path_length:.2f}")
            print(f"Robot2 路徑長度: {robot2_path_length:.2f}")
            
            # 為下一回合重置
            if map_file_path is None:
                # 使用預設的 reset 方法（將切換到下一個地圖）
                state = robot1.reset()
                robot2.reset()
            else:
                # 對於指定地圖，我們要重新初始化相同的地圖
                # 這裡我們重新創建機器人以確保使用相同的地圖
                robot1, robot2 = create_robots_with_specific_map(
                    map_file_path,
                    train=False,
                    plot=True
                )
                state = robot1.begin()
                robot2.begin()
        
        # 列印並儲存整體結果
        print("\n整體測試結果:")
        results = {
            '平均步數': f"{np.mean(episode_stats['steps']):.2f}",
            '平均探索進度': f"{np.mean(episode_stats['exploration_progress']):.1%}",
            '平均 Robot1 路徑長度': f"{np.mean(episode_stats['robot1_path_length']):.2f}",
            '平均 Robot2 路徑長度': f"{np.mean(episode_stats['robot2_path_length']):.2f}"
        }
        
        for key, value in results.items():
            print(f"{key}: {value}")
        
        # 儲存統計信息
        with open(os.path.join(base_output_dir, 'test_statistics.txt'), 'w') as f:
            f.write("整體測試結果:\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
    except Exception as e:
        print(f"測試過程中出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理
        plt.close('all')  # 關閉所有剩餘的圖形
        for robot in [robot1, robot2]:
            if robot is not None and hasattr(robot, 'cleanup_visualization'):
                robot.cleanup_visualization()

def main():
    # 指定模型路徑
    model_path = os.path.join(MODEL_DIR, 'best0111.h5')
    
    if not os.path.exists(model_path):
        print(f"錯誤: 在 {model_path} 找不到模型檔案")
        exit(1)
    
    print(f"模型路徑: {model_path}")
    
    # 指定你想要評估的特定地圖檔案
    # 以下是一個示例路徑，請替換為你想要使用的實際地圖檔案路徑
    specific_map_path = os.path.join(os.getcwd(), 'data', 'DungeonMaps', 'test', 'img_6032b.png')
    # 檢查指定的地圖檔案是否存在
    if not os.path.exists(specific_map_path):
        print(f"警告: 在 {specific_map_path} 找不到指定的地圖檔案")
        print("請提供正確的地圖檔案路徑。")
        print("繼續使用預設測試地圖...")
        specific_map_path = None
    
    # 設置要運行的回合數量
    num_episodes = 3
    
    # 運行測試，使用指定的地圖
    test_model(model_path, map_file_path=specific_map_path, num_episodes=num_episodes)

if __name__ == '__main__':
    main()