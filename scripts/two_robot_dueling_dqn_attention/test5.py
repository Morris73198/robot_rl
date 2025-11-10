import os
# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import numpy as np
import matplotlib
# 設置 matplotlib 後端為 Agg (無需顯示)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial
import csv
from two_robot_dueling_dqn_attention.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_dueling_dqn_attention.environment.multi_robot_no_unknown import Robot
from two_robot_dueling_dqn_attention.config import MODEL_CONFIG, MODEL_DIR
from two_robot_dueling_dqn_attention.environment.robot_local_map_tracker import RobotIndividualMapTracker

def save_plot(robot, step, output_path):
    """儲存單個機器人的繪圖
    
    參數:
        robot: 機器人實例
        step: 當前步驟數
        output_path: 儲存繪圖的路徑
    """
    plt.figure(figsize=(10, 10))
    robot.plot_env()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')  # 關閉所有圖形以釋放記憶體

def create_robots_with_custom_positions(map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
    """創建使用特定地圖檔案和自定義起始位置的機器人
    
    參數:
        map_file_path: 指定地圖檔案的完整路徑
        robot1_pos: 機器人1的起始位置 [x, y]，如果為None則使用預設位置
        robot2_pos: 機器人2的起始位置 [x, y]，如果為None則使用預設位置
        train: 是否處於訓練模式
        plot: 是否繪製可視化
    
    返回:
        robot1, robot2: 兩個共享環境的機器人實例
    """
    # 創建一個自定義的 Robot 類繼承原始的 Robot 類
    class CustomRobot(Robot):
        @classmethod
        def create_shared_robots_with_custom_setup(cls, map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
            """創建共享環境的機器人實例，使用指定的地圖檔案和起始位置"""
            print(f"使用指定地圖創建共享環境的機器人: {map_file_path}")
            if robot1_pos is not None:
                print(f"機器人1自定義起始位置: {robot1_pos}")
            if robot2_pos is not None:
                print(f"機器人2自定義起始位置: {robot2_pos}")
            
            # 創建第一個機器人，它會載入和初始化地圖
            robot1 = cls(0, train, plot, is_primary=True)
            
            # 載入地圖但不使用預設起始位置
            global_map, initial_positions = robot1.map_setup(map_file_path)
            robot1.global_map = global_map
            
            # 建立自由空間KD樹 (用於確保起始位置在可行區域)
            robot1.t = robot1.map_points(global_map)
            robot1.free_tree = spatial.KDTree(robot1.free_points(global_map).tolist())
            
            # 設置機器人位置
            if robot1_pos is not None and robot2_pos is not None:
                # 驗證位置是否有效 (在地圖範圍內且不是障礙物)
                robot1_pos = np.array(robot1_pos, dtype=np.int64)
                robot2_pos = np.array(robot2_pos, dtype=np.int64)
                
                # 確保位置在地圖範圍內
                map_height, map_width = global_map.shape
                robot1_pos[0] = np.clip(robot1_pos[0], 0, map_width-1)
                robot1_pos[1] = np.clip(robot1_pos[1], 0, map_height-1)
                robot2_pos[0] = np.clip(robot2_pos[0], 0, map_width-1)
                robot2_pos[1] = np.clip(robot2_pos[1], 0, map_height-1)
                
                # 確保位置不在障礙物上
                if global_map[robot1_pos[1], robot1_pos[0]] == 1:
                    print("警告: 機器人1的指定位置是障礙物，將移至最近的自由空間")
                    robot1_pos = robot1.nearest_free(robot1.free_tree, robot1_pos)
                    
                if global_map[robot2_pos[1], robot2_pos[0]] == 1:
                    print("警告: 機器人2的指定位置是障礙物，將移至最近的自由空間")
                    robot2_pos = robot1.nearest_free(robot1.free_tree, robot2_pos)
                
                # 使用自定義起始位置
                robot1.robot_position = robot1_pos
                robot1.other_robot_position = robot2_pos
            else:
                # 使用地圖中的預設位置或隨機選擇位置
                robot1.robot_position = initial_positions[0].astype(np.int64)
                robot1.other_robot_position = initial_positions[1].astype(np.int64)
            
            # 初始化其他屬性
            robot1.op_map = np.ones(global_map.shape) * 127
            robot1.map_size = np.shape(global_map)
            
            # 創建第二個機器人，共享第一個機器人的地圖和相關資源
            robot2 = cls(0, train, plot, is_primary=False, shared_env=robot1)
            
            robot1.other_robot = robot2
            
            # 重新初始化可視化
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
    return CustomRobot.create_shared_robots_with_custom_setup(
        map_file_path, 
        robot1_pos=robot1_pos, 
        robot2_pos=robot2_pos, 
        train=train, 
        plot=plot
    )

def test_multiple_start_points(model_path, map_file_path, start_points_list, output_dir='results_multi_startpoints'):
    """測試多個起始點位置
    
    參數:
        model_path: 已訓練模型的路徑
        map_file_path: 要使用的特定地圖檔案的路徑
        start_points_list: 包含多個起始點的列表，每個元素是 [robot1_pos, robot2_pos]
        output_dir: 輸出目錄
    """
    
    # === 新增：懲罰參數 ===
    # 懲罰距離：如果一個 frontier 在另一個機器人目標的這個距離內，Q-value 將被大幅降低
    # 單位是像素，您可以調整這個值
    PENALTY_DISTANCE = 100.0 
    
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 創建CSV檔案來記錄每個步驟的覆蓋率數據
    csv_path = os.path.join(output_dir, 'coverage_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['StartPoint', 'Step', 'Robot1Coverage', 'Robot2Coverage', 'IntersectionCoverage', 'UnionCoverage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # 載入模型
    print("正在載入模型:", model_path)
    model = MultiRobotNetworkModel(
        input_shape=MODEL_CONFIG['input_shape'],
        max_frontiers=MODEL_CONFIG['max_frontiers']
    )
    model.load(model_path)
    
    # 為每個起始點運行測試
    for start_idx, (robot1_pos, robot2_pos) in enumerate(start_points_list):
        print(f"\n===== 測試起始點 {start_idx+1}/{len(start_points_list)} =====")
        print(f"機器人1起始位置: {robot1_pos}")
        print(f"機器人2起始位置: {robot2_pos}")
        
        # 創建當前起始點的輸出目錄
        current_output_dir = os.path.join(output_dir, f'start_point_{start_idx+1}')
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        
        # 創建個人地圖目錄
        individual_maps_dir = os.path.join(current_output_dir, 'individual_maps')
        if not os.path.exists(individual_maps_dir):
            os.makedirs(individual_maps_dir)
        
        # 創建共享環境的機器人
        robot1, robot2 = create_robots_with_custom_positions(
            map_file_path,
            robot1_pos=robot1_pos,
            robot2_pos=robot2_pos,
            train=False,
            plot=True
        )
        
        # 創建地圖追蹤器
        tracker = RobotIndividualMapTracker(
            robot1, 
            robot2, 
            save_dir=individual_maps_dir
        )
        
        try:
            # 初始化環境
            state = robot1.begin()
            robot2.begin()
            
            # 開始追蹤個人地圖
            tracker.start_tracking()
            
            # 儲存初始狀態
            save_plot(robot1, 0, os.path.join(current_output_dir, 'robot1_step_0000.png'))
            save_plot(robot2, 0, os.path.join(current_output_dir, 'robot2_step_0000.png'))
            
            # 更新並儲存初始個人地圖
            tracker.update()
            tracker.save_current_maps(0)
            
            steps = 0
            intersection_data = []
            
            while not (robot1.check_done() or robot2.check_done()):
                frontiers = robot1.get_frontiers()
                if len(frontiers) == 0:
                    break
                
                # 獲取當前位置並規範化
                robot1_pos_norm = robot1.get_normalized_position()
                robot2_pos_norm = robot2.get_normalized_position()
                old_robot1_pos = robot1.robot_position.copy()
                old_robot2_pos = robot2.robot_position.copy()
                
                # 規範化目標
                map_dims = np.array([float(robot1.map_size[1]), float(robot1.map_size[0])])
                robot1_target_norm = (np.zeros(2) if robot1.current_target_frontier is None 
                                      else robot1.current_target_frontier / map_dims)
                robot2_target_norm = (np.zeros(2) if robot2.current_target_frontier is None 
                                      else robot2.current_target_frontier / map_dims)
                
                # 準備模型輸入並獲取預測
                predictions = model.predict(
                    np.expand_dims(state, 0),
                    np.expand_dims(model.pad_frontiers(frontiers), 0),
                    np.expand_dims(robot1_pos_norm, 0),
                    np.expand_dims(robot2_pos_norm, 0),
                    np.expand_dims(robot1_target_norm, 0),
                    np.expand_dims(robot2_target_norm, 0)
                )
                
                # === 修改：動作選擇與 Q-value 懲罰機制 ===
                
                valid_frontiers_count = min(MODEL_CONFIG['max_frontiers'], len(frontiers))

                # 1. 獲取原始的 Q-values 和可用的 frontiers
                #    使用 .copy() 確保我們不會修改到原始預測值
                q_values_r1 = predictions['robot1'][0, :valid_frontiers_count].copy()
                q_values_r2 = predictions['robot2'][0, :valid_frontiers_count].copy()
                
                # 確保 frontiers 是一個 NumPy 陣列
                available_frontiers = np.array(frontiers[:valid_frontiers_count])

                # 2. 獲取另一個機器人 "上一步" 選擇的目標
                target_r1_last = robot1.current_target_frontier
                target_r2_last = robot2.current_target_frontier

                # 3. 對 Robot 1 的 Q-values 應用懲罰 (基於 R2 的目標)
                if target_r2_last is not None and available_frontiers.size > 0:
                    # 計算所有可用 frontiers 到 R2 目標的距離
                    distances_to_r2_target = np.linalg.norm(available_frontiers - target_r2_last, axis=1)
                    # 找到太近的 frontiers 的索引
                    nearby_indices = np.where(distances_to_r2_target < PENALTY_DISTANCE)[0]
                    # 「大幅下降」: 減去一個非常大的數，使其不會被選中
                    if nearby_indices.size > 0:
                        q_values_r1[nearby_indices] -= 1e9 

                # 4. 對 Robot 2 的 Q-values 應用懲罰 (基於 R1 的目標)
                if target_r1_last is not None and available_frontiers.size > 0:
                    # 計算所有可用 frontiers 到 R1 目標的距離
                    distances_to_r1_target = np.linalg.norm(available_frontiers - target_r1_last, axis=1)
                    # 找到太近的 frontiers 的索引
                    nearby_indices = np.where(distances_to_r1_target < PENALTY_DISTANCE)[0]
                    # 「大幅下降」
                    if nearby_indices.size > 0:
                        q_values_r2[nearby_indices] -= 1e9 

                # 5. 從被懲罰過的 Q-values 中選擇最佳 action
                robot1_action = np.argmax(q_values_r1)
                robot2_action = np.argmax(q_values_r2)
                
                # === 懲罰機制結束 ===

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
                
                # 更新個人地圖追蹤器
                tracker.update()
                
                # 計算並記錄覆蓋率數據
                robot1_map = tracker.robot1_individual_map
                robot2_map = tracker.robot2_individual_map
                
                if robot1_map is not None and robot2_map is not None:
                    # 計算全局地圖的可探索區域總數
                    total_explorable = np.sum(robot1.global_map == 255)
                    
                    # 計算機器人1的覆蓋區域
                    robot1_explored = np.sum(robot1_map == 255)
                    robot1_coverage = robot1_explored / total_explorable if total_explorable > 0 else 0
                    
                    # 計算機器人2的覆蓋區域
                    robot2_explored = np.sum(robot2_map == 255)
                    robot2_coverage = robot2_explored / total_explorable if total_explorable > 0 else 0
                    
                    # 計算兩個機器人都探索過的區域（交集）
                    intersection = np.sum((robot1_map == 255) & (robot2_map == 255))
                    intersection_coverage = intersection / total_explorable if total_explorable > 0 else 0
                    
                    # 計算任一機器人探索過的區域（聯集）
                    union = np.sum((robot1_map == 255) | (robot2_map == 255))
                    union_coverage = union / total_explorable if total_explorable > 0 else 0
                    
                    # 記錄數據
                    intersection_data.append({
                        'step': steps,
                        'robot1_coverage': robot1_coverage,
                        'robot2_coverage': robot2_coverage,
                        'intersection_coverage': intersection_coverage,
                        'union_coverage': union_coverage
                    })
                    
                    # 寫入CSV
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({
                            'StartPoint': start_idx+1,
                            'Step': steps,
                            'Robot1Coverage': robot1_coverage,
                            'Robot2Coverage': robot2_coverage,
                            'IntersectionCoverage': intersection_coverage,
                            'UnionCoverage': union_coverage
                        })
                
                state = next_state1
                steps += 1
                
                # 每 10 步儲存繪圖和個人地圖
                if steps % 10 == 0:
                    save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_step_{steps:04d}.png'))
                    save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_step_{steps:04d}.png'))
                    
                    # 儲存當前個人地圖
                    tracker.save_current_maps(steps)
            
            # 儲存最終狀態
            save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_final_step_{steps:04d}.png'))
            save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_final_step_{steps:04d}.png'))
            
            # 儲存最終個人地圖
            tracker.save_current_maps(steps)
            
            # 生成每個起始點的覆蓋率時間變化圖表
            plt.figure(figsize=(12, 8))
            steps_data = [data['step'] for data in intersection_data]
            robot1_coverage = [data['robot1_coverage'] for data in intersection_data]
            robot2_coverage = [data['robot2_coverage'] for data in intersection_data]
            intersection_coverage = [data['intersection_coverage'] for data in intersection_data]
            union_coverage = [data['union_coverage'] for data in intersection_data]
            
            # 繪製四條曲線
            plt.plot(steps_data, robot1_coverage, 'b-', linewidth=2, label='Robot 1')
            plt.plot(steps_data, robot2_coverage, 'r-', linewidth=2, label='Robot 2')
            plt.plot(steps_data, intersection_coverage, 'g-', linewidth=2, label='Intersection')
            plt.plot(steps_data, union_coverage, 'k-', linewidth=2, label='Union')
            
            # 添加標籤和標題
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Coverage', fontsize=14)
            plt.title(f'Time-Coverage Analysis (Start Point {start_idx+1})', fontsize=16)
            
            # 添加網格和圖例
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            # 設置y軸範圍
            plt.ylim(0, 1.05)
            
            plt.savefig(os.path.join(current_output_dir, 'time_coverage_analysis.png'), dpi=300)
            plt.close()
            
            # 停止追蹤
            tracker.stop_tracking()
            
            # 清理資源
            tracker.cleanup()
            
            # 清理機器人資源
            for robot in [robot1, robot2]:
                if robot is not None and hasattr(robot, 'cleanup_visualization'):
                    robot.cleanup_visualization()
            
            print(f"完成起始點 {start_idx+1} 的測試，總步數: {steps}")
            
        except Exception as e:
            print(f"測試過程中出錯: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 確保清理資源
            try:
                if tracker is not None:
                    tracker.cleanup()
                
                for robot in [robot1, robot2]:
                    if robot is not None and hasattr(robot, 'cleanup_visualization'):
                        robot.cleanup_visualization()
            except:
                pass
    
    # 生成所有起始點的比較圖表
    try:
        # 從CSV檔案讀取所有數據
        all_data = {}
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                start_point = int(row['StartPoint'])
                step = int(row['Step'])
                
                if start_point not in all_data:
                    all_data[start_point] = {}
                    all_data[start_point]['steps'] = []
                    all_data[start_point]['robot1'] = []
                    all_data[start_point]['robot2'] = []
                    all_data[start_point]['intersection'] = []
                    all_data[start_point]['union'] = []
                
                all_data[start_point]['steps'].append(step)
                all_data[start_point]['robot1'].append(float(row['Robot1Coverage']))
                all_data[start_point]['robot2'].append(float(row['Robot2Coverage']))
                all_data[start_point]['intersection'].append(float(row['IntersectionCoverage']))
                all_data[start_point]['union'].append(float(row['UnionCoverage']))
        
        # 為每個起始點創建單獨的全面覆蓋率圖表
        for start_point in sorted(all_data.keys()):
            plt.figure(figsize=(12, 8))
            
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['robot1'], 
                     'b-', linewidth=2, label='Robot 1')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['robot2'], 
                     'r-', linewidth=2, label='Robot 2')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['intersection'], 
                     'g-', linewidth=2, label='Intersection')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['union'], 
                     'k-', linewidth=2, label='Union')
            
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Coverage', fontsize=14)
            plt.title(f'Time-Coverage Analysis for Start Point {start_point}', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.ylim(0, 1.05)
            
            plt.savefig(os.path.join(output_dir, f'time_coverage_startpoint_{start_point}.png'), dpi=300)
            plt.close()
        
        # 創建聯集覆蓋率比較圖表
        plt.figure(figsize=(12, 8))
        
        for start_point in sorted(all_data.keys()):
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['union'], 
                     linewidth=2, label=f'Start Point {start_point}')
        
        plt.xlabel('Time (steps)', fontsize=14)
        plt.ylabel('Total Coverage (Union)', fontsize=14)
        plt.title('Total Coverage Comparison Across Different Start Points', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.ylim(0, 1.05)
        
        plt.savefig(os.path.join(output_dir, 'all_total_coverage_comparison.png'), dpi=300)
        plt.close()
        
        print(f"已生成所有起始點的覆蓋率分析圖表")
        
    except Exception as e:
        print(f"生成比較圖表時出錯: {str(e)}")
    
    print(f"\n===== 完成所有起始點的測試 =====")
    print(f"結果儲存在: {output_dir}")
    print(f"覆蓋率數據儲存在: {csv_path}")
    print(f"測試完成！共生成了 {len(start_points_list)} 組不同起始點的時間-覆蓋率分析圖表")

def main():
    # 指定模型路徑
    model_path = os.path.join(MODEL_DIR, 'dueling_attention.h5')
    
    if not os.path.exists(model_path):
        print(f"錯誤: 在 {model_path} 找不到模型檔案")
        exit(1)
    
    print(f"模型路徑: {model_path}")
    
    # 指定地圖檔案路徑
    map_file_path = os.path.join(os.getcwd(), 'data', 'DungeonMaps', 'test', 'ttttttt.png')
    
    # 檢查地圖檔案是否存在
    if not os.path.exists(map_file_path):
        print(f"警告: 在 {map_file_path} 找不到指定的地圖檔案")
        print("請提供正確的地圖檔案路徑。")
        exit(1)
    
    # === 建議：使用 "協同" 起始點 ===
    # 每個場景中, 機器人 1 和 2 的位置不同
    # 這能更有效地測試模型的協同能力
    start_points = [
        [[100, 100], [100, 100]],  # 起始點 1
        [[520, 120], [520, 120]],  # 起始點 2
        [[630, 150], [630, 150]],   # 起始點 3
        [[250, 130], [250, 130]],   # 起始點 4
        [[250, 100], [250, 100]],  # 起始點 5
        [[400, 120], [400, 120]],  # 起始點 6
        [[140, 410], [140, 410]],   # 起始點 7
        [[110, 590], [110, 590]],   # 起始點 8
        [[900, 300], [90, 300]],   # 起始點 9
        [[260, 200], [260, 200]],  # 起始點 10
    ]
    
    # 設置輸出目錄
    output_dir = 'results_multi_startpoints_penalized'
    
    # 運行測試
    test_multiple_start_points(model_path, map_file_path, start_points, output_dir)

if __name__ == '__main__':
    main()