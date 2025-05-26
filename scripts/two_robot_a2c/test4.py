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
from two_robot_a2c.models.multi_robot_network import MultiRobotACModel
from two_robot_a2c.environment.multi_robot_no_unknown import Robot
from two_robot_a2c.config import ROBOT_CONFIG, MODEL_DIR
from two_robot_a2c.environment.robot_local_map_tracker import RobotIndividualMapTracker

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

def pad_frontiers(frontiers, max_frontiers=50, map_size=(600, 600)):
    """填充frontier點到固定長度並進行標準化"""
    padded = np.zeros((max_frontiers, 2))
    
    if len(frontiers) > 0:
        frontiers = np.array(frontiers)
        
        # 標準化座標
        normalized_frontiers = frontiers.copy()
        normalized_frontiers[:, 0] = frontiers[:, 0] / float(map_size[1])
        normalized_frontiers[:, 1] = frontiers[:, 1] / float(map_size[0])
        
        n_frontiers = min(len(frontiers), max_frontiers)
        padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
    
    return padded

def get_normalized_target(target, map_size=(600, 600)):
    """標準化目標位置"""
    if target is None:
        return np.array([0.0, 0.0])
    normalized = np.array([
        target[0] / float(map_size[1]),
        target[1] / float(map_size[0])
    ])
    return normalized

def test_multiple_start_points(model_path, map_file_path, start_points_list, output_dir='results_multi_startpoints_a2c'):
    """測試多個起始點位置
    
    參數:
        model_path: 已訓練模型的路徑
        map_file_path: 要使用的特定地圖檔案的路徑
        start_points_list: 包含多個起始點的列表，每個元素是 [robot1_pos, robot2_pos]
        output_dir: 輸出目錄
    """
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 創建CSV檔案來記錄每個步驟的覆蓋率數據
    csv_path = os.path.join(output_dir, 'coverage_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['StartPoint', 'Step', 'Robot1Coverage', 'Robot2Coverage', 'IntersectionCoverage', 'UnionCoverage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # 載入 A2C 模型
    print("正在載入 A2C 模型:", model_path)
    model = MultiRobotACModel(
        input_shape=(84, 84, 1),
        max_frontiers=50
    )
    
    # 檢查模型文件是否存在
    if not os.path.exists(model_path + '_actor.h5') or not os.path.exists(model_path + '_critic.h5'):
        print(f"錯誤: 找不到模型文件 {model_path}_actor.h5 或 {model_path}_critic.h5")
        return
    
    if not model.load(model_path):
        print("載入模型失敗!")
        return
    
    print("模型載入成功!")
    
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
            max_steps = 15000000  # 設置最大步數
            
            while not (robot1.check_done() or robot2.check_done()) and steps < max_steps:
                frontiers = robot1.get_frontiers()
                if len(frontiers) == 0:
                    break
                
                # 獲取當前位置並規範化
                robot1_pos_norm = robot1.get_normalized_position()
                robot2_pos_norm = robot2.get_normalized_position()
                
                # 規範化目標
                robot1_target = get_normalized_target(
                    robot1.current_target_frontier, robot1.map_size
                )
                robot2_target = get_normalized_target(
                    robot2.current_target_frontier, robot2.map_size
                )
                
                # 準備模型輸入
                state_batch = np.expand_dims(state, 0)
                frontiers_batch = np.expand_dims(pad_frontiers(frontiers, 50, robot1.map_size), 0)
                robot1_pos_batch = np.expand_dims(robot1_pos_norm, 0)
                robot2_pos_batch = np.expand_dims(robot2_pos_norm, 0)
                robot1_target_batch = np.expand_dims(robot1_target, 0)
                robot2_target_batch = np.expand_dims(robot2_target, 0)
                
                # 獲取動作預測
                try:
                    policy_predictions = model.predict_policy(
                        state_batch, frontiers_batch,
                        robot1_pos_batch, robot2_pos_batch,
                        robot1_target_batch, robot2_target_batch
                    )
                    
                    # 選擇動作
                    valid_frontiers = min(50, len(frontiers))
                    
                    # 使用貪婪策略選擇動作（測試時不需要探索）
                    robot1_probs = policy_predictions['robot1_policy'][0, :valid_frontiers]
                    robot2_probs = policy_predictions['robot2_policy'][0, :valid_frontiers]
                    
                    # 確保概率分布有效
                    if np.sum(robot1_probs) > 0:
                        robot1_action = np.argmax(robot1_probs)
                    else:
                        robot1_action = 0
                        
                    if np.sum(robot2_probs) > 0:
                        robot2_action = np.argmax(robot2_probs)
                    else:
                        robot2_action = 0
                    
                    # 確保動作在有效範圍內
                    robot1_action = min(robot1_action, valid_frontiers - 1)
                    robot2_action = min(robot2_action, valid_frontiers - 1)
                    
                except Exception as e:
                    print(f"模型預測出錯: {str(e)}")
                    # 使用隨機動作作為備用方案
                    valid_frontiers = min(50, len(frontiers))
                    robot1_action = np.random.randint(0, valid_frontiers)
                    robot2_action = np.random.randint(0, valid_frontiers)
                
                # 執行動作
                robot1_target_pos = frontiers[robot1_action]
                robot2_target_pos = frontiers[robot2_action]
                
                # 移動機器人
                next_state1, r1, d1 = robot1.move_to_frontier(robot1_target_pos)
                robot2.op_map = robot1.op_map.copy()
                
                next_state2, r2, d2 = robot2.move_to_frontier(robot2_target_pos)
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
                    
                    # 每50步打印一次進度
                    if steps % 50 == 0:
                        exploration_progress = robot1.get_exploration_progress()
                        print(f"步數: {steps}, 探索進度: {exploration_progress:.1%}")
            
            # 儲存最終狀態
            save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_final_step_{steps:04d}.png'))
            save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_final_step_{steps:04d}.png'))
            
            # 儲存最終個人地圖
            tracker.save_current_maps(steps)
            
            # 生成每個起始點的覆蓋率時間變化圖表
            if intersection_data:
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
                
                plt.savefig(os.path.join(current_output_dir, 'time_coverage_analysis.png'), dpi=300, bbox_inches='tight')
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
            
            plt.savefig(os.path.join(output_dir, f'time_coverage_startpoint_{start_point}.png'), dpi=300, bbox_inches='tight')
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
        
        plt.savefig(os.path.join(output_dir, 'all_total_coverage_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成所有起始點的覆蓋率分析圖表")
        
    except Exception as e:
        print(f"生成比較圖表時出錯: {str(e)}")
    
    print(f"\n===== 完成所有起始點的測試 =====")
    print(f"結果儲存在: {output_dir}")
    print(f"覆蓋率數據儲存在: {csv_path}")
    print(f"測試完成！共生成了 {len(start_points_list)} 組不同起始點的時間-覆蓋率分析圖表")

def main():
    # 指定 A2C 模型路徑（不包含副檔名，因為會自動添加 _actor.h5 和 _critic.h5）
    model_path = os.path.join(MODEL_DIR, 'multi_robot_model_ac_ep000800')  # 根據您的模型命名調整
    
    # 檢查模型文件是否存在
    if not os.path.exists(model_path + '_actor.h5') or not os.path.exists(model_path + '_critic.h5'):
        print(f"錯誤: 在以下位置找不到模型檔案:")
        print(f"  {model_path}_actor.h5")
        print(f"  {model_path}_critic.h5")
        print("\n請檢查以下可能的模型文件:")
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.endswith('_actor.h5'):
                    print(f"  {os.path.join(MODEL_DIR, file)}")
        exit(1)
    
    print(f"模型路徑: {model_path}")
    
    # 指定地圖檔案路徑
    map_file_path = os.path.join(os.getcwd(),  'data', 'DungeonMaps', 'test', 'img_6112b.png')
    
    # 檢查地圖檔案是否存在
    if not os.path.exists(map_file_path):
        print(f"警告: 在 {map_file_path} 找不到指定的地圖檔案")
        
        # 嘗試其他可能的路徑
        alternative_paths = [
            os.path.join(os.getcwd(), 'data', 'DungeonMaps', 'test', 'img_6012b.png'),
            os.path.join(os.getcwd(), 'two_robot_a2c', 'data', 'DungeonMaps', 'test', 'img_6060.png'),
            os.path.join(os.path.dirname(__file__), 'data', 'DungeonMaps', 'test', 'img_6060.png')
        ]
        
        found = False
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                map_file_path = alt_path
                found = True
                print(f"找到地圖檔案: {map_file_path}")
                break
        
        if not found:
            print("請提供正確的地圖檔案路徑。")
            exit(1)
    
    # 定義10個起始點位置 [robot1_pos, robot2_pos]
    # 可以根據需要調整這些坐標
    start_points = [
        [[100, 100], [100, 100]],  # 起始點 1
        [[520, 120], [520, 120]],  # 起始點 2
        [[250, 250], [250, 250]],   # 起始點 3
        [[250, 130], [250, 130]],   # 起始點 4
        [[250, 100], [250, 100]],  # 起始點 5
        [[400, 120], [400, 120]],  # 起始點 6
        [[140, 410], [140, 410]],   # 起始點 7
        [[110, 590], [110, 590]],   # 起始點 8
        [[90, 300], [90, 300]],   # 起始點 9
        [[260, 200], [260, 200]],  # 起始點 10
    ]
    
    # 設置輸出目錄
    output_dir = 'results_multi_startpoints_a2c'
    
    # 運行測試
    test_multiple_start_points(model_path, map_file_path, start_points, output_dir)

if __name__ == '__main__':
    main()