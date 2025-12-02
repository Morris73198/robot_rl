import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 強制用CPU避免GPU OOM

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial
import csv
from tensorflow.keras.models import load_model

from two_robot_dueling_dqn_attention.models.multi_robot_network import (
    MultiRobotNetworkModel,
    SpatialAttention, LayerNormalization, MultiHeadAttention, PositionalEncoding, FeedForward
)
from two_robot_dueling_dqn_attention.environment.multi_robot_no_unknown import Robot
from two_robot_dueling_dqn_attention.config import MODEL_CONFIG, MODEL_DIR
from two_robot_dueling_dqn_attention.environment.robot_local_map_tracker import RobotIndividualMapTracker

def _huber_loss(y_true, y_pred):
    import tensorflow as tf
    return tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)

def save_plot(robot, step, output_path):
    plt.figure(figsize=(10, 10))
    robot.plot_env()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')

def create_robots_with_custom_positions(map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
    class CustomRobot(Robot):
        @classmethod
        def create_shared_robots_with_custom_setup(cls, map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
            print(f"使用指定地圖創建共享環境的機器人: {map_file_path}")
            if robot1_pos is not None:
                print(f"機器人1自定義起始位置: {robot1_pos}")
            if robot2_pos is not None:
                print(f"機器人2自定義起始位置: {robot2_pos}")
            robot1 = cls(0, train, plot, is_primary=True)
            global_map, initial_positions = robot1.map_setup(map_file_path)
            robot1.global_map = global_map
            robot1.t = robot1.map_points(global_map)
            robot1.free_tree = spatial.KDTree(robot1.free_points(global_map).tolist())
            if robot1_pos is not None and robot2_pos is not None:
                robot1_pos = np.array(robot1_pos, dtype=np.int64)
                robot2_pos = np.array(robot2_pos, dtype=np.int64)
                map_height, map_width = global_map.shape
                robot1_pos[0] = np.clip(robot1_pos[0], 0, map_width-1)
                robot1_pos[1] = np.clip(robot1_pos[1], 0, map_height-1)
                robot2_pos[0] = np.clip(robot2_pos[0], 0, map_width-1)
                robot2_pos[1] = np.clip(robot2_pos[1], 0, map_height-1)
                if global_map[robot1_pos[1], robot1_pos[0]] == 1:
                    print("警告: 機器人1的指定位置是障礙物，將移至最近的自由空間")
                    robot1_pos = robot1.nearest_free(robot1.free_tree, robot1_pos)
                if global_map[robot2_pos[1], robot2_pos[0]] == 1:
                    print("警告: 機器人2的指定位置是障礙物，將移至最近的自由空間")
                    robot2_pos = robot1.nearest_free(robot1.free_tree, robot2_pos)
                robot1.robot_position = robot1_pos
                robot1.other_robot_position = robot2_pos
            else:
                robot1.robot_position = initial_positions[0].astype(np.int64)
                robot1.other_robot_position = initial_positions[1].astype(np.int64)
            robot1.op_map = np.ones(global_map.shape) * 127
            robot1.map_size = np.shape(global_map)
            robot2 = cls(0, train, plot, is_primary=False, shared_env=robot1)
            robot1.other_robot = robot2
            if plot:
                if hasattr(robot1, 'fig'):
                    plt.close(robot1.fig)
                if hasattr(robot2, 'fig'):
                    plt.close(robot2.fig)
                robot1.initialize_visualization()
                robot2.initialize_visualization()
            return robot1, robot2

    return CustomRobot.create_shared_robots_with_custom_setup(
        map_file_path, 
        robot1_pos=robot1_pos, 
        robot2_pos=robot2_pos, 
        train=train, 
        plot=plot
    )

def test_multiple_start_points(model_path, map_file_path, start_points_list, output_dir='results_multi_startpoints'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, 'coverage_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['StartPoint', 'Step', 'Robot1Coverage', 'Robot2Coverage', 'IntersectionCoverage', 'UnionCoverage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print("正在載入模型:", model_path)
    custom_objects = {
        'SpatialAttention': SpatialAttention,
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'PositionalEncoding': PositionalEncoding,
        'FeedForward': FeedForward,
        '_huber_loss': _huber_loss,
    }
    model_keras = load_model(model_path, custom_objects=custom_objects)
    model = MultiRobotNetworkModel(
        input_shape=MODEL_CONFIG['input_shape'],
        max_frontiers=MODEL_CONFIG['max_frontiers']
    )
    model.model = model_keras

    for start_idx, (robot1_pos, robot2_pos) in enumerate(start_points_list):
        print(f"\n===== 測試起始點 {start_idx+1}/{len(start_points_list)} =====")
        print(f"機器人1起始位置: {robot1_pos}")
        print(f"機器人2起始位置: {robot2_pos}")
        current_output_dir = os.path.join(output_dir, f'start_point_{start_idx+1}')
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        individual_maps_dir = os.path.join(current_output_dir, 'individual_maps')
        if not os.path.exists(individual_maps_dir):
            os.makedirs(individual_maps_dir)
        robot1, robot2 = create_robots_with_custom_positions(
            map_file_path,
            robot1_pos=robot1_pos,
            robot2_pos=robot2_pos,
            train=False,
            plot=True
        )
        tracker = RobotIndividualMapTracker(
            robot1, 
            robot2, 
            save_dir=individual_maps_dir
        )
        try:
            state = robot1.begin()
            robot2.begin()
            tracker.start_tracking()
            save_plot(robot1, 0, os.path.join(current_output_dir, 'robot1_step_0000.png'))
            save_plot(robot2, 0, os.path.join(current_output_dir, 'robot2_step_0000.png'))
            tracker.update()
            tracker.save_current_maps(0)
            steps = 0
            intersection_data = []
            while not (robot1.check_done() or robot2.check_done()):
                frontiers = robot1.get_frontiers()
                if len(frontiers) == 0:
                    break
                robot1_pos = robot1.get_normalized_position()
                robot2_pos = robot2.get_normalized_position()
                map_dims = np.array([float(robot1.map_size[1]), float(robot1.map_size[0])])
                robot1_target = (np.zeros(2) if robot1.current_target_frontier is None 
                               else robot1.current_target_frontier / map_dims)
                robot2_target = (np.zeros(2) if robot2.current_target_frontier is None 
                               else robot2.current_target_frontier / map_dims)
                predictions = model.predict(
                    np.expand_dims(state, 0),
                    np.expand_dims(model.pad_frontiers(frontiers), 0),
                    np.expand_dims(robot1_pos, 0),
                    np.expand_dims(robot2_pos, 0),
                    np.expand_dims(robot1_target, 0),
                    np.expand_dims(robot2_target, 0)
                )
                valid_frontiers = min(MODEL_CONFIG['max_frontiers'], len(frontiers))
                robot1_action = np.argmax(predictions['robot1'][0, :valid_frontiers])
                robot2_action = np.argmax(predictions['robot2'][0, :valid_frontiers])
                robot1_target = frontiers[robot1_action]
                robot2_target = frontiers[robot2_action]
                next_state1, r1, d1 = robot1.move_to_frontier(robot1_target)
                robot2.op_map = robot1.op_map.copy()
                next_state2, r2, d2 = robot2.move_to_frontier(robot2_target)
                robot1.op_map = robot2.op_map.copy()
                robot1.other_robot_position = robot2.robot_position.copy()
                robot2.other_robot_position = robot1.robot_position.copy()
                tracker.update()
                robot1_map = tracker.robot1_individual_map
                robot2_map = tracker.robot2_individual_map
                if robot1_map is not None and robot2_map is not None:
                    total_explorable = np.sum(robot1.global_map == 255)
                    robot1_explored = np.sum(robot1_map == 255)
                    robot1_coverage = robot1_explored / total_explorable if total_explorable > 0 else 0
                    robot2_explored = np.sum(robot2_map == 255)
                    robot2_coverage = robot2_explored / total_explorable if total_explorable > 0 else 0
                    intersection = np.sum((robot1_map == 255) & (robot2_map == 255))
                    intersection_coverage = intersection / total_explorable if total_explorable > 0 else 0
                    union = np.sum((robot1_map == 255) | (robot2_map == 255))
                    union_coverage = union / total_explorable if total_explorable > 0 else 0
                    intersection_data.append({
                        'step': steps,
                        'robot1_coverage': robot1_coverage,
                        'robot2_coverage': robot2_coverage,
                        'intersection_coverage': intersection_coverage,
                        'union_coverage': union_coverage
                    })
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=['StartPoint', 'Step', 'Robot1Coverage', 'Robot2Coverage', 'IntersectionCoverage', 'UnionCoverage'])
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
                if steps % 10 == 0:
                    save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_step_{steps:04d}.png'))
                    save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_step_{steps:04d}.png'))
                    tracker.save_current_maps(steps)
            save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_final_step_{steps:04d}.png'))
            save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_final_step_{steps:04d}.png'))
            tracker.save_current_maps(steps)
            plt.figure(figsize=(12, 8))
            steps_data = [data['step'] for data in intersection_data]
            robot1_coverage = [data['robot1_coverage'] for data in intersection_data]
            robot2_coverage = [data['robot2_coverage'] for data in intersection_data]
            intersection_coverage = [data['intersection_coverage'] for data in intersection_data]
            union_coverage = [data['union_coverage'] for data in intersection_data]
            plt.plot(steps_data, robot1_coverage, 'b-', linewidth=2, label='Robot 1')
            plt.plot(steps_data, robot2_coverage, 'r-', linewidth=2, label='Robot 2')
            plt.plot(steps_data, intersection_coverage, 'g-', linewidth=2, label='Intersection')
            plt.plot(steps_data, union_coverage, 'k-', linewidth=2, label='Union')
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Coverage', fontsize=14)
            plt.title(f'Time-Coverage Analysis (Start Point {start_idx+1})', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.ylim(0, 1.05)
            plt.savefig(os.path.join(current_output_dir, 'time_coverage_analysis.png'), dpi=300)
            plt.close()
            tracker.stop_tracking()
            tracker.cleanup()
            for robot in [robot1, robot2]:
                if robot is not None and hasattr(robot, 'cleanup_visualization'):
                    robot.cleanup_visualization()
            print(f"完成起始點 {start_idx+1} 的測試，總步數: {steps}")
        except Exception as e:
            print(f"測試過程中出錯: {str(e)}")
            import traceback
            traceback.print_exc()
            try:
                if tracker is not None:
                    tracker.cleanup()
                for robot in [robot1, robot2]:
                    if robot is not None and hasattr(robot, 'cleanup_visualization'):
                        robot.cleanup_visualization()
            except:
                pass
    try:
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
        for start_point in sorted(all_data.keys()):
            plt.figure(figsize=(12, 8))
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['robot1'], 'b-', linewidth=2, label='Robot 1')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['robot2'], 'r-', linewidth=2, label='Robot 2')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['intersection'], 'g-', linewidth=2, label='Intersection')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['union'], 'k-', linewidth=2, label='Union')
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Coverage', fontsize=14)
            plt.title(f'Time-Coverage Analysis for Start Point {start_point}', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.ylim(0, 1.05)
            plt.savefig(os.path.join(output_dir, f'time_coverage_startpoint_{start_point}.png'), dpi=300)
            plt.close()
        plt.figure(figsize=(12, 8))
        for start_point in sorted(all_data.keys()):
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['union'], linewidth=2, label=f'Start Point {start_point}')
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
    model_path = os.path.join(MODEL_DIR, 'dueling.keras')
    if not os.path.exists(model_path):
        print(f"錯誤: 在 {model_path} 找不到模型檔案")
        exit(1)
    print(f"模型路徑: {model_path}")
    map_file_path = os.path.join(os.getcwd(), 'data', 'DungeonMaps', 'test', 'img_6032b.png')
    if not os.path.exists(map_file_path):
        print(f"警告: 在 {map_file_path} 找不到指定的地圖檔案")
        print("請提供正確的地圖檔案路徑。")
        exit(1)
    start_points = [
        [[100, 100], [100, 100]],
        [[520, 120], [520, 120]],
        [[250, 250], [250, 250]],
        [[250, 130], [250, 130]],
        [[250, 100], [250, 100]],
        [[400, 120], [400, 120]],
        [[140, 410], [140, 410]],
        [[110, 590], [110, 590]],
        [[90, 300], [90, 300]],
        [[260, 200], [260, 200]],
    ]
    output_dir = 'results_multi_startpoints'
    test_multiple_start_points(model_path, map_file_path, start_points, output_dir)

if __name__ == '__main__':
    main()