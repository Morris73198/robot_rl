import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from two_robot_cnndqn_simple.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_cnndqn_simple.environment.multi_robot import Robot
from two_robot_cnndqn_simple.config import MODEL_CONFIG, MODEL_DIR

def ensure_dir(directory):
    """確保目錄存在，如果不存在則創建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def test_model(model_path, num_episodes=5, plot=True):
    try:
        # 創建圖片保存目錄
        base_output_dir = 'result_simple'
        ensure_dir(base_output_dir)
        
        # 載入模型
        print("Loading model from:", model_path)
        model = MultiRobotNetworkModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        model.load(model_path)
        
        # 創建測試環境
        print("Initializing test environment...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0,
            train=False,
            plot=plot
        )
        
        # 測試統計
        episode_stats = {
            'exploration_progress': [],
            'steps': [],
            'robot1_path_length': [],
            'robot2_path_length': [],
            'completion_time': []
        }
        
        for episode in range(num_episodes):
            print(f"\nStarting episode {episode + 1}/{num_episodes}")
            
            # 為每個episode創建單獨的目錄
            episode_dir = os.path.join(base_output_dir, f'episode_{episode+1:03d}')
            ensure_dir(episode_dir)
            
            # 重置環境
            state = robot1.begin()
            robot2.begin()
            
            # 保存初始狀態
            if plot:
                # 保存Robot1的初始狀態
                robot1.plot_env()
                plt.savefig(os.path.join(episode_dir, f'robot1_step_000.png'))
                
                # 保存Robot2的初始狀態
                robot2.plot_env()
                plt.savefig(os.path.join(episode_dir, f'robot2_step_000.png'))
            
            steps = 0
            robot1_path_length = 0
            robot2_path_length = 0
            
            while not (robot1.check_done() or robot2.check_done()):
                frontiers = robot1.get_frontiers()
                if len(frontiers) == 0:
                    break
                    
                robot1_pos = robot1.get_normalized_position()
                robot2_pos = robot2.get_normalized_position()
                
                # 使用模型預測動作
                state_batch = np.expand_dims(state, 0)
                frontiers_batch = np.expand_dims(
                    robot1.pad_frontiers(frontiers) 
                    if hasattr(robot1, 'pad_frontiers') 
                    else np.zeros((50, 2)), 0
                )
                robot1_pos_batch = np.expand_dims(robot1_pos, 0)
                robot2_pos_batch = np.expand_dims(robot2_pos, 0)
                
                predictions = model.predict(
                    state_batch, frontiers_batch,
                    robot1_pos_batch, robot2_pos_batch
                )
                
                # 選擇動作
                MIN_TARGET_DISTANCE = 10  # 最小目標距離閾值
                valid_frontiers = min(MODEL_CONFIG['max_frontiers'], len(frontiers))
                
                # 獲取Q值
                robot1_q = predictions['robot1'][0, :valid_frontiers].copy()
                robot2_q = predictions['robot2'][0, :valid_frontiers].copy()
                
                # 根據當前目標調整Q值
                # if robot2.current_target_frontier is not None:
                #     for i in range(valid_frontiers):
                #         if np.linalg.norm(frontiers[i] - robot2.current_target_frontier) < MIN_TARGET_DISTANCE:
                #             robot1_q[i] *= 0.0001  # 大幅降低太近的目標的Q值
                            
                # if robot1.current_target_frontier is not None:
                #     for i in range(valid_frontiers):
                #         if np.linalg.norm(frontiers[i] - robot1.current_target_frontier) < MIN_TARGET_DISTANCE:
                #             robot2_q[i] *= 0.0001  # 大幅降低太近的目標的Q值
                
                # 選擇動作
                robot1_action = np.argmax(robot1_q)
                robot2_action = np.argmax(robot2_q)
                
                # 執行動作
                robot1_target = frontiers[robot1_action]
                robot2_target = frontiers[robot2_action]
                
                # Robot 1移動
                next_state1, r1, d1 = robot1.move_to_frontier(robot1_target)
                robot1_path_length += np.linalg.norm(
                    robot1.robot_position - robot1.old_position
                )
                
                # 更新地圖
                robot2.op_map = robot1.op_map.copy()
                
                # Robot 2移動
                next_state2, r2, d2 = robot2.move_to_frontier(robot2_target)
                robot2_path_length += np.linalg.norm(
                    robot2.robot_position - robot2.old_position
                )
                
                # 確保地圖同步
                robot1.op_map = robot2.op_map.copy()
                
                # 更新機器人位置信息
                robot1.other_robot_position = robot2.robot_position.copy()
                robot2.other_robot_position = robot1.robot_position.copy()
                
                state = next_state1
                steps += 1
                
                # 更新可視化並保存圖片
                if plot and steps % 10 == 0:  # 每10步保存一次
                    # 保存Robot1的狀態
                    robot1.plot_env()
                    plt.savefig(os.path.join(episode_dir, f'robot1_step_{steps:03d}.png'))
                    
                    # 保存Robot2的狀態
                    robot2.plot_env()
                    plt.savefig(os.path.join(episode_dir, f'robot2_step_{steps:03d}.png'))
                    
                    plt.pause(0.001)  # 短暫暫停以更新顯示
            
            # 保存最終狀態
            if plot:
                robot1.plot_env()
                plt.savefig(os.path.join(episode_dir, f'robot1_final.png'))
                robot2.plot_env()
                plt.savefig(os.path.join(episode_dir, f'robot2_final.png'))
            
            # 記錄episode統計
            final_progress = robot1.get_exploration_progress()
            episode_stats['exploration_progress'].append(final_progress)
            episode_stats['steps'].append(steps)
            episode_stats['robot1_path_length'].append(robot1_path_length)
            episode_stats['robot2_path_length'].append(robot2_path_length)
            
            print(f"Episode {episode + 1} Results:")
            print(f"Steps taken: {steps}")
            print(f"Final exploration progress: {final_progress:.1%}")
            print(f"Robot1 path length: {robot1_path_length:.2f}")
            print(f"Robot2 path length: {robot2_path_length:.2f}")
            
            # 重置環境
            state = robot1.reset()
            robot2.reset()
        
        # 顯示總體統計
        print("\nOverall Test Results:")
        print(f"Average steps: {np.mean(episode_stats['steps']):.2f}")
        print(f"Average exploration progress: {np.mean(episode_stats['exploration_progress']):.1%}")
        print(f"Average Robot1 path length: {np.mean(episode_stats['robot1_path_length']):.2f}")
        print(f"Average Robot2 path length: {np.mean(episode_stats['robot2_path_length']):.2f}")
        
        # 繪製並保存測試結果圖表
        plot_test_results(episode_stats)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理資源
        if plot:
            if hasattr(robot1, 'cleanup_visualization'):
                robot1.cleanup_visualization()
            if hasattr(robot2, 'cleanup_visualization'):
                robot2.cleanup_visualization()

# plot_test_results 函數保持不變...

def main():
    # 指定要使用的模型文件
    model_file = 'multi_robot_model_simple_ep000560.h5'
    model_path = os.path.join(MODEL_DIR, model_file)
    
    # 檢查文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        print(f"Available files in {MODEL_DIR}:")
        for file in os.listdir(MODEL_DIR):
            print(f"  - {file}")
        return
        
    print(f"\nUsing model: {model_file}")
    print(f"Model path: {model_path}")
    
    # 設定默認的測試輪數
    num_episodes = 5
    test_model(model_path, num_episodes=num_episodes, plot=True)

if __name__ == '__main__':
    main()
