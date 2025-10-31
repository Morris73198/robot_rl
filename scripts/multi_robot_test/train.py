import os
import sys
import numpy as np
import tensorflow as tf

# 添加路徑以便導入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_robot_test.models.multi_robot_network import MultiRobotNetworkModel
from multi_robot_test.models.multi_robot_trainer import MultiRobotTrainer
from multi_robot_test.environment.multi_robot_no_unknown import Robot
from multi_robot_test.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

def fine_tune_without_architecture_change(pretrained_path, num_robots, max_robots=10):
    """在不改變架構的情況下微調模型"""
    # 載入完全相同架構的模型
    model = MultiRobotNetworkModel(
        input_shape=MODEL_CONFIG['input_shape'],
        max_frontiers=MODEL_CONFIG['max_frontiers'],
        max_robots=max_robots
    )
    
    # 載入預訓練權重
    model.load(pretrained_path)
    
    # 只修改學習率和優化器參數
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,  # 使用較小的學習率
        decay_steps=1000,
        decay_rate=0.95
    )
    
    # 重新編譯模型，使用更適合的優化器設置
    loss_dict = {f'robot{i}': model._huber_loss for i in range(max_robots)}
    
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=0.5,  # 更小的梯度裁剪值，穩定訓練
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss=loss_dict
    )
    
    return model

def create_multi_robot_environment(num_robots=3, index_map=0, train=True, plot=True):
    """創建多機器人環境
    
    Args:
        num_robots: 機器人數量
        index_map: 地圖索引
        train: 是否為訓練模式
        plot: 是否顯示可視化
    
    Returns:
        list: 機器人實例列表
    """
    print(f"正在創建 {num_robots} 個機器人的環境...")
    
    try:
        robots = Robot.create_shared_robots(
            index_map=index_map,
            num_robots=num_robots,
            train=train,
            plot=plot
        )
        
        print(f"成功創建了 {len(robots)} 個機器人")
        for i, robot in enumerate(robots):
            print(f"機器人 {i}: 位置 {robot.robot_position}, 其他機器人數量: {len(robot.other_robots)}")
        
        return robots
        
    except ImportError as e:
        if "multi_robot_no_unknown" in str(e):
            print("錯誤: 無法導入 multi_robot_no_unknown 模組")
            print("請確保已將 environment/multi_robot_no_unknown.py 文件放在正確位置")
        raise e
        
    except Exception as e:
        print(f"創建機器人環境時出錯: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主訓練函數"""
    # 配置參數
    NUM_ROBOTS = 3  # 可以修改這個數字來改變機器人數量
    MAX_ROBOTS = 4  # 模型支援的最大機器人數量
    
    # 確保機器人數量合理
    if NUM_ROBOTS > MAX_ROBOTS:
        print(f"錯誤: 機器人數量 ({NUM_ROBOTS}) 超過模型支援的最大數量 ({MAX_ROBOTS})")
        return
    
    if NUM_ROBOTS < 1:
        print("錯誤: 機器人數量必須至少為 1")
        return
    
    print(f"開始多機器人訓練 - 機器人數量: {NUM_ROBOTS}")
    
    try:
        # 檢查是否有預訓練模型
        model_path = os.path.join(MODEL_DIR, 'best_multi_robot.h5')
        
        if os.path.exists(model_path):
            print(f"正在載入並微調模型: {model_path}")
            model = fine_tune_without_architecture_change(
                model_path, 
                NUM_ROBOTS, 
                MAX_ROBOTS
            )
        else:
            print(f"在 {model_path} 未找到模型檔案")
            print("將開始全新訓練...")
            
            print("正在創建模型...")
            model = MultiRobotNetworkModel(
                input_shape=MODEL_CONFIG['input_shape'],
                max_frontiers=MODEL_CONFIG['max_frontiers'],
                max_robots=MAX_ROBOTS
            )
        
        # 創建多機器人環境
        robots = create_multi_robot_environment(
            num_robots=NUM_ROBOTS,
            index_map=0, 
            train=True, 
            plot=False
        )
        
        if robots is None:
            print("無法創建機器人環境，訓練終止")
            return
        
        # 創建訓練器
        print("正在創建訓練器...")
        trainer = MultiRobotTrainer(
            model=model,
            robots=robots,
            memory_size=MODEL_CONFIG['memory_size'],
            batch_size=MODEL_CONFIG['batch_size'],
            gamma=MODEL_CONFIG['gamma']
        )
        
        # 根據是否有預訓練模型調整探索參數
        if os.path.exists(model_path):
            # 微調時使用較小的探索率
            trainer.epsilon = 0.35
            trainer.epsilon_min = 0.05
            trainer.epsilon_decay = 0.99995
            print(f"微調模式 - 當前 epsilon: {trainer.epsilon}")
        else:
            # 全新訓練時使用較大的探索率
            trainer.epsilon = 1.0
            trainer.epsilon_min = 0.075
            trainer.epsilon_decay = 0.9995
            print(f"全新訓練模式 - 當前 epsilon: {trainer.epsilon}")
        
        # 確保模型保存目錄存在
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        print("開始訓練...")
        print(f"訓練配置:")
        print(f"  - 機器人數量: {NUM_ROBOTS}")
        print(f"  - 最大支援機器人數: {MAX_ROBOTS}")
        print(f"  - 訓練回合數: {TRAIN_CONFIG['episodes']}")
        print(f"  - 批次大小: {MODEL_CONFIG['batch_size']}")
        print(f"  - 記憶體大小: {MODEL_CONFIG['memory_size']}")
        print(f"  - 目標網絡更新頻率: {TRAIN_CONFIG['target_update_freq']}")
        print(f"  - 保存頻率: {TRAIN_CONFIG['save_freq']}")
        print(f"  - Epsilon: {trainer.epsilon} -> {trainer.epsilon_min} (衰減率: {trainer.epsilon_decay})")
        
        # 開始訓練
        trainer.train(
            episodes=TRAIN_CONFIG['episodes'],
            target_update_freq=TRAIN_CONFIG['target_update_freq'],
            save_freq=TRAIN_CONFIG['save_freq']
        )
        
        # 訓練完成後生成統計報告
        print("\n" + "="*60)
        print("訓練完成統計報告")
        print("="*60)
        
        if trainer.training_history['episode_rewards']:
            total_episodes = len(trainer.training_history['episode_rewards'])
            avg_reward = np.mean(trainer.training_history['episode_rewards'])
            final_epsilon = trainer.epsilon
            
            print(f"總訓練回合數: {total_episodes}")
            print(f"平均總獎勵: {avg_reward:.2f}")
            print(f"最終Epsilon: {final_epsilon:.4f}")
            
            # 各機器人統計
            for i in range(NUM_ROBOTS):
                robot_rewards = trainer.training_history['robot_rewards'][f'robot{i}']
                if robot_rewards:
                    print(f"Robot{i} 平均獎勵: {np.mean(robot_rewards):.2f}")
            
            # 探索進度
            if trainer.training_history['exploration_progress']:
                final_exploration = trainer.training_history['exploration_progress'][-1]
                print(f"最終探索進度: {final_exploration:.1%}")
        
        print("="*60)
        
        # 訓練完成後繪製訓練進度
        print("正在生成訓練進度圖...")
        progress_plot_path = os.path.join(MODEL_DIR, f'training_progress_{NUM_ROBOTS}_robots.png')
        trainer.plot_training_progress(save_path=progress_plot_path)
        
        # 保存最終模型
        final_model_path = os.path.join(MODEL_DIR, f'final_multi_robot_{NUM_ROBOTS}_robots.h5')
        model.save(final_model_path)
        print(f"最終模型已保存至: {final_model_path}")
        
    except Exception as e:
        print(f"訓練過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

def test_model_scalability():
    """測試模型在不同機器人數量下的性能"""
    print("開始測試模型可擴展性...")
    
    robot_counts = [1, 2, 3, 4, 5]  # 測試不同的機器人數量
    
    for num_robots in robot_counts:
        print(f"\n測試 {num_robots} 個機器人...")
        try:
            # 創建模型
            model = MultiRobotNetworkModel(
                input_shape=MODEL_CONFIG['input_shape'],
                max_frontiers=MODEL_CONFIG['max_frontiers'],
                max_robots=10
            )
            
            # 創建機器人環境
            robots = create_multi_robot_environment(
                num_robots=num_robots,
                index_map=0,
                train=False,
                plot=False
            )
            
            if robots:
                print(f"✓ {num_robots} 個機器人環境創建成功")
                
                # 測試預測功能
                state = robots[0].begin()
                frontiers = robots[0].get_frontiers()[:MODEL_CONFIG['max_frontiers']]
                
                if len(frontiers) > 0:
                    robots_poses = [robot.get_normalized_position() for robot in robots]
                    robots_targets = [np.zeros(2) for _ in robots]
                    
                    predictions = model.predict(
                        state, frontiers, robots_poses, robots_targets, num_robots
                    )
                    
                    print(f"✓ 預測功能正常，輸出形狀:")
                    for i in range(num_robots):
                        print(f"   Robot{i}: {predictions[f'robot{i}'].shape}")
                
                # 清理資源
                for robot in robots:
                    if hasattr(robot, 'cleanup_visualization'):
                        robot.cleanup_visualization()
                        
            else:
                print(f"✗ {num_robots} 個機器人環境創建失敗")
                
        except Exception as e:
            print(f"✗ {num_robots} 個機器人測試失敗: {str(e)}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='多機器人DQN訓練')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='運行模式: train(訓練) 或 test(測試)')
    parser.add_argument('--num_robots', type=int, default=3,
                        help='機器人數量 (默認: 3)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 修改全局機器人數量
        NUM_ROBOTS = args.num_robots
        main()
    elif args.mode == 'test':
        test_model_scalability()
    
    print("程式執行完畢")
