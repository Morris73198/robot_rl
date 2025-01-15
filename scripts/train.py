import os
import sys
import tensorflow as tf
from two_robot_actor_critic.models.multi_robot_actor_critic import MultiRobotActorCriticModel
from two_robot_actor_critic.models.multi_robot_actor_critic_trainer import MultiRobotActorCriticTrainer
from two_robot_actor_critic.environment.multi_robot import Robot
from two_robot_actor_critic.config import MODEL_CONFIG, TRAIN_CONFIG, MODEL_DIR

def main():
    try:
        # 設置GPU內存增長
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        
        # 創建模型目錄
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 創建 Actor-Critic 模型
        print("正在創建 Actor-Critic 模型...")
        model = MultiRobotActorCriticModel(
            input_shape=MODEL_CONFIG['input_shape'],
            max_frontiers=MODEL_CONFIG['max_frontiers']
        )
        
        # 編譯模型
        model.compile(
            actor_lr=TRAIN_CONFIG['actor_learning_rate'],
            critic_lr=TRAIN_CONFIG['critic_learning_rate']
        )
        
        # 如果指定了檢查點路徑，則載入模型
        checkpoint_path = os.path.join(MODEL_DIR, 'latest_checkpoint')
        if os.path.exists(checkpoint_path):
            print(f"正在載入檢查點: {checkpoint_path}")
            model.load(checkpoint_path)
            print("模型載入成功")
        else:
            print("將開始全新訓練...")
        
        # 創建共享環境的兩個機器人
        print("正在創建機器人環境...")
        robot1, robot2 = Robot.create_shared_robots(
            index_map=0,
            train=True,
            plot=True
        )
        
        # 創建訓練器
        print("正在創建訓練器...")
        trainer = MultiRobotActorCriticTrainer(
            model=model,
            robot1=robot1,
            robot2=robot2,
            memory_size=MODEL_CONFIG['memory_size'],
            batch_size=MODEL_CONFIG['batch_size'],
            gamma=MODEL_CONFIG['gamma']
        )
        
        print("開始訓練...")
        trainer.train(
            episodes=TRAIN_CONFIG['episodes'],
            save_freq=TRAIN_CONFIG['save_freq']
        )
        
    except Exception as e:
        print(f"訓練過程出現錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == '__main__':
    main()