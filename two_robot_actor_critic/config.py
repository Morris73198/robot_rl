import os

# 模型配置
MODEL_CONFIG = {
    'input_shape': (84, 84, 1),  # 輸入地圖大小
    'max_frontiers': 50,         # 最大前沿點數量
    'memory_size': 50000,        # 經驗回放緩衝區大小
    'batch_size': 64,            # 批次大小
    'gamma': 0.99,               # 折扣因子
    'd_model': 256,              # 模型維度
    'num_heads': 8,              # 注意力頭數
    'dff': 512,                  # 前饋網路維度
    'dropout_rate': 0.1,         # Dropout率
    'learning_rate': 3e-4,      # 調整學習率
    'gradient_clip': 0.5,       # 添加梯度裁剪
    'update_freq': 4,           # 每 4 步更新一次網絡
}

# 訓練配置
TRAIN_CONFIG = {
    'episodes': 1000000,              # 訓練輪數
    'save_freq': 10,                  # 保存頻率
    'actor_learning_rate': 0.0003,    # Actor學習率
    'critic_learning_rate': 0.001,    # Critic學習率
    'target_update_freq': 10,         # 目標網絡更新頻率
    'clip_epsilon': 0.2,              # PPO裁剪參數
    'c1': 1.0,                        # 值損失係數
    'c2': 0.01,                       # 熵係數
    'gae_lambda': 0.95,               # GAE係數
    'max_steps_per_episode':10000000000000000
}

# 機器人配置
ROBOT_CONFIG = {
    'movement_step': 3.0,              # 移動步長
    'finish_percent': 0.95,            # 完成閾值
    'sensor_range': 80,                # 傳感器範圍
    'robot_size': 2,                   # 機器人大小
    'local_size': 40,                  # 局部地圖大小
    'min_frontier_dist': 30,           # 最小前沿點距離
    'safety_distance': 5,              # 安全距離
    'path_simplification': 3,          # 路徑簡化參數
    'target_reach_threshold': 3,       # 目標到達閾值
    'plot_interval': 1,               # 繪圖間隔
    'diagonal_weight': 1.414           # 對角線移動權重
}

# 獎勵配置
REWARD_CONFIG = {
    'exploration_weight': 1.0,         # 探索獎勵權重
    'movement_penalty': -0.01,         # 移動懲罰
    'collision_penalty': -1.0,         # 碰撞懲罰
    'target_reward': 1.0,              # 到達目標獎勵
    'path_efficiency_weight': 0.5,     # 路徑效率權重
    'cooperation_weight': 0.3          # 協作獎勵權重
}

# 目錄配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 創建必要的目錄
for directory in [MODEL_DIR, LOG_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)