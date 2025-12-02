import os

# 路徑配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

# 模型配置
MODEL_CONFIG = {
    'input_shape': (84, 84, 1),      # 輸入影像的形狀
    'max_frontiers': 50,            # 最大前沿點數量，需與動作空間大小匹配
    'memory_size': 20000,            # 經驗回放緩衝區大小
    'batch_size': 32,                # 訓練批次大小
    'gamma': 0.98,                   # 獎勵折扣因子
    'epsilon_min': 0.01,              # 最小探索率
    'epsilon_decay': 0.995           # 探索率衰減係數
}

# 訓練配置
TRAIN_CONFIG = {
    'episodes': 1000000,             # 訓練總回合數
    'steps_per_episode': 5000,       # 每回合最大步數
    'target_update_freq': 10,        # 目標網路更新頻率
    'save_freq': 20,                 # 模型儲存頻率
}

# 機器人配置
ROBOT_CONFIG = {
    # 基礎配置
    'sensor_range': 50,              # 感測器範圍
    'robot_size': 2,                 # 機器人尺寸
    'local_size': 40,                # 局部地圖大小
    'finish_percent': 0.995,         # 探索完成閾值
    
    # 移動優化配置
    'movement_step': 2,              # 基本移動步長
    'max_frontier_skip': 3,          # 每次跳過的路徑點數
    'path_simplification': 0.5,      # 路徑簡化閾值
    'min_frontier_dist': 30,         # 最小前沿點選擇距離
    'target_reach_threshold': 10,     # 目標到達閾值
    
    # A*尋路優化
    'safety_distance': 10,           # 安全距離
    'diagonal_weight': 1.2,          # 對角線移動代價權重
    
    # 視覺化更新頻率
    'plot_interval': 2               # 繪圖更新間隔
}

# 獎勵函數配置
REWARD_CONFIG = {
    'exploration_weight': 100,       # 探索獎勵權重
    'movement_penalty': -0.5,   # 移動懲罰
    'collision_penalty': -0.0005,     # 碰撞懲罰
    # 'target_completion_reward': 5.0,  # 目標完成獎勵（已註釋）
    # 'completion_reward': 3000.0        # 探索完成獎勵
}