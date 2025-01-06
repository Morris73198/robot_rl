import tensorflow as tf
import numpy as np

class MultiRobotNetworkModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        """初始化多機器人網路模型，保持原有接口"""
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    def _build_cnn_block(self, inputs, filters, kernel_size):
        """構建CNN區塊，包含殘差連接"""
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        # 殘差連接
        if inputs.shape[-1] == filters:
            x = tf.keras.layers.Add()([inputs, x])
        return tf.keras.layers.Activation('relu')(x)
    
    def _build_frontier_attention(self, frontiers, robot_pos, other_pos, other_target):
        """改進後的frontier注意力機制
        
        Args:
            frontiers: (batch, n_frontiers, 2) - frontier點坐標
            robot_pos: (batch, 2) - 當前機器人位置
            other_pos: (batch, 2) - 另一個機器人位置
            other_target: (batch, 2) - 另一個機器人的目標
        
        Returns:
            attention_weights: (batch, n_frontiers, 1) - 注意力權重
        """
        # 1. 計算與當前機器人的距離特徵
        rel_to_robot = frontiers - tf.expand_dims(robot_pos, axis=1)
        dist_to_robot = tf.norm(rel_to_robot, axis=-1, keepdims=True)  # 基礎距離特徵
        
        # 2. 計算與其他機器人相關的特徵
        rel_to_other = frontiers - tf.expand_dims(other_pos, axis=1)
        dist_to_other = tf.norm(rel_to_other, axis=-1, keepdims=True)  # 到其他機器人的距離
        
        # 3. 計算與其他機器人目標的關係
        rel_to_other_target = frontiers - tf.expand_dims(other_target, axis=1)
        dist_to_other_target = tf.norm(rel_to_other_target, axis=-1, keepdims=True)
        
        # 4. 計算相對方向（用於評估是否在同一區域）
        robot_to_other = tf.expand_dims(other_pos - robot_pos, axis=1)  # 機器人之間的向量
        robot_to_frontier = rel_to_robot  # frontier相對於當前機器人的向量
        
        # 標準化向量
        robot_to_other = tf.math.l2_normalize(robot_to_other, axis=-1)
        robot_to_frontier = tf.math.l2_normalize(robot_to_frontier, axis=-1)
        
        # 計算夾角餘弦
        angle_cos = tf.reduce_sum(robot_to_frontier * robot_to_other, axis=-1, keepdims=True)
        
        # 5. 組合特徵
        attention_features = tf.concat([
            1.0 / (1.0 + dist_to_robot),      # 距離機器人的近程指標
            1.0 / (1.0 + dist_to_other),      # 距離其他機器人的近程指標
            1.0 / (1.0 + dist_to_other_target),# 距離其他目標的近程指標
            angle_cos,                         # 方向特徵
        ], axis=-1)
        
        # 6. 注意力網絡
        attention = tf.keras.layers.Dense(32, activation='relu')(attention_features)
        attention = tf.keras.layers.Dense(1, activation='sigmoid')(attention)
        
        return attention
    
    def _build_model(self):
        """構建改進的網路模型，保持原有輸入輸出接口"""
        # 1. 輸入層
        map_input = tf.keras.layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = tf.keras.layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = tf.keras.layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = tf.keras.layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = tf.keras.layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = tf.keras.layers.Input(shape=(2,), name='robot2_target_input')
        
        # 2. 地圖特徵提取
        # 2.1 主幹網絡
        x = self._build_cnn_block(map_input, 32, 3)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = self._build_cnn_block(x, 64, 3)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = self._build_cnn_block(x, 128, 3)
        
        # 2.2 全局特徵
        map_features = tf.keras.layers.GlobalAveragePooling2D()(x)
        map_features = tf.keras.layers.Dense(256, activation='relu')(map_features)
        
        # 3. Frontier特徵處理
        frontier_features = tf.keras.layers.Dense(64, activation='relu')(frontier_input)
        
        # 4. 為兩個機器人構建注意力
        robot1_attention = self._build_frontier_attention(
            frontier_input, robot1_pos, robot2_pos, robot2_target)  # 只關注robot2的目標
    
        robot2_attention = self._build_frontier_attention(
            frontier_input, robot2_pos, robot1_pos, robot1_target)  # 只關注robot1的目標
        
        # 5. 應用注意力
        robot1_frontier_features = tf.keras.layers.Multiply()([frontier_features, robot1_attention])
        robot2_frontier_features = tf.keras.layers.Multiply()([frontier_features, robot2_attention])
        
        # 6. 特徵聚合
        robot1_frontier_context = tf.keras.layers.GlobalAveragePooling1D()(robot1_frontier_features)
        robot2_frontier_context = tf.keras.layers.GlobalAveragePooling1D()(robot2_frontier_features)
        
        # 7. 位置和目標編碼
        robot1_pos_encoding = tf.keras.layers.Dense(64, activation='relu')(robot1_pos)
        robot2_pos_encoding = tf.keras.layers.Dense(64, activation='relu')(robot2_pos)
        robot1_target_encoding = tf.keras.layers.Dense(64, activation='relu')(robot1_target)
        robot2_target_encoding = tf.keras.layers.Dense(64, activation='relu')(robot2_target)
        
        # 8. 特徵融合
        robot1_combined = tf.concat([
            map_features,
            robot1_frontier_context,
            robot1_pos_encoding,
            robot2_pos_encoding,
            robot1_target_encoding
        ], axis=-1)
        
        robot2_combined = tf.concat([
            map_features,
            robot2_frontier_context,
            robot2_pos_encoding,
            robot1_pos_encoding,
            robot2_target_encoding
        ], axis=-1)
        
        # 9. 決策網絡
        # Robot 1
        robot1_hidden = tf.keras.layers.Dense(512, activation='relu')(robot1_combined)
        robot1_hidden = tf.keras.layers.Dropout(0.2)(robot1_hidden)
        robot1_hidden = tf.keras.layers.Dense(256, activation='relu')(robot1_hidden)
        robot1_hidden = tf.keras.layers.Dropout(0.2)(robot1_hidden)
        robot1_output = tf.keras.layers.Dense(self.max_frontiers)(robot1_hidden)
        
        # Robot 2
        robot2_hidden = tf.keras.layers.Dense(512, activation='relu')(robot2_combined)
        robot2_hidden = tf.keras.layers.Dropout(0.2)(robot2_hidden)
        robot2_hidden = tf.keras.layers.Dense(256, activation='relu')(robot2_hidden)
        robot2_hidden = tf.keras.layers.Dropout(0.2)(robot2_hidden)
        robot2_output = tf.keras.layers.Dense(self.max_frontiers)(robot2_hidden)
        
        # 10. 構建模型
        model = tf.keras.Model(
            inputs={
                'map_input': map_input,
                'frontier_input': frontier_input,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            },
            outputs={
                'robot1': robot1_output,
                'robot2': robot2_output
            }
        )
        
        # 11. 編譯模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """更新目標網路"""
        self.target_model.set_weights(self.model.get_weights())
    
    def predict(self, state, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target):
        """預測動作值"""
        if len(state.shape) == 3:
            state = np.expand_dims(state, 0)
        if len(frontiers.shape) == 2:
            frontiers = np.expand_dims(frontiers, 0)
        if len(robot1_pos.shape) == 1:
            robot1_pos = np.expand_dims(robot1_pos, 0)
        if len(robot2_pos.shape) == 1:
            robot2_pos = np.expand_dims(robot2_pos, 0)
        if len(robot1_target.shape) == 1:
            robot1_target = np.expand_dims(robot1_target, 0)
        if len(robot2_target.shape) == 1:
            robot2_target = np.expand_dims(robot2_target, 0)
            
        return self.model.predict({
            'map_input': state,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        })
    
    def train_on_batch(self, states, frontiers, robot1_pos, robot2_pos, 
                      robot1_target, robot2_target,
                      robot1_targets, robot2_targets):
        """訓練一個批次"""
        return self.model.train_on_batch(
            {
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            },
            {
                'robot1': robot1_targets,
                'robot2': robot2_targets
            }
        )
    
    def save(self, path):
        """保存模型"""
        self.model.save(path)
    
    def load(self, path):
        """載入模型"""
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)