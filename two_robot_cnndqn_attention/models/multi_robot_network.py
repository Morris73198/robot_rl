import tensorflow as tf
import numpy as np

class MultiRobotNetworkModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        """初始化多機器人網路模型
        
        Args:
            input_shape: 輸入地圖的形狀，預設(84, 84, 1)
            max_frontiers: 最大frontier點數量，預設50
        """
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    def _build_cnn_block(self, inputs, filters, kernel_size):
        """構建CNN區塊,包含BN和殘差連接"""
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        if inputs.shape[-1] == filters:
            x = tf.keras.layers.Add()([inputs, x])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    def _build_model(self):
        """構建改進的網路模型"""
        # 1. 地圖輸入處理
        map_input = tf.keras.layers.Input(shape=self.input_shape, name='map_input')
        
        # 多尺度特徵提取
        x1 = self._build_cnn_block(map_input, 32, 3)
        x1 = tf.keras.layers.MaxPooling2D(2)(x1)
        
        x2 = self._build_cnn_block(map_input, 32, 5)
        x2 = tf.keras.layers.MaxPooling2D(2)(x2)
        
        x3 = self._build_cnn_block(map_input, 32, 7)
        x3 = tf.keras.layers.MaxPooling2D(2)(x3)
        
        # 合併多尺度特徵
        x = tf.keras.layers.Concatenate()([x1, x2, x3])
        
        x = self._build_cnn_block(x, 64, 3)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = self._build_cnn_block(x, 64, 3)
        map_features = tf.keras.layers.Flatten()(x)
        
        # 2. Frontier輸入處理
        frontier_input = tf.keras.layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        
        # 3. 機器人位置輸入
        robot1_pos_input = tf.keras.layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos_input = tf.keras.layers.Input(shape=(2,), name='robot2_pos_input')
        
        # 4. 新增: 目標位置輸入
        robot1_target_input = tf.keras.layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target_input = tf.keras.layers.Input(shape=(2,), name='robot2_target_input')
        
        # 計算相對位置特徵
        relative_pos = tf.keras.layers.Subtract()([robot1_pos_input, robot2_pos_input])
        relative_targets = tf.keras.layers.Subtract()([robot1_target_input, robot2_target_input])
        
        # 4. Frontier注意力機制
        frontier_features = tf.keras.layers.Dense(64, activation='relu')(frontier_input)
        
        # 計算每個frontier相對於機器人和目標的位置
        robot1_pos_expanded = tf.keras.layers.RepeatVector(self.max_frontiers)(robot1_pos_input)
        robot2_pos_expanded = tf.keras.layers.RepeatVector(self.max_frontiers)(robot2_pos_input)
        robot1_target_expanded = tf.keras.layers.RepeatVector(self.max_frontiers)(robot1_target_input)
        robot2_target_expanded = tf.keras.layers.RepeatVector(self.max_frontiers)(robot2_target_input)
        
        # 相對位置特徵
        rel_to_robot1 = tf.keras.layers.Subtract()([frontier_input, robot1_pos_expanded])
        rel_to_robot2 = tf.keras.layers.Subtract()([frontier_input, robot2_pos_expanded])
        rel_to_target1 = tf.keras.layers.Subtract()([frontier_input, robot1_target_expanded])
        rel_to_target2 = tf.keras.layers.Subtract()([frontier_input, robot2_target_expanded])
        
        # 計算注意力權重
        attention_features = tf.keras.layers.Concatenate()([
            frontier_features,
            rel_to_robot1,
            rel_to_robot2,
            rel_to_target1,
            rel_to_target2
        ])
        
        attention_dense = tf.keras.layers.Dense(128, activation='relu')(attention_features)
        attention_scores = tf.keras.layers.Dense(1)(attention_dense)
        attention_weights = tf.keras.layers.Softmax(axis=1)(attention_scores)
        
        # 加權frontier特徵
        weighted_frontiers = tf.keras.layers.Multiply()([frontier_features, attention_weights])
        frontier_context = tf.keras.layers.GlobalAveragePooling1D()(weighted_frontiers)
        
        # 5. 特徵融合
        robot1_features = tf.keras.layers.Dense(64, activation='relu')(robot1_pos_input)
        robot2_features = tf.keras.layers.Dense(64, activation='relu')(robot2_pos_input)
        robot1_target_features = tf.keras.layers.Dense(64, activation='relu')(robot1_target_input)
        robot2_target_features = tf.keras.layers.Dense(64, activation='relu')(robot2_target_input)
        relative_features = tf.keras.layers.Dense(64, activation='relu')(relative_pos)
        relative_target_features = tf.keras.layers.Dense(64, activation='relu')(relative_targets)
        
        # 合併所有特徵
        combined = tf.keras.layers.Concatenate()([
            map_features,
            frontier_context,
            robot1_features,
            robot2_features,
            robot1_target_features,
            robot2_target_features,
            relative_features,
            relative_target_features
        ])
        
        # 6. 深度特徵處理
        shared = tf.keras.layers.Dense(512, activation='relu')(combined)
        shared = tf.keras.layers.Dropout(0.2)(shared)
        shared = tf.keras.layers.Dense(256, activation='relu')(shared)
        shared = tf.keras.layers.Dropout(0.2)(shared)
        
        # 7. DQN輸出
        robot1_output = tf.keras.layers.Dense(self.max_frontiers)(shared)
        robot2_output = tf.keras.layers.Dense(self.max_frontiers)(shared)
        
        # 8. 構建最終模型
        model = tf.keras.Model(
            inputs={
                'map_input': map_input,
                'frontier_input': frontier_input,
                'robot1_pos_input': robot1_pos_input,
                'robot2_pos_input': robot2_pos_input,
                'robot1_target_input': robot1_target_input,
                'robot2_target_input': robot2_target_input
            },
            outputs={
                'robot1': robot1_output,
                'robot2': robot2_output
            }
        )
        
        # 9. 編譯模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse'
        )
        
        return model
    
    def update_target_model(self):
        """更新目標網路的權重"""
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