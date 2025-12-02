import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers
import json

class LayerNormalization(layers.Layer):
    """自定義層正規化層"""
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True)
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        return self.gamma * (x - mean) / tf.sqrt(variance + self.epsilon) + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config

class MultiHeadAttention(layers.Layer):
    """多頭注意力層"""
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """計算注意力權重"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = tf.matmul(attention_weights, v)
        return output
        
    def call(self, inputs, mask=None, training=None):
        q, k, v = inputs
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output)
        
        return output

class PositionalEncoding(layers.Layer):
    """位置編碼層"""
    def __init__(self, max_position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(max_position, d_model)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_position': self.max_position,
            'd_model': self.d_model
        })
        return config
        
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
        
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

class FeedForward(layers.Layer):
    """前饋神經網路層"""
    def __init__(self, d_model, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.dense1 = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config
        
    def call(self, x, training=None):
        ffn_output = self.dense1(x)
        ffn_output = self.dropout(ffn_output, training=training)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout(ffn_output, training=training)
        ffn_output = self.layer_norm(x + ffn_output)
        return ffn_output

class SpatialAttention(layers.Layer):
    """空間注意力層"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(1, 7, padding='same', use_bias=False)
        self.norm = LayerNormalization()
        super().build(input_shape)
        
    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention_map = self.conv1(concat)
        attention_map = tf.sigmoid(attention_map)
        
        output = inputs * attention_map
        output = self.norm(output)
        return output

class MultiRobotNetworkModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50, max_robots=10):
        """初始化多機器人網路模型
        
        Args:
            input_shape: 輸入地圖的形狀，默認(84, 84, 1)
            max_frontiers: 最大frontier點數量，默認50
            max_robots: 支援的最大機器人數量，默認10
        """
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.max_robots = max_robots
        self.d_model = 256  # 模型維度
        self.num_heads = 8  # 注意力頭數
        self.dff = 512  # 前饋網路維度
        self.dropout_rate = 0.1
        
        # 構建並編譯模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        
        # 初始化目標網路權重
        self.update_target_model()
        
    def _build_perception_module(self, map_input):
        """構建感知模塊"""
        # CNN特徵提取
        x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(map_input)
        x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), strides=1, activation='relu')(x)
        
        # 應用空間注意力
        x = SpatialAttention()(x)
        
        return x
        
    def _build_robot_state_module(self, robots_poses, robots_targets):
        """構建機器人狀態編碼模塊
        
        Args:
            robots_poses: 所有機器人位置 [batch, max_robots, 2]
            robots_targets: 所有機器人目標 [batch, max_robots, 2]
        """
        # 將位置和目標連接
        robot_states = layers.Concatenate(axis=-1)([robots_poses, robots_targets])  # [batch, max_robots, 4]
        
        # 編碼每個機器人的狀態
        robot_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01),
            name='robot_state_encoder'
        )(robot_states)
        
        # 添加位置編碼
        robot_features = PositionalEncoding(self.max_robots, self.d_model)(robot_features)
        
        return robot_features
        
    def _build_multi_robot_coordination_module(self, robot_features):
        """構建多機器人協調模塊
        
        Args:
            robot_features: 機器人特徵 [batch, max_robots, d_model]
        """
        # 多頭自注意力機制 - 讓機器人之間可以相互感知和協調
        attention_output = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            name='robot_coordination_attention'
        )([robot_features, robot_features, robot_features])
        
        # 前饋神經網路
        coordinated_features = FeedForward(
            d_model=self.d_model,
            dff=self.dff,
            dropout_rate=self.dropout_rate,
            name='robot_coordination_ffn'
        )(attention_output)
        
        return coordinated_features
        
    def _build_frontier_evaluation_module(self, frontier_input, robot_features):
        """構建frontier評估模塊
        
        Args:
            frontier_input: frontier點座標 [batch, max_frontiers, 2]
            robot_features: 協調後的機器人特徵 [batch, max_robots, d_model]
        """
        # 編碼frontier特徵
        frontier_features = layers.Dense(64, activation='relu')(frontier_input)
        frontier_features = layers.Dropout(self.dropout_rate)(frontier_features)
        frontier_features = PositionalEncoding(self.max_frontiers, 64)(frontier_features)
        
        # 擴展機器人特徵以匹配frontier維度進行交互
        # robot_features: [batch, max_robots, d_model]
        # 我們需要為每個機器人和每個frontier計算交互特徵
        
        # 將機器人特徵投影到合適的維度
        robot_projected = layers.Dense(64, activation='relu')(robot_features)  # [batch, max_robots, 64]
        
        return frontier_features, robot_projected
        
    def _build_dueling_output_head(self, combined_features, robot_id):
        """為每個機器人構建Dueling DQN輸出頭
        
        Args:
            combined_features: 結合的特徵
            robot_id: 機器人ID用於命名
        """
        # 共享特徵層
        shared = layers.Dense(512, activation='relu')(combined_features)
        shared = layers.Dropout(self.dropout_rate)(shared)
        shared = layers.Dense(256, activation='relu')(shared)
        shared = layers.Dropout(self.dropout_rate)(shared)
        
        # 價值流 (Value Stream) - 估計狀態價值
        value_stream = layers.Dense(128, activation='relu')(shared)
        value = layers.Dense(1, name=f'robot{robot_id}_value')(value_stream)
        
        # 優勢流 (Advantage Stream) - 估計動作優勢
        advantage_stream = layers.Dense(128, activation='relu')(shared)
        advantage = layers.Dense(
            self.max_frontiers, 
            name=f'robot{robot_id}_advantage'
        )(advantage_stream)
        
        # Dueling架構：Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        mean_advantage = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=1, keepdims=True)
        )(advantage)
        
        q_values = layers.Add(name=f'robot{robot_id}')([
            value,
            layers.Subtract()([advantage, mean_advantage])
        ])
        
        return q_values

    def _build_model(self):
        """構建完整的多機器人Dueling DQN模型"""
        # 輸入層定義
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        
        # 動態機器人狀態輸入 - 使用單一輸入來接收所有機器人的pose和target
        robots_poses = layers.Input(shape=(self.max_robots, 2), name='robots_poses_input')
        robots_targets = layers.Input(shape=(self.max_robots, 2), name='robots_targets_input')
        
        # 地圖感知處理
        map_features = self._build_perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features)
        
        # 機器人狀態編碼和協調
        robot_features = self._build_robot_state_module(robots_poses, robots_targets)
        coordinated_robot_features = self._build_multi_robot_coordination_module(robot_features)
        
        # Frontier評估
        frontier_features, robot_projected = self._build_frontier_evaluation_module(
            frontier_input, coordinated_robot_features)
        
        # 為每個機器人構建輸出頭
        outputs = {}
        
        for robot_id in range(self.max_robots):
            # 提取該機器人的協調特徵
            robot_coord_features = layers.Lambda(
                lambda x, i=robot_id: x[:, i, :],
                name=f'robot{robot_id}_coord_extract'
            )(coordinated_robot_features)
            
            # 結合地圖特徵、機器人協調特徵和frontier特徵
            # 使用注意力機制來融合frontier和機器人特徵
            robot_frontier_attention = layers.MultiHeadAttention(
                num_heads=4, 
                key_dim=16,
                name=f'robot{robot_id}_frontier_attention'
            )(
                query=layers.RepeatVector(self.max_frontiers)(robot_coord_features),
                value=frontier_features,
                key=frontier_features
            )
            
            # 平均池化注意力輸出
            robot_frontier_features = layers.GlobalAveragePooling1D()(robot_frontier_attention)
            
            # 結合所有特徵
            combined_features = layers.Concatenate()([
                robot_coord_features,
                robot_frontier_features,
                map_features_flat
            ])
            
            # 生成該機器人的Q值
            robot_output = self._build_dueling_output_head(combined_features, robot_id)
            outputs[f'robot{robot_id}'] = robot_output
        
        # 創建模型
        model = models.Model(
            inputs=[map_input, frontier_input, robots_poses, robots_targets],
            outputs=outputs,
            name='MultiRobotDuelingDQN'
        )
        
        # 編譯模型
        model.compile(
            optimizer='adam',
            loss={f'robot{i}': self._huber_loss for i in range(self.max_robots)}
        )
        
        return model
    
    def pad_frontiers(self, frontiers):
        """填充frontier點到固定長度並進行標準化"""
        padded = np.zeros((self.max_frontiers, 2))
        
        if len(frontiers) > 0:
            frontiers = np.array(frontiers)
            n_frontiers = min(len(frontiers), self.max_frontiers)
            padded[:n_frontiers] = frontiers[:n_frontiers]
        
        return padded
    
    def pad_robot_states(self, robots_poses, robots_targets, num_active_robots):
        """填充機器人狀態到固定長度
        
        Args:
            robots_poses: 實際機器人位置列表
            robots_targets: 實際機器人目標列表
            num_active_robots: 實際活躍的機器人數量
        """
        padded_poses = np.zeros((self.max_robots, 2))
        padded_targets = np.zeros((self.max_robots, 2))
        
        if num_active_robots > 0:
            n_robots = min(num_active_robots, self.max_robots)
            padded_poses[:n_robots] = np.array(robots_poses)[:n_robots]
            padded_targets[:n_robots] = np.array(robots_targets)[:n_robots]
        
        return padded_poses, padded_targets

    def _huber_loss(self, y_true, y_pred):
        """Huber損失函數"""
        return tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)

    def update_target_model(self):
        """更新目標網路"""
        tau = 0.001  # 軟更新係數
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
        self.target_model.set_weights(target_weights)
    
    def predict(self, state, frontiers, robots_poses, robots_targets, num_active_robots):
        """預測動作值"""
        # 確保輸入形狀正確
        if len(state.shape) == 3:
            state = np.expand_dims(state, 0)
        if len(frontiers.shape) == 2:
            frontiers = np.expand_dims(frontiers, 0)
            
        # 填充機器人狀態
        padded_poses, padded_targets = self.pad_robot_states(
            robots_poses, robots_targets, num_active_robots)
        
        if len(padded_poses.shape) == 2:
            padded_poses = np.expand_dims(padded_poses, 0)
        if len(padded_targets.shape) == 2:
            padded_targets = np.expand_dims(padded_targets, 0)
            
        return self.model.predict(
            {
                'map_input': state,
                'frontier_input': frontiers,
                'robots_poses_input': padded_poses,
                'robots_targets_input': padded_targets
            },
            verbose=0
        )
    
    def train_on_batch(self, states, frontiers, robots_poses, robots_targets, 
                      robot_targets_dict, num_active_robots_list):
        """訓練一個批次"""
        # 確保robots_poses和robots_targets已經正確填充到max_robots維度
        history = self.model.train_on_batch(
            {
                'map_input': states,
                'frontier_input': frontiers,
                'robots_poses_input': robots_poses,
                'robots_targets_input': robots_targets
            },
            robot_targets_dict
        )
        return history
    
    def save(self, filepath):
        """保存模型"""
        # 保存模型架構和權重
        self.model.save(filepath)
        # 保存額外的配置信息
        config = {
            'input_shape': self.input_shape,
            'max_frontiers': self.max_frontiers,
            'max_robots': self.max_robots,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        }
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f)
    
    def load(self, filepath):
        """載入模型"""
        # 創建自定義對象字典
        custom_objects = {
            'LayerNormalization': LayerNormalization,
            'MultiHeadAttention': MultiHeadAttention,
            'PositionalEncoding': PositionalEncoding,
            'FeedForward': FeedForward,
            'SpatialAttention': SpatialAttention,
            '_huber_loss': self._huber_loss
        }
        
        # 載入模型
        self.model = models.load_model(
            filepath,
            custom_objects=custom_objects
        )
        self.target_model = models.load_model(
            filepath,
            custom_objects=custom_objects
        )