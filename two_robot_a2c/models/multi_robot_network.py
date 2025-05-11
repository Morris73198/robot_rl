import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers, optimizers
import json

# 自定義層正規化層
class LayerNormalization(layers.Layer):
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
        
    def get_config(self):
        return super().get_config()


# 添加新的梯度流解決層
class ResidualConnection(layers.Layer):
    """具有跳躍連接的殘差模塊，有助於解決梯度流問題"""
    def __init__(self, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dropout = layers.Dropout(dropout_rate)
        self.norm = LayerNormalization()
        
    def build(self, input_shape):
        self.projection = layers.Dense(input_shape[-1], use_bias=False)
        super().build(input_shape)
        
    def call(self, inputs, sublayer_output, training=None):
        # 跳躍連接
        # 子層輸出 + 原始輸入
        output = self.projection(sublayer_output) + inputs
        output = self.dropout(output, training=training)
        output = self.norm(output)
        return output
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'dropout_rate': self.dropout_rate
        })
        return config


class MultiRobotA2CModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        """初始化多機器人A2C模型
        
        Args:
            input_shape: 輸入地圖的形狀，默認(84, 84, 1)
            max_frontiers: 最大frontier點數量，默認50
        """
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.d_model = 256  # 模型維度
        self.num_heads = 8  # 注意力頭數
        self.dff = 512  # 前饋網絡維度
        self.dropout_rate = 0.1
        self.entropy_beta = 0.01  # 熵正則化係數
        self.value_loss_weight = 0.5  # 價值損失權重
        
        # 構建分離的模型
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
    def pad_frontiers(self, frontiers):
        """填充frontier點到固定長度並標準化座標"""
        padded = np.zeros((self.max_frontiers, 2))
        
        if len(frontiers) > 0:
            frontiers = np.array(frontiers)
            
            map_width = self.input_shape[1]
            map_height = self.input_shape[0]
            
            normalized_frontiers = frontiers.copy()
            normalized_frontiers[:, 0] = frontiers[:, 0] / float(map_width)
            normalized_frontiers[:, 1] = frontiers[:, 1] / float(map_height)
            
            n_frontiers = min(len(frontiers), self.max_frontiers)
            padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
        
        return padded
        
    def _build_perception_module(self, inputs):
        """改進的感知模塊，增強梯度流"""
        conv_configs = [
            {'filters': 32, 'kernel_size': 3, 'strides': 1},
            {'filters': 32, 'kernel_size': 5, 'strides': 1},
            {'filters': 32, 'kernel_size': 7, 'strides': 1}
        ]
        
        features = []
        for config in conv_configs:
            # 主路徑
            x = layers.Conv2D(
                filters=config['filters'],
                kernel_size=config['kernel_size'],
                strides=config['strides'],
                padding='same',
                kernel_initializer=tf.keras.initializers.HeNormal(),  # He初始化，適合ReLU
                kernel_regularizer=regularizers.l2(0.001)  # 減小正則化強度
            )(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # 空間注意力
            x = SpatialAttention()(x)
            
            # 第二個卷積
            x = layers.Conv2D(
                filters=config['filters'],
                kernel_size=config['kernel_size'],
                strides=config['strides'],
                padding='same',
                kernel_initializer=tf.keras.initializers.HeNormal(),
                kernel_regularizer=regularizers.l2(0.001)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            features.append(x)
            
        # 結合多尺度特徵
        concat_features = layers.Concatenate()(features)
        
        # 1x1 卷積整合通道
        x = layers.Conv2D(64, 1, kernel_initializer=tf.keras.initializers.HeNormal())(concat_features)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # 池化減少維度
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        return x

    def _build_coordination_module(self, robot1_state, robot2_state):
        """構建協調模塊，改進梯度流"""
        # 擴展維度以支持注意力
        robot1_expanded = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1)
        )(robot1_state)
        robot2_expanded = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1)
        )(robot2_state)
        
        # 合併狀態
        combined_states = layers.Concatenate(axis=1)([
            robot1_expanded, robot2_expanded
        ])
        
        # 保存原始輸入，用於殘差連接
        original_states = combined_states
        
        # 多頭注意力
        attention_output = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )([combined_states, combined_states, combined_states])
        
        # 殘差連接和層正規化 - 手動實現
        attention_output = layers.Add()([attention_output, original_states])
        attention_output = LayerNormalization()(attention_output)
        
        # 前饋網絡
        ffn_output = FeedForward(
            d_model=self.d_model,
            dff=self.dff,
            dropout_rate=self.dropout_rate
        )(attention_output)
        
        # 提取每個機器人的協調狀態
        robot1_coord = layers.Lambda(lambda x: x[:, 0, :])(ffn_output)
        robot2_coord = layers.Lambda(lambda x: x[:, 1, :])(ffn_output)
        
        return robot1_coord, robot2_coord
        
    def _build_frontier_module(self, frontier_input, robot_state):
        """構建改進的frontier評估模塊"""
        # 初始編碼
        x = layers.Dense(64, activation='relu')(frontier_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 位置編碼
        x = PositionalEncoding(self.max_frontiers, 64)(x)
        
        # 自注意力處理frontier
        # 保存原始輸入用於殘差連接
        original_x = x
        
        # 多頭注意力
        attention = MultiHeadAttention(
            d_model=64,
            num_heads=4,
            dropout_rate=self.dropout_rate
        )([x, x, x])
        
        # 殘差連接
        x = layers.Add()([attention, original_x])
        x = LayerNormalization()(x)
        
        # 結合機器人狀態
        robot_state_expanded = layers.RepeatVector(self.max_frontiers)(robot_state)
        
        # 使用深度可分離卷積進行特徵融合 - 提供更好的參數效率
        combined_features = layers.Concatenate()([x, robot_state_expanded])
        
        # 雙向LSTM處理序列關係
        lstm_out = layers.Bidirectional(layers.LSTM(
            32, 
            return_sequences=True,
            recurrent_initializer='glorot_uniform',  # 使用Glorot初始化，適合LSTM
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_dropout=0.0  # 移除循環丟棄以提高穩定性
        ))(combined_features)
        
        # 殘差連接
        x = layers.Concatenate()([lstm_out, combined_features])
        
        return x

    def _build_actor_network(self, features, name_prefix):
        """改進的Actor網絡，更好的梯度流"""
        # 保存初始特徵以用於殘差連接
        initial_features = features
        
        # 第一層 - 使用He初始化適合ReLU激活函數
        x = layers.Dense(
            256, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name=f"{name_prefix}_actor_dense1"
        )(features)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 第二層
        x = layers.Dense(
            128, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name=f"{name_prefix}_actor_dense2"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 殘差連接 - 將初始特徵投影到相同維度
        residual = layers.Dense(
            128, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            use_bias=False
        )(initial_features)
        residual = layers.BatchNormalization()(residual)
        
        # 合併主路徑和殘差
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        # 策略邏輯 - 使用小的初始權重避免大梯度
        logits = layers.Dense(
            self.max_frontiers,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros(),
            name=f"{name_prefix}_policy_logits"
        )(x)
        
        # 使用溫度參數的Softmax，以控制輸出分布的銳度
        temperature = 1.0  # 可調整，值越小輸出分布越尖銳
        policy = layers.Lambda(
            lambda x: tf.nn.softmax(x / temperature),
            name=f"{name_prefix}_policy"
        )(logits)
        
        return policy, logits
    
    def _build_critic_network(self, features, name_prefix):
        """改進的Critic網絡，更好的梯度流"""
        # 保存初始特徵以用於殘差連接
        initial_features = features
        
        # 第一層
        x = layers.Dense(
            256, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name=f"{name_prefix}_critic_dense1"
        )(features)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 第二層
        x = layers.Dense(
            128, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name=f"{name_prefix}_critic_dense2"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 殘差連接 - 將初始特徵投影到相同維度
        residual = layers.Dense(
            128, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            use_bias=False
        )(initial_features)
        residual = layers.BatchNormalization()(residual)
        
        # 合併主路徑和殘差
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        
        # 第三層
        x = layers.Dense(
            64, 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name=f"{name_prefix}_critic_dense3"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 最終值輸出 - 使用很小的初始權重避免大梯度
        value = layers.Dense(
            1, 
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros(),
            name=f"{name_prefix}_value"
        )(x)
        
        return value
    
    def _build_actor(self):
        """構建 Actor 網絡"""
        # 輸入層
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(
            shape=(self.max_frontiers, 2),
            name='frontier_input'
        )
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 1. 地圖感知處理
        map_features = self._build_perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features)
        
        # 2. 機器人狀態編碼
        robot1_state = layers.Concatenate()([robot1_pos, robot1_target])
        robot2_state = layers.Concatenate()([robot2_pos, robot2_target])
        
        robot1_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(0.001)
        )(robot1_state)
        robot1_features = layers.BatchNormalization()(robot1_features)
        
        robot2_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(0.001)
        )(robot2_state)
        robot2_features = layers.BatchNormalization()(robot2_features)
        
        # 3. 協調模塊
        robot1_coord, robot2_coord = self._build_coordination_module(
            robot1_features, robot2_features
        )
        
        # 4. Frontier評估
        robot1_frontier = self._build_frontier_module(frontier_input, robot1_coord)
        robot2_frontier = self._build_frontier_module(frontier_input, robot2_coord)
        
        # 5. Actor 輸出
        # Robot 1的特徵
        robot1_features = layers.Concatenate()([
            layers.Flatten()(robot1_frontier),
            robot1_coord,
            map_features_flat
        ])
        
        # Robot 2的特徵
        robot2_features = layers.Concatenate()([
            layers.Flatten()(robot2_frontier),
            robot2_coord,
            map_features_flat
        ])
        
        # 構建Actor網絡
        robot1_policy, robot1_logits = self._build_actor_network(robot1_features, "robot1")
        robot2_policy, robot2_logits = self._build_actor_network(robot2_features, "robot2")
        
        # 構建Actor模型
        actor_model = models.Model(
            inputs={
                'map_input': map_input,
                'frontier_input': frontier_input,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            },
            outputs={
                'robot1_policy': robot1_policy,
                'robot2_policy': robot2_policy,
                'robot1_logits': robot1_logits,
                'robot2_logits': robot2_logits
            }
        )
        
        # 使用改進的優化器設置
        actor_model.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,  # 較小的學習率
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=True  # 使用AMSGrad變體
        )
        
        return actor_model
    
    def _build_critic(self):
        """構建 Critic 網絡"""
        # 輸入層
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(
            shape=(self.max_frontiers, 2),
            name='frontier_input'
        )
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 1. 地圖感知處理
        map_features = self._build_perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features)
        
        # 2. 機器人狀態編碼
        robot1_state = layers.Concatenate()([robot1_pos, robot1_target])
        robot2_state = layers.Concatenate()([robot2_pos, robot2_target])
        
        robot1_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(0.001)
        )(robot1_state)
        robot1_features = layers.BatchNormalization()(robot1_features)
        
        robot2_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=regularizers.l2(0.001)
        )(robot2_state)
        robot2_features = layers.BatchNormalization()(robot2_features)
        
        # 3. 協調模塊
        robot1_coord, robot2_coord = self._build_coordination_module(
            robot1_features, robot2_features
        )
        
        # 4. Frontier評估
        robot1_frontier = self._build_frontier_module(frontier_input, robot1_coord)
        robot2_frontier = self._build_frontier_module(frontier_input, robot2_coord)
        
        # 5. Critic 輸出
        # Robot 1的特徵
        robot1_features = layers.Concatenate()([
            layers.Flatten()(robot1_frontier),
            robot1_coord,
            map_features_flat
        ])
        
        # Robot 2的特徵
        robot2_features = layers.Concatenate()([
            layers.Flatten()(robot2_frontier),
            robot2_coord,
            map_features_flat
        ])
        
        # 構建Critic網絡
        robot1_value = self._build_critic_network(robot1_features, "robot1")
        robot2_value = self._build_critic_network(robot2_features, "robot2")
        
        # 構建Critic模型
        critic_model = models.Model(
            inputs={
                'map_input': map_input,
                'frontier_input': frontier_input,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            },
            outputs={
                'robot1_value': robot1_value,
                'robot2_value': robot2_value
            }
        )
        
        # 優化器設置 - 對Critic使用較小的學習率
        critic_model.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=True
        )
        critic_model.compile(optimizer=critic_model.optimizer, loss='mse')
        
        return critic_model

    def train_actor(self, states, frontiers, robot1_pos, robot2_pos, 
                robot1_target, robot2_target, 
                robot1_actions, robot2_actions, 
                robot1_advantages, robot2_advantages,
                robot1_old_logits=None, robot2_old_logits=None,
                training_history=None, episode=0):
        """強化熵正則化的Actor訓練，促進探索但不使用ε-greedy"""
        with tf.GradientTape() as tape:
            # 獲取策略預測
            policy_dict = self.actor({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            # 使用epsilon平滑處理策略，避免極端值
            epsilon = 1e-8
            robot1_policy_smoothed = (1 - epsilon) * policy_dict['robot1_policy'] + epsilon / self.max_frontiers
            robot2_policy_smoothed = (1 - epsilon) * policy_dict['robot2_policy'] + epsilon / self.max_frontiers
            
            # 計算對數概率
            actions_one_hot_1 = tf.one_hot(robot1_actions, self.max_frontiers)
            actions_one_hot_2 = tf.one_hot(robot2_actions, self.max_frontiers)
            
            # 使用安全的對數計算
            log_prob_1 = tf.math.log(tf.reduce_sum(robot1_policy_smoothed * actions_one_hot_1, axis=1) + 1e-10)
            log_prob_2 = tf.math.log(tf.reduce_sum(robot2_policy_smoothed * actions_one_hot_2, axis=1) + 1e-10)
            
            # 計算策略損失
            policy_loss_coef = 1.0  # 策略損失係數
            
            # 基本的REINFORCE損失
            robot1_loss = -tf.reduce_mean(log_prob_1 * robot1_advantages) * policy_loss_coef
            robot2_loss = -tf.reduce_mean(log_prob_2 * robot2_advantages) * policy_loss_coef
            
            # 強化熵正則化 - 使用固定係數或從參數獲取
            entropy_coef = self.get_entropy_coefficient(training_history, episode)
            entropy_1 = -tf.reduce_mean(tf.reduce_sum(
                robot1_policy_smoothed * tf.math.log(robot1_policy_smoothed + 1e-10), 
                axis=1
            ))
            entropy_2 = -tf.reduce_mean(tf.reduce_sum(
                robot2_policy_smoothed * tf.math.log(robot2_policy_smoothed + 1e-10), 
                axis=1
            ))
            
            # 熵獎勵（熵越大，總損失越小，促進探索）
            entropy_reward = entropy_coef * (entropy_1 + entropy_2)
            
            # PPO風格的比率限制
            if robot1_old_logits is not None and robot2_old_logits is not None:
                # 計算舊策略的對數概率
                robot1_old_logits = tf.convert_to_tensor(robot1_old_logits, dtype=tf.float32)
                robot2_old_logits = tf.convert_to_tensor(robot2_old_logits, dtype=tf.float32)
                
                robot1_old_policy = tf.nn.softmax(robot1_old_logits, axis=-1)
                robot2_old_policy = tf.nn.softmax(robot2_old_logits, axis=-1)
                
                # 平滑處理舊策略
                robot1_old_policy = (1 - epsilon) * robot1_old_policy + epsilon / self.max_frontiers
                robot2_old_policy = (1 - epsilon) * robot2_old_policy + epsilon / self.max_frontiers
                
                # 計算舊策略下的動作概率
                old_log_prob_1 = tf.math.log(tf.reduce_sum(robot1_old_policy * actions_one_hot_1, axis=1) + 1e-10)
                old_log_prob_2 = tf.math.log(tf.reduce_sum(robot2_old_policy * actions_one_hot_2, axis=1) + 1e-10)
                
                # 計算概率比率
                ratio_1 = tf.exp(log_prob_1 - old_log_prob_1)
                ratio_2 = tf.exp(log_prob_2 - old_log_prob_2)
                
                # 裁剪比率
                clip_range = 0.2  # PPO裁剪範圍
                clipped_ratio_1 = tf.clip_by_value(ratio_1, 1 - clip_range, 1 + clip_range)
                clipped_ratio_2 = tf.clip_by_value(ratio_2, 1 - clip_range, 1 + clip_range)
                
                # 計算裁剪後的策略損失
                robot1_clipped_loss = -tf.reduce_mean(
                    tf.minimum(
                        ratio_1 * robot1_advantages,
                        clipped_ratio_1 * robot1_advantages
                    )
                ) * policy_loss_coef
                
                robot2_clipped_loss = -tf.reduce_mean(
                    tf.minimum(
                        ratio_2 * robot2_advantages,
                        clipped_ratio_2 * robot2_advantages
                    )
                ) * policy_loss_coef
                
                # 使用裁剪後的損失
                robot1_loss = robot1_clipped_loss
                robot2_loss = robot2_clipped_loss
            
            # 總損失 = 策略損失 - 熵獎勵
            policy_loss = robot1_loss + robot2_loss
            total_loss = policy_loss - entropy_reward
            
            # 增加額外的協調損失，鼓勵兩個機器人選擇不同的動作
            # 計算兩個策略分布的相似度
            similarity = tf.reduce_mean(
                tf.reduce_sum(
                    tf.sqrt(robot1_policy_smoothed * robot2_policy_smoothed), 
                    axis=1
                )
            )
            
            # 協調係數
            coordination_coef = 0.1
            coordination_loss = coordination_coef * similarity
            
            # 將協調損失加入總損失
            total_loss += coordination_loss
        
        # 計算梯度
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        
        # 梯度處理
        # 1. 替換 NaN 梯度為零
        grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else g for g in grads]
        
        # 2. 梯度裁剪，避免梯度爆炸
        clipped_grads, grad_norm = tf.clip_by_global_norm(grads, 1.0)
        
        # 應用梯度
        self.actor.optimizer.apply_gradients(zip(clipped_grads, self.actor.trainable_variables))
        
        # 返回指標
        metrics = {
            'total_policy_loss': total_loss,
            'robot1_policy_loss': robot1_loss,
            'robot2_policy_loss': robot2_loss,
            'robot1_entropy': entropy_1,
            'robot2_entropy': entropy_2,
            'entropy_coef': entropy_coef,
            'coordination_loss': coordination_loss,
            'gradient_norm': grad_norm
        }
        
        # 每隔一段時間打印一些訓練信息
        # if np.random.random() < 0.05:  # 5%的機率打印
        #     print(f"\n訓練指標:")
        #     print(f"Robot1熵: {entropy_1.numpy():.3f}, Robot2熵: {entropy_2.numpy():.3f}")
        #     print(f"熵係數: {entropy_coef:.4f}")
        #     print(f"協調損失: {coordination_loss.numpy():.4f}")
        #     print(f"梯度範數: {grad_norm.numpy():.4f}")
        
        return metrics

    def get_entropy_coefficient(self, training_history=None, episode=0):
        """動態調整熵係數以平衡探索和利用
        
        Args:
            training_history: 可選的訓練歷史字典
            episode: 當前訓練回合數
        """
        # 預設的基本熵係數
        base_coef = 0.01
        
        # 如果沒有提供訓練歷史或者回合數太少，使用較高的熵係數促進初期探索
        if training_history is None or episode < 20:
            return 0.05  # 初期使用較高值
        
        # 計算近期的熵值（如果有記錄）
        if ('robot1_entropy' in training_history and 
            len(training_history['robot1_entropy']) > 0):
            
            recent_entropy1 = np.mean(training_history['robot1_entropy'][-10:])
            recent_entropy2 = np.mean(training_history['robot2_entropy'][-10:])
            avg_entropy = (recent_entropy1 + recent_entropy2) / 2
            
            # 理論最大熵 (均勻分布的熵)
            max_entropy = np.log(self.max_frontiers)
            
            # 計算熵占最大熵的比例
            entropy_ratio = avg_entropy / max_entropy if max_entropy > 0 else 0
            
            # 根據熵比例動態調整係數
            if entropy_ratio < 0.3:
                # 熵太低，增加係數促進探索
                coef = base_coef * 2.0
            elif entropy_ratio > 0.7:
                # 熵太高，減少係數增加利用
                coef = base_coef * 0.5
            else:
                # 熵處於合理範圍
                coef = base_coef
                
        else:
            # 沒有熵歷史記錄，使用預設值
            coef = base_coef
        
        # 週期性地增加熵係數，避免陷入局部最優
        if episode % 100 == 0 and episode > 0:
            # 每100輪增加一次熵係數
            coef = coef * 1.5
            print(f"週期性增加熵係數至 {coef:.4f} 以促進探索")
        
        # 限制係數的範圍
        coef = np.clip(coef, 0.001, 0.1)
        
        return coef
    
    def train_critic(self, states, frontiers, robot1_pos, robot2_pos, 
                    robot1_target, robot2_target, 
                    robot1_returns, robot2_returns):
        """改進的Critic訓練，增強梯度處理"""
        with tf.GradientTape() as tape:
            # 獲取價值預測
            values = self.critic({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            # 計算機器人1的價值損失 - 使用Huber損失避免異常值影響
            robot1_value_loss = tf.keras.losses.Huber(delta=1.0)(
                robot1_returns, values['robot1_value'])
            
            # 計算機器人2的價值損失
            robot2_value_loss = tf.keras.losses.Huber(delta=1.0)(
                robot2_returns, values['robot2_value'])
            
            # 總價值損失
            value_loss = robot1_value_loss + robot2_value_loss
        
        # 計算梯度
        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        
        # 梯度處理
        # 1. 替換 NaN 梯度為零
        grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else g for g in grads]
        
        # 2. 裁剪異常梯度
        clipped_grads = []
        for g in grads:
            if g is not None:
                # 檢測異常大的梯度
                norm = tf.norm(g)
                
                # 如果範數過大，進行裁剪
                if norm > 1.0:
                    g = g * (1.0 / norm)
            clipped_grads.append(g)
        
        # 全局梯度裁剪 - 為Critic使用更嚴格的裁剪
        clipped_grads, _ = tf.clip_by_global_norm(clipped_grads, 0.5)
        
        # 應用梯度
        self.critic.optimizer.apply_gradients(zip(clipped_grads, self.critic.trainable_variables))
        
        return {
            'total_value_loss': value_loss,
            'robot1_value_loss': robot1_value_loss,
            'robot2_value_loss': robot2_value_loss
        }
    
    def train_batch(self, states, frontiers, robot1_pos, robot2_pos, 
                robot1_target, robot2_target,
                robot1_actions, robot2_actions, 
                robot1_advantages, robot2_advantages,
                robot1_returns, robot2_returns,
                robot1_old_values, robot2_old_values,
                robot1_old_logits, robot2_old_logits,
                training_history=None):  # Accept training_history as a parameter
        """改進的批次訓練過程"""
        # 訓練Actor
        current_episode = len(training_history['episode_rewards']) if training_history else 0
        
        actor_metrics = self.train_actor(
            states, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target,
            robot1_actions, robot2_actions, 
            robot1_advantages, robot2_advantages,
            robot1_old_logits, robot2_old_logits,
            training_history, current_episode  # Pass the training history
        )
        
        # 訓練Critic
        critic_metrics = self.train_critic(
            states, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target,
            robot1_returns, robot2_returns
        )
        
        # 計算熵
        # 為了穩定計算，我們使用模型預測而不是傳入的舊值
        policy_dict = self.actor.predict({
            'map_input': states,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        }, verbose=0)
        
        # 計算熵 - 使用epsilon平滑
        epsilon = 1e-8
        robot1_policy = policy_dict['robot1_policy']
        robot2_policy = policy_dict['robot2_policy']
        
        robot1_policy_smoothed = (1 - epsilon) * robot1_policy + epsilon / self.max_frontiers
        robot2_policy_smoothed = (1 - epsilon) * robot2_policy + epsilon / self.max_frontiers
        
        robot1_entropy = -tf.reduce_mean(tf.reduce_sum(
            robot1_policy_smoothed * tf.math.log(robot1_policy_smoothed + 1e-10), 
            axis=1
        ))
        robot2_entropy = -tf.reduce_mean(tf.reduce_sum(
            robot2_policy_smoothed * tf.math.log(robot2_policy_smoothed + 1e-10), 
            axis=1
        ))
        
        # 合併所有指標
        return {
            'total_loss': actor_metrics['total_policy_loss'] + critic_metrics['total_value_loss'],
            'robot1_policy_loss': actor_metrics['robot1_policy_loss'],
            'robot2_policy_loss': actor_metrics['robot2_policy_loss'],
            'robot1_value_loss': critic_metrics['robot1_value_loss'],
            'robot2_value_loss': critic_metrics['robot2_value_loss'],
            'robot1_entropy': robot1_entropy,
            'robot2_entropy': robot2_entropy,
            'gradient_norm': actor_metrics.get('gradient_norm', 0.0)  # 保持接口一致
        }
    
    def predict(self, state, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target):
        """預測動作分佈和狀態價值"""
        # 確保輸入形狀正確
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
        
        # 構建輸入字典
        inputs = {
            'map_input': state,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        }
        
        # 獲取Actor輸出
        actor_outputs = self.actor.predict(inputs, verbose=0)
        
        # 獲取Critic輸出
        critic_outputs = self.critic.predict(inputs, verbose=0)
        
        # 檢查策略輸出的有效性
        for key in ['robot1_policy', 'robot2_policy']:
            policy = actor_outputs[key]
            if np.any(np.isnan(policy)):
                print(f"警告: {key} 包含 NaN 值，替換為均勻分布")
                actor_outputs[key] = np.ones_like(policy) / policy.shape[-1]
            elif np.any(policy < 0) or np.any(policy > 1):
                print(f"警告: {key} 包含超出範圍的值，進行裁剪")
                actor_outputs[key] = np.clip(policy, 0, 1)
                # 重新歸一化
                actor_outputs[key] = actor_outputs[key] / np.sum(actor_outputs[key], axis=-1, keepdims=True)
                
        # 返回預測結果
        return {
            'robot1_policy': actor_outputs['robot1_policy'],
            'robot2_policy': actor_outputs['robot2_policy'],
            'robot1_value': critic_outputs['robot1_value'],
            'robot2_value': critic_outputs['robot2_value'],
            'robot1_logits': actor_outputs['robot1_logits'],
            'robot2_logits': actor_outputs['robot2_logits']
        }
    
    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        """改進的回報和優勢計算，增強數值穩定性"""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        
        # 保持合理的獎勵尺度
        reward_scale = 1.0
        scaled_rewards = rewards * reward_scale
        
        # GAE參數
        gamma = 0.99  # 折扣因子
        lambda_param = 0.95  # GAE平衡參數
        
        # 從後向前計算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_val = values[t+1]
            
            # 計算TD誤差
            delta = scaled_rewards[t] + gamma * next_val * next_non_terminal - values[t]
            
            # 計算GAE
            gae = delta + gamma * lambda_param * next_non_terminal * gae
            
            # 保存優勢和回報
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # 穩健的標準化
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        
        # 標準化優勢函數
        normalized_advantages = (advantages - adv_mean) / adv_std
        
        # 限制範圍，避免極端值
        max_value = 15.0  # 較大的範圍以保持信號強度
        normalized_advantages = np.clip(normalized_advantages, -max_value, max_value)
        
        # 周期性打印優勢函數統計信息，用於監控
        # if np.random.random() < 0.1:  # 10%的機率打印，減少日誌噪音
        #     print(f"優勢函數統計: 均值={np.mean(normalized_advantages):.2f}, "
        #           f"標準差={np.std(normalized_advantages):.2f}, "
        #           f"最小值={np.min(normalized_advantages):.2f}, "
        #           f"最大值={np.max(normalized_advantages):.2f}")
        
        return returns, normalized_advantages
    
    def save(self, filepath):
        """保存模型為 .h5 格式"""
        # 保存 Actor 模型
        actor_path = filepath + '_actor.h5'
        print(f"保存 actor 模型到: {actor_path}")
        self.actor.save(actor_path, save_format='h5')
        
        # 保存 Critic 模型
        critic_path = filepath + '_critic.h5'
        print(f"保存 critic 模型到: {critic_path}")
        self.critic.save(critic_path, save_format='h5')
        
        # 保存額外的配置信息
        config = {
            'input_shape': self.input_shape,
            'max_frontiers': self.max_frontiers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'entropy_beta': self.entropy_beta,
            'value_loss_weight': self.value_loss_weight
        }
        
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f)
            
        print(f"模型已儲存為 .h5 格式")
    
    def load(self, filepath):
        """載入 .h5 格式的模型"""
        # 創建自定義對象字典
        custom_objects = {
            'LayerNormalization': LayerNormalization,
            'MultiHeadAttention': MultiHeadAttention,
            'PositionalEncoding': PositionalEncoding,
            'FeedForward': FeedForward,
            'SpatialAttention': SpatialAttention,
            'ResidualConnection': ResidualConnection
        }
        
        try:
            # 載入配置
            with open(filepath + '_config.json', 'r') as f:
                config = json.load(f)
                
            # 更新模型配置
            self.input_shape = tuple(config['input_shape'])
            self.max_frontiers = config['max_frontiers']
            self.d_model = config['d_model']
            self.num_heads = config['num_heads']
            self.dff = config['dff']
            self.dropout_rate = config['dropout_rate']
            
            # 載入 actor 模型
            actor_path = filepath + '_actor.h5'
            print(f"載入 actor 模型: {actor_path}")
            self.actor = tf.keras.models.load_model(
                actor_path,
                custom_objects=custom_objects
            )
            
            # 載入 critic 模型
            critic_path = filepath + '_critic.h5'
            print(f"載入 critic 模型: {critic_path}")
            self.critic = tf.keras.models.load_model(
                critic_path,
                custom_objects=custom_objects
            )
            
            # 設置優化器
            if not hasattr(self.actor, 'optimizer') or self.actor.optimizer is None:
                self.actor.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=0.0001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-8,
                    amsgrad=True
                )
            
            if not hasattr(self.critic, 'optimizer') or self.critic.optimizer is None:
                self.critic.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=0.0001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-8,
                    amsgrad=True
                )
                self.critic.compile(optimizer=self.critic.optimizer, loss='mse')
            
            print("模型載入成功")
            return True
            
        except Exception as e:
            print(f"載入模型時出錯: {str(e)}")
            return False