import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers, optimizers
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


class TemporalAttentionModule(layers.Layer):
    """時間注意力模塊 (從圖2)"""
    def __init__(self, d_model, memory_length=5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.memory_length = memory_length
        self.memory = None
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'memory_length': self.memory_length
        })
        return config
        
    def build(self, input_shape):
        # 初始化記憶張量
        self.memory = self.add_weight(
            shape=(1, self.memory_length, self.d_model),
            initializer="zeros",
            trainable=False,
            name="memory_tensor"
        )
        self.attention_weights = self.add_weight(
            shape=(self.d_model,),
            initializer="glorot_uniform",
            trainable=True,
            name="attention_weights"
        )
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # 擴展維度使輸入與記憶相容
        expanded_input = tf.expand_dims(inputs, axis=1)
        
        if training:
            # 更新記憶 (90% 舊 + 10% 新)
            updated_memory = tf.concat([
                self.memory[:, 1:, :],  # 移除最舊的記憶
                expanded_input[:1, :, :]  # 添加當前輸入的第一個批次
            ], axis=1)
            
            self.memory.assign(0.9 * self.memory + 0.1 * updated_memory)
        
        # 擴展記憶以匹配批次大小
        repeated_memory = tf.tile(self.memory, [batch_size, 1, 1])
        
        # 簡單注意力計算 (不使用MultiHeadAttention以避免形狀兼容性問題)
        # 計算擴展輸入和記憶之間的相似度
        similarity = tf.reduce_sum(
            expanded_input * self.attention_weights[tf.newaxis, tf.newaxis, :],
            axis=-1, keepdims=True
        )
        
        # 加權組合記憶和當前輸入
        attention_scores = tf.nn.softmax(similarity, axis=1)
        memory_context = tf.reduce_sum(repeated_memory * attention_scores, axis=1)
        
        # 結合記憶上下文和當前輸入
        temporal_features = inputs + 0.1 * memory_context
        
        return temporal_features


class AdaptiveAttentionFusion(layers.Layer):
    """自適應注意力融合 (從圖1)"""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model
        })
        return config
        
    def build(self, input_shape):
        self.weight_network = layers.Dense(3, activation='softmax')
        self.norm = LayerNormalization()
        super().build(input_shape)
        
    def call(self, inputs):
        frontier_features, cross_robot_features, temporal_features, map_features = inputs
        
        # 串聯所有特徵以計算權重
        concat_features = tf.concat([frontier_features, cross_robot_features, temporal_features], axis=-1)
        
        # 計算權重
        weights = self.weight_network(concat_features)
        
        # 分離權重
        w1 = tf.expand_dims(weights[:, 0], axis=-1)
        w2 = tf.expand_dims(weights[:, 1], axis=-1)
        w3 = tf.expand_dims(weights[:, 2], axis=-1)
        
        # 加權特徵
        weighted_features = w1 * frontier_features + w2 * cross_robot_features + w3 * temporal_features
        
        # 殘差連接和正規化
        fusion_features = self.norm(weighted_features + map_features)
        
        return fusion_features


class EnhancedMultiRobotA2CModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        """初始化增強版多機器人A2C模型 - 調整為與原始模型類似的參數量
        
        Args:
            input_shape: 輸入地圖的形狀，默認(84, 84, 1)
            max_frontiers: 最大frontier點數量，默認50
        """
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        # 增大模型維度以匹配原始模型
        self.d_model = 512  # 從128增加到512
        self.num_heads = 8  # 從4增加到8
        self.dff = 2048  # 從256增加到2048
        self.dropout_rate = 0.1
        self.entropy_beta = 0.05  # 熵正則化係數
        self.value_loss_weight = 0.5  # 價值損失權重
        
        # 構建模型
        self.model = self._build_model()
        
    def pad_frontiers(self, frontiers):
        """將frontier點填充到固定長度並標準化坐標"""
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
        """構建感知模塊 (圖3中的Perception部分)"""
        # 增加卷積核大小的CNN塊的過濾器數量
        conv_configs = [
            {'filters': 64, 'kernel_size': 3, 'strides': 1},
            {'filters': 64, 'kernel_size': 5, 'strides': 1},
            {'filters': 64, 'kernel_size': 7, 'strides': 1}
        ]
        
        features = []
        for config in conv_configs:
            x = layers.Conv2D(
                filters=config['filters'],
                kernel_size=config['kernel_size'],
                strides=config['strides'],
                padding='same',
                kernel_regularizer=regularizers.l2(0.01)
            )(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # 添加空間注意力
            x = SpatialAttention()(x)
            
            x = layers.Conv2D(
                filters=config['filters'],
                kernel_size=config['kernel_size'],
                strides=config['strides'],
                padding='same',
                kernel_regularizer=regularizers.l2(0.01)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            features.append(x)
            
        # 特徵融合
        concat_features = layers.Concatenate()(features)
        
        # 1x1卷積和池化
        x = layers.Conv2D(128, 1)(concat_features)  # 增加到128
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        return x

    def _build_cross_robot_attention(self, robot1_features, robot2_features):
        """構建跨機器人注意力模塊 - 增強版"""
        # 對機器人特徵進行更複雜的轉換
        r1_transformed = layers.Dense(self.d_model, activation='relu')(robot1_features)
        r1_transformed = layers.Dense(self.d_model, activation='relu')(r1_transformed)  # 額外層
        
        r2_transformed = layers.Dense(self.d_model, activation='relu')(robot2_features)
        r2_transformed = layers.Dense(self.d_model, activation='relu')(r2_transformed)  # 額外層
        
        # 多頭注意力風格的計算
        def attention_head(q, k, v, head_size):
            att_weights = tf.matmul(q, k, transpose_b=True)
            att_weights = att_weights / tf.sqrt(tf.cast(head_size, tf.float32))
            att_weights = tf.nn.softmax(att_weights, axis=-1)
            return tf.matmul(att_weights, v)
        
        # 分割為多頭
        head_size = self.d_model // self.num_heads
        
        # R1注意R2
        r1_heads = []
        for i in range(self.num_heads):
            q = layers.Dense(head_size)(r1_transformed)
            k = layers.Dense(head_size)(r2_transformed)
            v = layers.Dense(head_size)(r2_transformed)
            r1_heads.append(attention_head(q, k, v, head_size))
        
        r1_attended = layers.Concatenate()(r1_heads)
        r1_attended = layers.Dense(self.d_model)(r1_attended)
        
        # R2注意R1
        r2_heads = []
        for i in range(self.num_heads):
            q = layers.Dense(head_size)(r2_transformed)
            k = layers.Dense(head_size)(r1_transformed)
            v = layers.Dense(head_size)(r1_transformed)
            r2_heads.append(attention_head(q, k, v, head_size))
        
        r2_attended = layers.Concatenate()(r2_heads)
        r2_attended = layers.Dense(self.d_model)(r2_attended)
        
        # 合併注意力結果
        cross_features = layers.Concatenate()([r1_attended, r2_attended])
        cross_robot_features = layers.Dense(self.d_model, activation='relu')(cross_features)
        
        return cross_robot_features

    def _build_frontier_module(self, frontier_input, robot_state):
        """構建frontier評估模塊 - 更大版本"""
        # 初始特徵提取 - 增加維度
        frontier_features = layers.Dense(256, activation='relu')(frontier_input)
        frontier_features = layers.Dropout(self.dropout_rate)(frontier_features)
        frontier_features = layers.Dense(256, activation='relu')(frontier_features)  # 額外層
        frontier_features = layers.Dropout(self.dropout_rate)(frontier_features)
        
        # 為每個frontier點擴展機器人狀態
        # 首先將robot_state擴展到frontier維度
        robot_state_expanded = layers.RepeatVector(self.max_frontiers)(robot_state)
        
        # 將frontier特徵與機器人狀態連接在一起（在特徵維度上）
        combined_features = layers.Concatenate(axis=2)([
            frontier_features, robot_state_expanded
        ])
        
        # 使用更大的Dense層處理每個frontier點
        transformed_features = layers.TimeDistributed(
            layers.Dense(256, activation='relu')
        )(combined_features)
        transformed_features = layers.Dropout(self.dropout_rate)(transformed_features)
        
        # 使用兩層較大的LSTM
        sequence_features = layers.Bidirectional(layers.LSTM(
            128, return_sequences=True, recurrent_dropout=0.1
        ))(transformed_features)
        
        sequence_features = layers.Bidirectional(layers.LSTM(
            128, return_sequences=True, recurrent_dropout=0.1
        ))(sequence_features)
        
        # 全局平均池化以獲取固定大小輸出
        frontier_embedding = layers.GlobalAveragePooling1D()(sequence_features)
        frontier_embedding = layers.Dense(512, activation='relu')(frontier_embedding)
        frontier_embedding = layers.Dropout(self.dropout_rate)(frontier_embedding)
        frontier_embedding = layers.Dense(self.d_model, activation='relu')(frontier_embedding)
        
        return frontier_embedding

    def _build_actor_network(self, features, name_prefix):
        """構建Actor網絡 - 增大版本"""
        # 共享特徵層
        shared = layers.Dense(256, activation='relu', name=f"{name_prefix}_actor_dense1")(features)
        shared = layers.Dropout(self.dropout_rate)(shared)
        shared = layers.Dense(128, activation='relu', name=f"{name_prefix}_actor_dense2")(shared)
        shared = layers.Dropout(self.dropout_rate)(shared)
        shared = layers.Dense(64, activation='relu', name=f"{name_prefix}_actor_dense3")(shared)
        shared = layers.Dropout(self.dropout_rate)(shared)
        
        # 輸出策略（概率分佈）
        logits = layers.Dense(
            self.max_frontiers,
            name=f"{name_prefix}_policy_logits"
        )(shared)
        
        # 使用softmax輸出動作概率
        policy = layers.Softmax(name=f"{name_prefix}_policy")(logits)
        
        return policy, logits
    
    def _build_critic_network(self, features, name_prefix):
        """構建Critic網絡 - 增大版本"""
        # 共享特徵層
        shared = layers.Dense(256, activation='relu', name=f"{name_prefix}_critic_dense1")(features)
        shared = layers.Dropout(self.dropout_rate)(shared)
        shared = layers.Dense(128, activation='relu', name=f"{name_prefix}_critic_dense2")(shared)
        shared = layers.Dropout(self.dropout_rate)(shared)
        shared = layers.Dense(64, activation='relu', name=f"{name_prefix}_critic_dense3")(shared)
        shared = layers.Dropout(self.dropout_rate)(shared)
        
        # 輸出狀態價值
        value = layers.Dense(1, name=f"{name_prefix}_value")(shared)
        
        return value

    def _build_model(self):
        """構建完整的增強版A2C模型 - 增大版本"""
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
        map_features_conv = self._build_perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features_conv)
        
        # 添加更多處理層
        map_features = layers.Dense(1024, activation='relu')(map_features_flat)
        map_features = layers.Dropout(self.dropout_rate)(map_features)
        map_features = layers.Dense(512, activation='relu')(map_features)
        map_features = layers.Dropout(self.dropout_rate)(map_features)
        map_features = layers.Dense(self.d_model, activation='relu')(map_features)
        
        # 2. 機器人狀態編碼
        robot1_state = layers.Concatenate()([robot1_pos, robot1_target])
        robot2_state = layers.Concatenate()([robot2_pos, robot2_target])
        
        # 增加狀態編碼的深度
        robot1_features = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(robot1_state)
        robot1_features = layers.Dropout(self.dropout_rate)(robot1_features)
        robot1_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(robot1_features)
        
        robot2_features = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(robot2_state)
        robot2_features = layers.Dropout(self.dropout_rate)(robot2_features)
        robot2_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(robot2_features)
        
        # 3. 跨機器人注意力
        cross_robot_features = self._build_cross_robot_attention(robot1_features, robot2_features)
        
        # 4. Frontier評估
        # 將兩個機器人狀態合併為單一狀態進行frontier評估
        combined_robot_state = layers.Concatenate()([robot1_features, robot2_features])
        combined_robot_state = layers.Dense(self.d_model, activation='relu')(combined_robot_state)
        
        frontier_features = self._build_frontier_module(frontier_input, combined_robot_state)
        
        # 5. 時間注意力
        temporal_features = TemporalAttentionModule(self.d_model, memory_length=10)(frontier_features)
        
        # 6. 自適應注意力融合
        fusion_features = AdaptiveAttentionFusion(self.d_model)(
            [frontier_features, cross_robot_features, temporal_features, map_features]
        )
        
        # 7. 處理層 - 更深的處理網絡
        process_features = layers.Concatenate()([fusion_features, map_features])
        
        process_features = layers.Dense(1024, activation='relu')(process_features)
        process_features = layers.Dropout(self.dropout_rate)(process_features)
        process_features = layers.Dense(512, activation='relu')(process_features)
        process_features = layers.Dropout(self.dropout_rate)(process_features)
        process_features = layers.Dense(256, activation='relu')(process_features)
        process_features = layers.Dropout(self.dropout_rate)(process_features)
        
        # 8. 分離Robot 1和Robot 2的特徵
        robot1_fusion = layers.Dense(256, activation='relu')(process_features)
        robot1_fusion = layers.Dropout(self.dropout_rate)(robot1_fusion)
        robot1_fusion = layers.Dense(128, activation='relu')(robot1_fusion)
        
        robot2_fusion = layers.Dense(256, activation='relu')(process_features)
        robot2_fusion = layers.Dropout(self.dropout_rate)(robot2_fusion)
        robot2_fusion = layers.Dense(128, activation='relu')(robot2_fusion)
        
        # 9. Actor和Critic網絡
        robot1_policy, robot1_logits = self._build_actor_network(robot1_fusion, "robot1")
        robot1_value = self._build_critic_network(robot1_fusion, "robot1")
        
        robot2_policy, robot2_logits = self._build_actor_network(robot2_fusion, "robot2")
        robot2_value = self._build_critic_network(robot2_fusion, "robot2")
        
        # 構建完整模型
        model = models.Model(
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
                'robot1_value': robot1_value,
                'robot2_policy': robot2_policy,
                'robot2_value': robot2_value,
                'robot1_logits': robot1_logits,
                'robot2_logits': robot2_logits
            }
        )
        
        return model
        
    def _compute_loss(self, robot_name, actions, advantages, values, old_values, returns, old_logits, logits):
        """計算A2C損失函數，優化策略更新和多樣性探索"""
        # 確保數據類型一致
        actions = tf.cast(actions, tf.int32)
        advantages = tf.cast(advantages, tf.float32)
        values = tf.cast(values, tf.float32)
        old_values = tf.cast(old_values, tf.float32)
        returns = tf.cast(returns, tf.float32)
        old_logits = tf.cast(old_logits, tf.float32)
        logits = tf.cast(logits, tf.float32)
        
        # 將動作轉為one-hot編碼
        actions_one_hot = tf.one_hot(actions, self.max_frontiers)
        
        # 計算策略概率和對數概率，應用溫度參數增加策略變異性
        temperature = tf.constant(1.5, dtype=tf.float32)  # 提高溫度增加探索
        scaled_logits = logits / temperature
        
        probs = tf.nn.softmax(scaled_logits, axis=-1)
        # 數值穩定性處理，避免極小值
        log_probs = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
        selected_log_probs = tf.reduce_sum(log_probs * actions_one_hot, axis=-1)
        
        # 計算策略熵
        entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
        entropy_loss = tf.reduce_mean(entropy)
        
        # 增強策略梯度
        policy_scale = 5.0  # 增加策略梯度尺度
        actor_loss = -policy_scale * tf.reduce_mean(selected_log_probs * advantages)
        
        # 動態熵正則化，當策略熵低時增加熵係數
        entropy_threshold = 1.0  # 根據實際熵水平調整
        current_entropy = tf.reduce_mean(entropy)
        adaptive_entropy_coef = tf.where(
            current_entropy < entropy_threshold,
            0.05,  # 熵低時使用較高係數
            0.01   # 熵高時使用較低係數
        )
        
        # 添加策略正則化以防止過度專一化
        policy_loss = actor_loss - adaptive_entropy_coef * entropy_loss
        
        # 改進的價值損失，使用較大的 Huber 閾值以允許更大的誤差範圍
        delta = returns - values
        abs_delta = tf.abs(delta)
        huber_threshold = 2.0  # 增加閾值
        quadratic = tf.minimum(abs_delta, huber_threshold)
        linear = abs_delta - quadratic
        value_loss = tf.reduce_mean(0.5 * tf.square(quadratic) + linear)
        
        # 總損失，增加策略損失權重
        policy_weight = 1.0  # 提高策略損失權重
        value_weight = 0.7   # 保持價值損失權重不變
        total_loss = policy_weight * policy_loss + value_weight * value_loss
        
        return total_loss, policy_loss, value_loss, entropy_loss
    
    def _create_train_function(self):
        """創建訓練函數"""
        # 創建優化器
        initial_learning_rate = 0.0001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        @tf.function
        def train_step(states, frontiers, robot1_pos, robot2_pos, 
                      robot1_target, robot2_target,
                      robot1_actions, robot2_actions, 
                      robot1_advantages, robot2_advantages,
                      robot1_returns, robot2_returns,
                      robot1_old_values, robot2_old_values,
                      robot1_old_logits, robot2_old_logits):
            """執行單步訓練"""
            # 確保所有輸入都是正確的數據類型
            states = tf.cast(states, tf.float32)
            frontiers = tf.cast(frontiers, tf.float32)
            robot1_pos = tf.cast(robot1_pos, tf.float32)
            robot2_pos = tf.cast(robot2_pos, tf.float32)
            robot1_target = tf.cast(robot1_target, tf.float32)
            robot2_target = tf.cast(robot2_target, tf.float32)
            robot1_actions = tf.cast(robot1_actions, tf.int32)
            robot2_actions = tf.cast(robot2_actions, tf.int32)
            robot1_advantages = tf.cast(robot1_advantages, tf.float32)
            robot2_advantages = tf.cast(robot2_advantages, tf.float32)
            robot1_returns = tf.cast(robot1_returns, tf.float32)
            robot2_returns = tf.cast(robot2_returns, tf.float32)
            robot1_old_values = tf.cast(robot1_old_values, tf.float32)
            robot2_old_values = tf.cast(robot2_old_values, tf.float32)
            robot1_old_logits = tf.cast(robot1_old_logits, tf.float32)
            robot2_old_logits = tf.cast(robot2_old_logits, tf.float32)
            
            with tf.GradientTape() as tape:
                # 獲取模型預測
                outputs = self.model({
                    'map_input': states,
                    'frontier_input': frontiers,
                    'robot1_pos_input': robot1_pos,
                    'robot2_pos_input': robot2_pos,
                    'robot1_target_input': robot1_target,
                    'robot2_target_input': robot2_target
                }, training=True)
                
                # 提取輸出
                robot1_policy = outputs['robot1_policy']
                robot1_value = outputs['robot1_value'][:, 0]
                robot1_logits = outputs['robot1_logits']
                
                robot2_policy = outputs['robot2_policy']
                robot2_value = outputs['robot2_value'][:, 0]
                robot2_logits = outputs['robot2_logits']
                
                # 計算機器人1的損失
                robot1_loss, robot1_policy_loss, robot1_value_loss, robot1_entropy = self._compute_loss(
                    'robot1', robot1_actions, robot1_advantages, robot1_value, 
                    robot1_old_values, robot1_returns, robot1_old_logits, robot1_logits
                )
                
                # 計算機器人2的損失
                robot2_loss, robot2_policy_loss, robot2_value_loss, robot2_entropy = self._compute_loss(
                    'robot2', robot2_actions, robot2_advantages, robot2_value, 
                    robot2_old_values, robot2_returns, robot2_old_logits, robot2_logits
                )
                
                # 總損失
                total_loss = robot1_loss + robot2_loss
                
            # 計算梯度並應用
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
    
            # 梯度裁剪
            gradients, _ = tf.clip_by_global_norm(gradients, 1.0)  
            
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return {
                'total_loss': total_loss,
                'robot1_loss': robot1_loss,
                'robot1_policy_loss': robot1_policy_loss,
                'robot1_value_loss': robot1_value_loss,
                'robot1_entropy': robot1_entropy,
                'robot2_loss': robot2_loss,
                'robot2_policy_loss': robot2_policy_loss,
                'robot2_value_loss': robot2_value_loss,
                'robot2_entropy': robot2_entropy
            }
            
        self.train_step = train_step
    
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
            
        return self.model.predict(
            {
                'map_input': state,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            },
            verbose=0
        )
    
    def train_batch(self, states, frontiers, robot1_pos, robot2_pos, 
                   robot1_target, robot2_target,
                   robot1_actions, robot2_actions, 
                   robot1_advantages, robot2_advantages,
                   robot1_returns, robot2_returns,
                   robot1_old_values, robot2_old_values,
                   robot1_old_logits, robot2_old_logits):
        """訓練一個批次的數據"""
        if not hasattr(self, 'train_step'):
            self._create_train_function()
            
        return self.train_step(
            states, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target,
            robot1_actions, robot2_actions, 
            robot1_advantages, robot2_advantages,
            robot1_returns, robot2_returns,
            robot1_old_values, robot2_old_values,
            robot1_old_logits, robot2_old_logits
        )
    
    def save(self, filepath):
        """保存模型"""
        # 保存模型架構和權重
        self.model.save(filepath)
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
    
    def load(self, filepath):
        """載入模型"""
        # 創建自定義對象字典
        custom_objects = {
            'LayerNormalization': LayerNormalization,
            'SpatialAttention': SpatialAttention,
            'TemporalAttentionModule': TemporalAttentionModule,
            'AdaptiveAttentionFusion': AdaptiveAttentionFusion
        }
        
        # 載入模型
        self.model = models.load_model(
            filepath,
            custom_objects=custom_objects
        )
        
        # 重新創建訓練函數
        if hasattr(self, 'train_step'):
            delattr(self, 'train_step')
        self._create_train_function()