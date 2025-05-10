import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers, optimizers
import json

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


class ResidualConnection(layers.Layer):
    def __init__(self, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.dropout = layers.Dropout(dropout_rate)
        self.norm = LayerNormalization()
        
    def build(self, input_shape):
        self.projection = layers.Dense(input_shape[-1], use_bias=False)
        super().build(input_shape)
        
    def call(self, inputs, sublayer_output, training=None):
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


# New components based on the diagrams

class PerceptionModule(layers.Layer):
    """Perception module for processing map input with multi-kernel CNN and spatial attention"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        # CNN blocks with different kernel sizes as shown in Image 3
        self.cnn_block3 = tf.keras.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization()
        ])
        
        self.cnn_block5 = tf.keras.Sequential([
            layers.Conv2D(32, 5, padding='same', activation='relu'),
            layers.BatchNormalization()
        ])
        
        self.cnn_block7 = tf.keras.Sequential([
            layers.Conv2D(32, 7, padding='same', activation='relu'),
            layers.BatchNormalization()
        ])
        
        # Spatial attention for each branch
        self.spatial_attention3 = SpatialAttention()
        self.spatial_attention5 = SpatialAttention()
        self.spatial_attention7 = SpatialAttention()
        
        # Feature fusion and pooling
        self.feature_fusion = layers.Conv2D(64, 1, activation='relu')
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        # Apply CNN blocks
        x3 = self.cnn_block3(inputs)
        x5 = self.cnn_block5(inputs)
        x7 = self.cnn_block7(inputs)
        
        # Apply spatial attention
        x3 = self.spatial_attention3(x3)
        x5 = self.spatial_attention5(x5)
        x7 = self.spatial_attention7(x7)
        
        # Concatenate features from different kernel sizes
        x = tf.concat([x3, x5, x7], axis=-1)
        
        # Feature fusion with 1x1 convolution
        x = self.feature_fusion(x)
        
        # Max pooling as shown in the diagram
        x = self.max_pool(x)
        
        return x

class CrossRobotAttention(layers.Layer):
    """跨機器人注意力模組，如圖3所示"""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        # 獲取輸入形狀
        robot1_shape, robot2_shape = input_shape
        
        # 安全地處理可能為 None 的維度
        # 所有特徵都使用線性投影到 d_model 維度
        self.robot1_proj = layers.Dense(self.d_model)
        self.robot2_proj = layers.Dense(self.d_model)
        
        # R1 關注 R2
        self.r1_attention = layers.Dense(self.d_model)
        self.r1_norm = LayerNormalization()
        
        # R2 關注 R1
        self.r2_attention = layers.Dense(self.d_model)
        self.r2_norm = LayerNormalization()
        
        # 交叉特徵融合層
        self.fusion_layer = layers.Dense(self.d_model * 2)
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        robot1_features, robot2_features = inputs
        
        # 將所有特徵投影到一致的維度
        robot1_features = self.robot1_proj(robot1_features)
        robot2_features = self.robot2_proj(robot2_features)
        
        # R1 關注 R2
        r1_attending = self.r1_attention(robot2_features)
        r1_attending = self.r1_norm(r1_attending)
        
        # R2 關注 R1
        r2_attending = self.r2_attention(robot1_features)
        r2_attending = self.r2_norm(r2_attending)
        
        # 合併關注特徵
        cross_robot_features = tf.concat([r1_attending, r2_attending], axis=-1)
        
        # 使用融合層確保輸出維度正確
        cross_robot_features = self.fusion_layer(cross_robot_features)
        
        return cross_robot_features

class FrontierAttentionLayer(layers.Layer):
    """Frontier 注意力層，如圖3所示"""
    def __init__(self, max_frontiers, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_frontiers = max_frontiers
        self.d_model = d_model
        
    def build(self, input_shape):
        # 確保我們不依賴於輸入形狀的具體維度
        # 初始稠密層 - 將輸入投影到 64 維
        self.dense = layers.Dense(64)
        
        # 位置編碼
        self.pos_encoding = self.positional_encoding(self.max_frontiers, 64)
        
        # 自注意力機制的替代方案
        self.self_attention_dense_q = layers.Dense(64)
        self.self_attention_dense_k = layers.Dense(64)
        self.self_attention_dense_v = layers.Dense(64)
        self.self_attention_combine = layers.Dense(64)
        
        # 前饋網絡
        self.feed_forward_1 = layers.Dense(128, activation='relu')
        self.feed_forward_2 = layers.Dense(64)
        
        # 層正規化
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        
        # 最終投影，確保輸出維度符合要求
        self.output_projection = layers.Dense(self.d_model)
        
        super().build(input_shape)
        
    def positional_encoding(self, position, d_model):
        """創建位置編碼"""
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # 將正弦應用於偶數位置
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # 將餘弦應用於奇數位置
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def get_angles(self, pos, i, d_model):
        """計算位置編碼的角度"""
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
        
    def call(self, inputs, training=None):
        # 初始投影
        x = self.dense(inputs)  # [batch_size, max_frontiers, 64]
        
        # 添加位置編碼
        seq_len = tf.shape(x)[1]
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # 自注意力機制 - 手動實現以避免維度問題
        q = self.self_attention_dense_q(x)  # [batch_size, max_frontiers, 64]
        k = self.self_attention_dense_k(x)  # [batch_size, max_frontiers, 64]
        v = self.self_attention_dense_v(x)  # [batch_size, max_frontiers, 64]
        
        # 計算注意力分數
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # [batch_size, max_frontiers, max_frontiers]
        
        # 縮放
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # softmax 得到注意力權重
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # [batch_size, max_frontiers, max_frontiers]
        
        # 應用注意力權重
        attention_output = tf.matmul(attention_weights, v)  # [batch_size, max_frontiers, 64]
        
        # 組合並應用殘差連接
        attention_output = self.self_attention_combine(attention_output)
        x = self.layer_norm1(x + attention_output)  # 殘差連接和層正規化
        
        # 前饋網絡
        ffn_output = self.feed_forward_1(x)
        ffn_output = self.feed_forward_2(ffn_output)
        
        # 殘差連接和層正規化
        x = self.layer_norm2(x + ffn_output)
        
        # 最終投影確保輸出維度符合要求
        output = self.output_projection(x)
        
        return output


class TemporalAttentionModule(layers.Layer):
    """時間注意力模組，如圖2所示"""
    def __init__(self, d_model, memory_length=5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.memory_length = memory_length
        
    def build(self, input_shape):
        # 使用更簡單的方法處理記憶體
        # 記憶張量用於存儲過去狀態 - 每個批次有獨立的記憶體
        self.memory = self.add_weight(
            name='memory_tensor',
            shape=(1, self.memory_length, self.d_model),
            initializer='zeros',
            trainable=False
        )
        
        # 總是使用輸入投影層，不依賴輸入維度是否匹配
        self.input_projection = layers.Dense(self.d_model)
        
        # 簡化的注意力機制
        self.attention_weights = layers.Dense(self.memory_length, activation='softmax')
        
        # 輸出投影層
        self.output_projection = layers.Dense(self.d_model)
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # 投影輸入到模型維度
        projected_inputs = self.input_projection(inputs)  # [batch_size, d_model]
        
        # 為每個批次項目使用相同的記憶體
        repeated_memory = tf.tile(self.memory, [batch_size, 1, 1])  # [batch_size, memory_length, d_model]
        
        # 計算注意力得分 - 簡化版本，避免形狀問題
        # 擴展輸入以進行點積
        expanded_inputs = tf.expand_dims(projected_inputs, 1)  # [batch_size, 1, d_model]
        
        # 計算相似度 (使用點積)
        similarity = tf.reduce_sum(expanded_inputs * repeated_memory, axis=2)  # [batch_size, memory_length]
        
        # 使用 softmax 獲取注意力權重
        attention_weights = tf.nn.softmax(similarity, axis=1)  # [batch_size, memory_length]
        attention_weights = tf.expand_dims(attention_weights, axis=2)  # [batch_size, memory_length, 1]
        
        # 使用注意力權重加權記憶體
        attended_memory = attention_weights * repeated_memory  # [batch_size, memory_length, d_model]
        
        # 求和獲得上下文向量
        context_vector = tf.reduce_sum(attended_memory, axis=1)  # [batch_size, d_model]
        
        # 組合上下文和輸入
        combined = context_vector + projected_inputs  # [batch_size, d_model]
        
        # 最終輸出
        output = self.output_projection(combined)  # [batch_size, d_model]
        
        # 如果是訓練模式，更新記憶體
        if training:
            # 超級簡化的記憶體更新 - 完全避免批次維度問題
            # 只使用批次中的第一個項目更新記憶體
            first_item = tf.expand_dims(projected_inputs[0], 0)  # [1, d_model]
            first_item = tf.expand_dims(first_item, 1)  # [1, 1, d_model]
            
            # 移除最老的記憶體，添加新項目
            new_memory = tf.concat([first_item, self.memory[:, :-1, :]], axis=1)  # [1, memory_length, d_model]
            
            # 更新記憶體張量 - 使用指數移動平均
            self.memory.assign(0.9 * self.memory + 0.1 * new_memory)
        
        return output


class AdaptiveAttentionFusion(layers.Layer):
    """自適應注意力融合層，如圖1所示"""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        # 獲取輸入形狀
        frontier_shape, cross_robot_shape, temporal_shape, map_shape = input_shape
        
        # 安全地處理可能為 None 的維度
        # 如果最後一個維度是 None，我們使用 d_model 作為預設值
        frontier_dim = frontier_shape[-1] if frontier_shape[-1] is not None else self.d_model
        cross_robot_dim = cross_robot_shape[-1] if cross_robot_shape[-1] is not None else self.d_model
        temporal_dim = temporal_shape[-1] if temporal_shape[-1] is not None else self.d_model
        map_dim = map_shape[-1] if map_shape[-1] is not None else self.d_model
        
        # 計算合併後的特徵維度
        concat_dim = frontier_dim + cross_robot_dim + temporal_dim
        
        # 權重網絡 - 顯式指定輸入維度
        self.weight_network = layers.Dense(3, activation='softmax')
        
        # 殘差連接和歸一化
        self.residual = ResidualConnection()
        self.norm = LayerNormalization()
        
        # 特徵投影 - 確保所有特徵維度一致
        # 對所有特徵添加線性投影以統一維度到 d_model
        self.frontier_proj = layers.Dense(self.d_model)
        self.cross_robot_proj = layers.Dense(self.d_model)
        self.temporal_proj = layers.Dense(self.d_model)
        self.map_proj = layers.Dense(self.d_model)
        
        # 連接特徵的投影層
        self.concat_proj = layers.Dense(self.d_model * 3)
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        frontier_features, cross_robot_features, temporal_features, map_features = inputs
        
        # 對所有特徵進行投影以統一維度到 d_model
        frontier_features = self.frontier_proj(frontier_features)
        cross_robot_features = self.cross_robot_proj(cross_robot_features)
        temporal_features = self.temporal_proj(temporal_features)
        map_features = self.map_proj(map_features)
        
        # 合併前三個特徵類型以確定權重
        # 首先確保所有特徵都具有明確的形狀
        batch_size = tf.shape(frontier_features)[0]
        
        # 連接特徵
        concat_features = tf.concat([
            frontier_features,
            cross_robot_features,
            temporal_features
        ], axis=-1)
        
        # 使用投影層處理連接特徵，確保維度正確
        concat_features = self.concat_proj(concat_features)
        
        # 確定權重
        weights = self.weight_network(concat_features)
        
        # 應用權重到特徵
        w1 = tf.expand_dims(weights[:, 0], axis=-1)
        w2 = tf.expand_dims(weights[:, 1], axis=-1)
        w3 = tf.expand_dims(weights[:, 2], axis=-1)
        
        weighted_features = (w1 * frontier_features + 
                            w2 * cross_robot_features + 
                            w3 * temporal_features)
        
        # 應用殘差連接與地圖特徵
        fusion_features = self.residual(map_features, weighted_features)
        
        # 歸一化
        fusion_features = self.norm(fusion_features)
        
        return fusion_features


class EnhancedMultiRobotA2CModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        """Initialize the multi-robot A2C model with the architecture shown in the diagrams
        
        Args:
            input_shape: Input map shape, default (84, 84, 1)
            max_frontiers: Maximum number of frontier points, default 50
        """
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.d_model = 256  # Model dimension
        self.num_heads = 8  # Number of attention heads
        self.dff = 512  # Feed-forward network dimension
        self.dropout_rate = 0.1
        self.entropy_beta = 0.01  # Entropy regularization coefficient
        self.value_loss_weight = 0.5  # Value loss weight
        
        # Build separate models
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
    def pad_frontiers(self, frontiers):
        """Pad frontier points to fixed length and normalize coordinates"""
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
    
    def _build_perception_module(self, map_input):
        """Build the perception module for map processing"""
        # Use the new PerceptionModule class based on Image 3
        perception_module = PerceptionModule()
        map_features = perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features)
        
        return map_features_flat
    
    def _build_robot_state_encoder(self, robot_pos, robot_target, prefix):
        """Build the robot state encoder"""
        # Combine position and target
        robot_state = layers.Concatenate(name=f"{prefix}_state")([robot_pos, robot_target])
        
        # Encode robot state
        robot_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name=f"{prefix}_features"
        )(robot_state)
        robot_features = layers.BatchNormalization()(robot_features)
        
        return robot_features
    
    def _build_actor(self):
        """根據圖3中的架構構建 actor 網絡"""
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
        
        # 1. 地圖感知模塊
        perception_module = PerceptionModule()
        map_features = perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features)
        
        # 確保地圖特徵維度是固定的
        map_features_flat = layers.Dense(self.d_model, name='map_features_projection')(map_features_flat)
        
        # 2. 機器人狀態處理
        # 首先確保位置和目標形狀是固定的
        robot1_pos_encoded = layers.Dense(32, activation='relu')(robot1_pos)
        robot1_target_encoded = layers.Dense(32, activation='relu')(robot1_target)
        robot2_pos_encoded = layers.Dense(32, activation='relu')(robot2_pos)
        robot2_target_encoded = layers.Dense(32, activation='relu')(robot2_target)
        
        # 合併位置和目標
        robot1_state = layers.Concatenate()([robot1_pos_encoded, robot1_target_encoded])
        robot2_state = layers.Concatenate()([robot2_pos_encoded, robot2_target_encoded])
        
        # 確保機器人狀態維度是固定的
        robot1_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name='robot1_features'
        )(robot1_state)
        robot1_features = layers.BatchNormalization()(robot1_features)
        
        robot2_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name='robot2_features'
        )(robot2_state)
        robot2_features = layers.BatchNormalization()(robot2_features)
        
        # 3. 跨機器人注意力
        cross_robot_attention = CrossRobotAttention(self.d_model)
        cross_robot_features = cross_robot_attention([robot1_features, robot2_features])
        
        # 確保跨機器人特徵維度是固定的
        cross_robot_features = layers.Dense(self.d_model, name='cross_robot_projection')(cross_robot_features)
        
        # 4. Frontier 注意力層
        frontier_attention = FrontierAttentionLayer(
            max_frontiers=self.max_frontiers, 
            d_model=self.d_model
        )
        frontier_features = frontier_attention(frontier_input)
        frontier_features_flat = layers.Flatten()(frontier_features)
        
        # 確保 frontier 特徵維度是固定的
        frontier_features_flat = layers.Dense(self.d_model, name='frontier_projection')(frontier_features_flat)
        
        # 5. 時間注意力
        temporal_module = TemporalAttentionModule(d_model=self.d_model)
        robot1_temporal = temporal_module(robot1_features)
        robot2_temporal = temporal_module(robot2_features)
        temporal_features = layers.Concatenate()([robot1_temporal, robot2_temporal])
        
        # 確保時間特徵維度是固定的
        temporal_features = layers.Dense(self.d_model, name='temporal_projection')(temporal_features)
        
        # 6. 自適應注意力融合
        fusion_module = AdaptiveAttentionFusion(self.d_model)
        fusion_features = fusion_module([
            frontier_features_flat, cross_robot_features, temporal_features, map_features_flat
        ])
        
        # 7. 處理層 - 使用固定維度
        process_features = layers.Concatenate()(
            [fusion_features, map_features_flat, frontier_features_flat]
        )
        process_features = layers.Dense(self.d_model, activation='relu')(process_features)
        process_features = layers.Dropout(self.dropout_rate)(process_features)
        process_features = layers.Dense(64, activation='relu')(process_features)
        process_features = layers.Dropout(self.dropout_rate)(process_features)
        
        # 8. Robot1 策略頭 (如圖3的 Actor 網絡部分所示)
        robot1_dense = layers.Dense(64, activation='relu', name='robot1_actor_dense1')(process_features)
        robot1_dropout1 = layers.Dropout(self.dropout_rate)(robot1_dense)
        robot1_dense2 = layers.Dense(32, activation='relu', name='robot1_actor_dense2')(robot1_dropout1)
        robot1_dropout2 = layers.Dropout(self.dropout_rate)(robot1_dense2)
        robot1_logits = layers.Dense(
            self.max_frontiers,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name='robot1_policy_logits'
        )(robot1_dropout2)
        robot1_policy = layers.Lambda(
            lambda x: tf.nn.softmax(x),
            name='robot1_policy'
        )(robot1_logits)
        
        # 9. Robot2 策略頭
        robot2_dense = layers.Dense(64, activation='relu', name='robot2_actor_dense1')(process_features)
        robot2_dropout1 = layers.Dropout(self.dropout_rate)(robot2_dense)
        robot2_dense2 = layers.Dense(32, activation='relu', name='robot2_actor_dense2')(robot2_dropout1)
        robot2_dropout2 = layers.Dropout(self.dropout_rate)(robot2_dense2)
        robot2_logits = layers.Dense(
            self.max_frontiers,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name='robot2_policy_logits'
        )(robot2_dropout2)
        robot2_policy = layers.Lambda(
            lambda x: tf.nn.softmax(x),
            name='robot2_policy'
        )(robot2_logits)
        
        # 構建 Actor 模型
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
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=True
        )
        
        return actor_model

    def _build_critic(self):
        """根據圖3中的架構構建 critic 網絡"""
        # 輸入層 (與 actor 相同)
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(
            shape=(self.max_frontiers, 2),
            name='frontier_input'
        )
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 重用與 actor 相同的特徵提取架構
        perception_module = PerceptionModule()
        map_features = perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features)
        
        # 確保地圖特徵維度是固定的
        map_features_flat = layers.Dense(self.d_model, name='map_features_projection')(map_features_flat)
        
        # 機器人狀態處理
        robot1_pos_encoded = layers.Dense(32, activation='relu')(robot1_pos)
        robot1_target_encoded = layers.Dense(32, activation='relu')(robot1_target)
        robot2_pos_encoded = layers.Dense(32, activation='relu')(robot2_pos)
        robot2_target_encoded = layers.Dense(32, activation='relu')(robot2_target)
        
        # 合併位置和目標
        robot1_state = layers.Concatenate()([robot1_pos_encoded, robot1_target_encoded])
        robot2_state = layers.Concatenate()([robot2_pos_encoded, robot2_target_encoded])
        
        robot1_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name='robot1_features'
        )(robot1_state)
        robot1_features = layers.BatchNormalization()(robot1_features)
        
        robot2_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            name='robot2_features'
        )(robot2_state)
        robot2_features = layers.BatchNormalization()(robot2_features)
        
        # 跨機器人注意力
        cross_robot_attention = CrossRobotAttention(self.d_model)
        cross_robot_features = cross_robot_attention([robot1_features, robot2_features])
        
        # 確保跨機器人特徵維度是固定的
        cross_robot_features = layers.Dense(self.d_model, name='cross_robot_projection')(cross_robot_features)
        
        # Frontier 注意力層
        frontier_attention = FrontierAttentionLayer(
            max_frontiers=self.max_frontiers, 
            d_model=self.d_model
        )
        frontier_features = frontier_attention(frontier_input)
        frontier_features_flat = layers.Flatten()(frontier_features)
        
        # 確保 frontier 特徵維度是固定的
        frontier_features_flat = layers.Dense(self.d_model, name='frontier_projection')(frontier_features_flat)
        
        # 時間注意力
        temporal_module = TemporalAttentionModule(d_model=self.d_model)
        robot1_temporal = temporal_module(robot1_features)
        robot2_temporal = temporal_module(robot2_features)
        temporal_features = layers.Concatenate()([robot1_temporal, robot2_temporal])
        
        # 確保時間特徵維度是固定的
        temporal_features = layers.Dense(self.d_model, name='temporal_projection')(temporal_features)
        
        # 自適應注意力融合
        fusion_module = AdaptiveAttentionFusion(self.d_model)
        fusion_features = fusion_module([
            frontier_features_flat, cross_robot_features, temporal_features, map_features_flat
        ])
        
        # 處理層 - 使用固定維度
        process_features = layers.Concatenate()(
            [fusion_features, map_features_flat, frontier_features_flat]
        )
        process_features = layers.Dense(self.d_model, activation='relu')(process_features)
        process_features = layers.Dropout(self.dropout_rate)(process_features)
        process_features = layers.Dense(64, activation='relu')(process_features)
        process_features = layers.Dropout(self.dropout_rate)(process_features)
        
        # Robot1 價值頭 (如圖3的 Critic 網絡部分所示)
        robot1_critic = layers.Dense(64, activation='relu', name='robot1_critic_dense1')(process_features)
        robot1_critic = layers.Dropout(self.dropout_rate)(robot1_critic)
        robot1_critic = layers.Dense(32, activation='relu', name='robot1_critic_dense2')(robot1_critic)
        robot1_critic = layers.Dropout(self.dropout_rate)(robot1_critic)
        robot1_value = layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name='robot1_value'
        )(robot1_critic)
        
        # Robot2 價值頭
        robot2_critic = layers.Dense(64, activation='relu', name='robot2_critic_dense1')(process_features)
        robot2_critic = layers.Dropout(self.dropout_rate)(robot2_critic)
        robot2_critic = layers.Dense(32, activation='relu', name='robot2_critic_dense2')(robot2_critic)
        robot2_critic = layers.Dropout(self.dropout_rate)(robot2_critic)
        robot2_value = layers.Dense(
            1,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name='robot2_value'
        )(robot2_critic)
        
        # 構建 Critic 模型
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
        
        # 優化器設置
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
                   robot1_advantages, robot2_advantages):
        """Improved Actor training with enhanced gradient handling"""
        with tf.GradientTape() as tape:
            # Get policy predictions
            policy_dict = self.actor({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            # Use epsilon smoothing to avoid extreme values
            epsilon = 1e-8
            robot1_policy_smoothed = (1 - epsilon) * policy_dict['robot1_policy'] + epsilon / self.max_frontiers
            robot2_policy_smoothed = (1 - epsilon) * policy_dict['robot2_policy'] + epsilon / self.max_frontiers
            
            # Calculate log probabilities
            actions_one_hot_1 = tf.one_hot(robot1_actions, self.max_frontiers)
            actions_one_hot_2 = tf.one_hot(robot2_actions, self.max_frontiers)
            
            # Use safe log calculation
            log_prob_1 = tf.math.log(tf.reduce_sum(robot1_policy_smoothed * actions_one_hot_1, axis=1) + 1e-10)
            log_prob_2 = tf.math.log(tf.reduce_sum(robot2_policy_smoothed * actions_one_hot_2, axis=1) + 1e-10)
            
            # Calculate policy loss
            policy_loss_coef = 1.0  # Policy loss coefficient
            robot1_loss = -tf.reduce_mean(log_prob_1 * robot1_advantages) * policy_loss_coef
            robot2_loss = -tf.reduce_mean(log_prob_2 * robot2_advantages) * policy_loss_coef
            
            # Entropy regularization - use smaller coefficient
            entropy_coef = 0.001  # Reduced entropy coefficient
            entropy_1 = -tf.reduce_mean(tf.reduce_sum(robot1_policy_smoothed * tf.math.log(robot1_policy_smoothed + 1e-10), axis=1))
            entropy_2 = -tf.reduce_mean(tf.reduce_sum(robot2_policy_smoothed * tf.math.log(robot2_policy_smoothed + 1e-10), axis=1))
            
            # Total loss - policy loss minus entropy reward
            total_loss = (robot1_loss + robot2_loss) - entropy_coef * (entropy_1 + entropy_2)
        
        # Calculate gradients
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        
        # Gradient processing - important step
        # 1. Replace NaN gradients with zero
        grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else g for g in grads]
        
        # 2. Handle gradient explosion
        clipped_grads = []
        for g in grads:
            if g is not None:
                # Detect abnormally large gradients
                norm = tf.norm(g)
                
                # If norm is too large, clip it
                if norm > 1.0:
                    g = g * (1.0 / norm)
                    
                # If gradient is extremely small but non-zero, slightly amplify it
                elif 0 < norm < 1e-4:
                    scale_factor = 1e-4 / (norm + 1e-10)
                    g = g * tf.minimum(scale_factor, 10.0)  # Limit max amplification factor
                    
            clipped_grads.append(g)
        
        # Global gradient clipping
        clipped_grads, _ = tf.clip_by_global_norm(clipped_grads, 1.0)
        
        # Apply gradients
        self.actor.optimizer.apply_gradients(zip(clipped_grads, self.actor.trainable_variables))
        
        return {
            'total_policy_loss': total_loss,
            'robot1_policy_loss': robot1_loss,
            'robot2_policy_loss': robot2_loss,
            'robot1_entropy': entropy_1,
            'robot2_entropy': entropy_2
        }
    
    def train_critic(self, states, frontiers, robot1_pos, robot2_pos, 
                    robot1_target, robot2_target, 
                    robot1_returns, robot2_returns):
        """Improved Critic training with enhanced gradient handling"""
        with tf.GradientTape() as tape:
            # Get value predictions
            values = self.critic({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            # Calculate robot1's value loss - use Huber loss to avoid outliers
            robot1_value_loss = tf.keras.losses.Huber(delta=1.0)(
                robot1_returns, values['robot1_value'])
            
            # Calculate robot2's value loss
            robot2_value_loss = tf.keras.losses.Huber(delta=1.0)(
                robot2_returns, values['robot2_value'])
            
            # Total value loss
            value_loss = robot1_value_loss + robot2_value_loss
        
        # Calculate gradients
        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        
        # Gradient processing
        # 1. Replace NaN gradients with zero
        grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else g for g in grads]
        
        # 2. Clip abnormal gradients
        clipped_grads = []
        for g in grads:
            if g is not None:
                # Detect abnormally large gradients
                norm = tf.norm(g)
                
                # If norm is too large, clip it
                if norm > 1.0:
                    g = g * (1.0 / norm)
            clipped_grads.append(g)
        
        # Global gradient clipping - use stricter clipping for Critic
        clipped_grads, _ = tf.clip_by_global_norm(clipped_grads, 0.5)
        
        # Apply gradients
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
                   robot1_old_logits, robot2_old_logits):
        """Improved batch training process"""
        # Train Actor
        actor_metrics = self.train_actor(
            states, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target,
            robot1_actions, robot2_actions, 
            robot1_advantages, robot2_advantages
        )
        
        # Train Critic
        critic_metrics = self.train_critic(
            states, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target,
            robot1_returns, robot2_returns
        )
        
        # Calculate entropy
        # For stable calculation, we use model predictions instead of passed old values
        policy_dict = self.actor.predict({
            'map_input': states,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        }, verbose=0)
        
        # Calculate entropy - use epsilon smoothing
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
        
        # Combine all metrics
        return {
            'total_loss': actor_metrics['total_policy_loss'] + critic_metrics['total_value_loss'],
            'robot1_policy_loss': actor_metrics['robot1_policy_loss'],
            'robot2_policy_loss': actor_metrics['robot2_policy_loss'],
            'robot1_value_loss': critic_metrics['robot1_value_loss'],
            'robot2_value_loss': critic_metrics['robot2_value_loss'],
            'robot1_entropy': robot1_entropy,
            'robot2_entropy': robot2_entropy,
            'gradient_norm': 0.0  # Keep interface consistent
        }
    
    def predict(self, state, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target):
        """Predict action distributions and state values"""
        # Ensure input shapes are correct
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
        
        # Build input dictionary
        inputs = {
            'map_input': state,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        }
        
        # Get Actor outputs
        actor_outputs = self.actor.predict(inputs, verbose=0)
        
        # Get Critic outputs
        critic_outputs = self.critic.predict(inputs, verbose=0)
        
        # Check validity of policy outputs
        for key in ['robot1_policy', 'robot2_policy']:
            policy = actor_outputs[key]
            if np.any(np.isnan(policy)):
                print(f"Warning: {key} contains NaN values, replacing with uniform distribution")
                actor_outputs[key] = np.ones_like(policy) / policy.shape[-1]
            elif np.any(policy < 0) or np.any(policy > 1):
                print(f"Warning: {key} contains out-of-range values, clipping")
                actor_outputs[key] = np.clip(policy, 0, 1)
                # Renormalize
                actor_outputs[key] = actor_outputs[key] / np.sum(actor_outputs[key], axis=-1, keepdims=True)
                
        # Return prediction results
        return {
            'robot1_policy': actor_outputs['robot1_policy'],
            'robot2_policy': actor_outputs['robot2_policy'],
            'robot1_value': critic_outputs['robot1_value'],
            'robot2_value': critic_outputs['robot2_value'],
            'robot1_logits': actor_outputs['robot1_logits'],
            'robot2_logits': actor_outputs['robot2_logits']
        }
    
    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        """Improved returns and advantages calculation, enhancing numerical stability"""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        
        # Keep reasonable reward scale
        reward_scale = 1.0
        scaled_rewards = rewards * reward_scale
        
        # GAE parameters
        gamma = 0.99  # Discount factor
        lambda_param = 0.95  # GAE balance parameter
        
        # Calculate backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_val = values[t+1]
            
            # Calculate TD error
            delta = scaled_rewards[t] + gamma * next_val * next_non_terminal - values[t]
            
            # Calculate GAE
            gae = delta + gamma * lambda_param * next_non_terminal * gae
            
            # Save advantage and return
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Robust normalization
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        
        # Normalize advantages
        normalized_advantages = (advantages - adv_mean) / adv_std
        
        # Limit range to avoid extreme values
        max_value = 15.0  # Larger range to maintain signal strength
        normalized_advantages = np.clip(normalized_advantages, -max_value, max_value)
        
        # Periodically print advantage statistics for monitoring
        if np.random.random() < 0.1:  # 10% chance to print, reduce log noise
            print(f"Advantage statistics: mean={np.mean(normalized_advantages):.2f}, "
                  f"std={np.std(normalized_advantages):.2f}, "
                  f"min={np.min(normalized_advantages):.2f}, "
                  f"max={np.max(normalized_advantages):.2f}")
        
        return returns, normalized_advantages
    
    def save(self, filepath):
        """Save the model in .h5 format"""
        # Save Actor model
        actor_path = filepath + '_actor.h5'
        print(f"Saving actor model to: {actor_path}")
        self.actor.save(actor_path, save_format='h5')
        
        # Save Critic model
        critic_path = filepath + '_critic.h5'
        print(f"Saving critic model to: {critic_path}")
        self.critic.save(critic_path, save_format='h5')
        
        # Save additional configuration information
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
            
        print(f"Model saved in .h5 format")
    
    def load(self, filepath):
        """Load model in .h5 format"""
        # Create custom objects dictionary
        custom_objects = {
            'LayerNormalization': LayerNormalization,
            'MultiHeadAttention': MultiHeadAttention,
            'PositionalEncoding': PositionalEncoding,
            'FeedForward': FeedForward,
            'SpatialAttention': SpatialAttention,
            'ResidualConnection': ResidualConnection,
            'PerceptionModule': PerceptionModule,
            'CrossRobotAttention': CrossRobotAttention,
            'FrontierAttentionLayer': FrontierAttentionLayer,
            'TemporalAttentionModule': TemporalAttentionModule,
            'AdaptiveAttentionFusion': AdaptiveAttentionFusion
        }
        
        try:
            # Load configuration
            with open(filepath + '_config.json', 'r') as f:
                config = json.load(f)
                
            # Update model configuration
            self.input_shape = tuple(config['input_shape'])
            self.max_frontiers = config['max_frontiers']
            self.d_model = config['d_model']
            self.num_heads = config['num_heads']
            self.dff = config['dff']
            self.dropout_rate = config['dropout_rate']
            
            # Load actor model
            actor_path = filepath + '_actor.h5'
            print(f"Loading actor model: {actor_path}")
            self.actor = tf.keras.models.load_model(
                actor_path,
                custom_objects=custom_objects
            )
            
            # Load critic model
            critic_path = filepath + '_critic.h5'
            print(f"Loading critic model: {critic_path}")
            self.critic = tf.keras.models.load_model(
                critic_path,
                custom_objects=custom_objects
            )
            
            # Set optimizers
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
            
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False