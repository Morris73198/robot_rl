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

    def get_config(self):
        return super().get_config()

class EnhancedSpatialAttention(layers.Layer):
    """記憶體優化版的增強空間注意力層"""
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        self.channels = input_shape[-1]
        
        # 簡化的多尺度卷積
        self.conv3x3 = layers.Conv2D(self.channels//self.reduction_ratio, 3, padding='same', use_bias=False)
        
        # 注意力映射
        self.attention_conv = layers.Conv2D(1, 1, padding='same', use_bias=False)
        self.norm = LayerNormalization()
        
        super().build(input_shape)
        
    def call(self, inputs):
        # 空間注意力 - 簡化處理
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # 多尺度特徵提取 - 只使用一個卷積核大小
        feat_3x3 = self.conv3x3(inputs)
        
        # 融合特徵 - 簡化連接
        multi_scale = tf.concat([
            tf.reduce_mean(feat_3x3, axis=-1, keepdims=True),
            avg_pool,
            max_pool
        ], axis=-1)
        
        # 生成空間注意力圖
        spatial_attn = self.attention_conv(multi_scale)
        spatial_attn = tf.sigmoid(spatial_attn)
        
        # 應用空間注意力
        refined = inputs * spatial_attn
        output = self.norm(refined)
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config

class CrossRobotAttention(layers.Layer):
    """記憶體優化版的機器人間交互注意力層"""
    def __init__(self, d_model, num_heads=2, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # 使用更輕量級的注意力機制
        self.attention = layers.Attention(use_scale=True, dropout=self.dropout_rate)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        super().build(input_shape)
        
    def call(self, inputs):
        # 提取兩個機器人的特徵
        robot1_features, robot2_features = inputs
        
        # 擴展輸入維度以適應注意力機制
        robot1_expanded = tf.expand_dims(robot1_features, axis=1)
        robot2_expanded = tf.expand_dims(robot2_features, axis=1)
        
        # 機器人1注意機器人2（使用一般注意力代替多頭注意力）
        r1_attends_r2 = self.attention([robot1_expanded, robot2_expanded, robot2_expanded])
        r1_enhanced = self.layernorm1(robot1_expanded + r1_attends_r2)
        
        # 機器人2注意機器人1
        r2_attends_r1 = self.attention([robot2_expanded, robot1_expanded, robot1_expanded])
        r2_enhanced = self.layernorm2(robot2_expanded + r2_attends_r1)
        
        # 移除多餘的維度
        return tf.squeeze(r1_enhanced, axis=1), tf.squeeze(r2_enhanced, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

class TemporalAttention(layers.Layer):
    """記憶體優化版的時間注意力層"""
    def __init__(self, d_model, memory_length=5, num_heads=2, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # 減少記憶長度以節省記憶體
        self.memory_length = memory_length
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # 將記憶體初始化為可訓練的權重而非不可訓練的變數
        self.memory = self.add_weight(
            name="temporal_memory",
            shape=[1, self.memory_length, self.d_model],
            initializer=tf.initializers.zeros(),
            trainable=True  # 使記憶可訓練，可以更好地融入模型
        )
        
        # 使用單一頭注意力以減少記憶體需求
        self.attention = layers.Attention(use_scale=True, dropout=self.dropout_rate)
        self.layernorm = LayerNormalization()
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        # 確保輸入是 2D [batch_size, d_model]
        if len(inputs.shape) > 2:
            inputs = tf.reduce_mean(inputs, axis=list(range(1, len(tf.shape(inputs))-1)))
        
        # 擴展輸入維度以適應注意力機制 [batch_size, 1, d_model]
        query = tf.expand_dims(inputs, axis=1)
        batch_size = tf.shape(query)[0]
        
        # 將記憶複製到批次維度
        key_value = tf.repeat(self.memory, batch_size, axis=0)
        
        # 使用內建注意力層代替自定義多頭注意力
        attended = self.attention([query, key_value, key_value])
        output = self.layernorm(query + attended)
        
        # 在訓練時更新記憶
        if training:
            # 使用移動平均更新記憶
            new_memory = tf.concat([self.memory[:, 1:, :], query[:1]], axis=1)
            self.memory.assign(0.9 * self.memory + 0.1 * new_memory)  # 緩慢更新
        
        # 移除多餘的維度
        return tf.squeeze(output, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'memory_length': self.memory_length,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

class AdaptiveAttentionFusion(layers.Layer):
    """記憶體優化版的自適應注意力融合層"""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        # 簡化結構，只使用一個全連接層作為權重生成
        self.weight_net = layers.Dense(3, activation='softmax')
        self.layernorm = LayerNormalization()
        super().build(input_shape)
        
    def call(self, inputs):
        # 假設輸入是[原始特徵, 空間注意力特徵, 交叉機器人特徵, 時間特徵]
        original, spatial_attn, cross_robot_attn, temporal_attn = inputs
        
        # 確保所有輸入特徵都是 2D [batch_size, features]
        spatial_attn_flat = tf.reduce_mean(spatial_attn, axis=list(range(1, len(tf.shape(spatial_attn))-1))) if len(spatial_attn.shape) > 2 else spatial_attn
        cross_robot_attn_flat = tf.reduce_mean(cross_robot_attn, axis=list(range(1, len(tf.shape(cross_robot_attn))-1))) if len(cross_robot_attn.shape) > 2 else cross_robot_attn
        temporal_attn_flat = tf.reduce_mean(temporal_attn, axis=list(range(1, len(tf.shape(temporal_attn))-1))) if len(temporal_attn.shape) > 2 else temporal_attn
        
        # 生成簡化版的融合權重
        features_concat = tf.concat([
            spatial_attn_flat,
            cross_robot_attn_flat,
            temporal_attn_flat
        ], axis=-1)
        
        weights = self.weight_net(features_concat)
        
        # 直接在特徵空間應用權重，避免高維乘法運算
        w1, w2, w3 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
        # 使用加權和代替直接乘法，減少記憶體需求
        weighted_spatial = spatial_attn_flat * w1
        weighted_cross = cross_robot_attn_flat * w2
        weighted_temporal = temporal_attn_flat * w3
        
        # 簡單加和並正規化
        fused = weighted_spatial + weighted_cross + weighted_temporal
        output = self.layernorm(original + fused)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config

class PositionalEncoding(layers.Layer):
    """位置編碼層 - 簡化版"""
    def __init__(self, max_position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.d_model = d_model
        
    def build(self, input_shape):
        # 使用簡化的線性位置編碼代替複雜的正弦/餘弦編碼
        position_enc = np.zeros((self.max_position, self.d_model))
        for pos in range(self.max_position):
            for i in range(self.d_model):
                position_enc[pos, i] = pos / np.power(10000, 2 * (i // 2) / self.d_model)
                
        # 轉換為張量並添加批次維度
        self.pos_encoding = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        self.pos_encoding = tf.expand_dims(self.pos_encoding, axis=0)
        super().build(input_shape)
        
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_position': self.max_position,
            'd_model': self.d_model
        })
        return config

class FeedForward(layers.Layer):
    """前饋神經網路層 - 簡化版"""
    def __init__(self, d_model, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.dense1 = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()
        
    def call(self, x, training=None):
        ffn_output = self.dense1(x)
        ffn_output = self.dropout(ffn_output, training=training)
        ffn_output = self.dense2(ffn_output)
        ffn_output = self.dropout(ffn_output, training=training)
        ffn_output = self.layer_norm(x + ffn_output)
        return ffn_output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config

class MultiRobotACModel:
    """多機器人 Actor-Critic 模型 - 記憶體優化版"""
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):  # 從50減少到25
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.d_model = 128  # 從256減少到128
        self.num_heads = 4  # 從8減少到4
        self.dff = 256  # 從512減少到256
        self.dropout_rate = 0.1
        
        # 創建 Actor 和 Critic 網絡
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
    def _build_perception_module(self, inputs):
        """構建共享的感知模塊 - 記憶體優化版"""
        # 降低輸入分辨率
        x = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
        
        # 使用不同大小的卷積核進行特征提取，但減少特徵數量
        conv_configs = [
            {'filters': 16, 'kernel_size': 3, 'strides': 2},  # 從32減半到16
            {'filters': 16, 'kernel_size': 5, 'strides': 2},
            {'filters': 16, 'kernel_size': 7, 'strides': 2}
        ]
        
        features = []
        for config in conv_configs:
            branch = layers.Conv2D(
                filters=config['filters'],
                kernel_size=config['kernel_size'],
                strides=config['strides'],
                padding='same',
                kernel_regularizer=regularizers.l2(0.01)
            )(x)
            branch = layers.BatchNormalization()(branch)
            branch = layers.Activation('relu')(branch)
            # 使用優化版的空間注意力
            branch = EnhancedSpatialAttention()(branch)
            features.append(branch)
            
        # 合併特征
        x = layers.Add()(features)
        x = layers.Conv2D(32, 1, padding='same')(x)  # 從64減半到32
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        return x
        
    def _build_shared_features(self, map_features, frontier_input, robot1_state, robot2_state):
        """構建記憶體優化版的共享特徵提取層"""
        # 減少中間層大小，從64降到32
        reduced_dim = 32
        
        # 處理 frontier 特征，減少中間特徵維度
        frontier_features = layers.Dense(reduced_dim, activation='relu')(frontier_input)
        frontier_features = layers.Dropout(self.dropout_rate)(frontier_features)
        
        # 使用更簡單的位置編碼
        # 添加簡化的位置編碼（線性位置編碼代替複雜的正弦/餘弦編碼）
        pos_indices = tf.range(start=0, limit=tf.shape(frontier_features)[1], delta=1)
        pos_indices = tf.cast(pos_indices, tf.float32)
        pos_indices = tf.expand_dims(pos_indices, axis=-1)
        pos_indices = tf.tile(pos_indices, [1, reduced_dim])
        pos_indices = pos_indices / tf.cast(self.max_frontiers, tf.float32)
        
        # 擴展位置編碼到批次維度
        batch_size = tf.shape(frontier_features)[0]
        pos_encoding = tf.expand_dims(pos_indices, axis=0)
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])
        
        # 添加位置編碼到特徵 (加法代替複雜的位置編碼層)
        frontier_features_with_pos = frontier_features + 0.1 * pos_encoding
        
        # 使用標準的Self-Attention而非自定義的多頭注意力
        # 先將特徵尺寸調整為注意力機制要求的形狀
        attention_output = layers.MultiHeadAttention(
            num_heads=2,  # 減少頭數
            key_dim=reduced_dim // 2,  # 減少每個頭的維度
            dropout=self.dropout_rate
        )(frontier_features_with_pos, frontier_features_with_pos)
        
        # 添加殘差連接和層正規化
        frontier_features = layers.LayerNormalization()(frontier_features_with_pos + attention_output)
        
        # 使用一個簡單的前饋網路代替FeedForward類
        frontier_features = layers.Dense(reduced_dim * 2, activation='relu')(frontier_features)
        frontier_features = layers.Dropout(self.dropout_rate)(frontier_features)
        frontier_features = layers.Dense(reduced_dim)(frontier_features)
        frontier_features = layers.LayerNormalization()(frontier_features_with_pos + frontier_features)
        
        # 處理機器人狀態，減少特徵維度
        robot1_feat = layers.Dense(reduced_dim // 2, activation='relu')(robot1_state)
        robot2_feat = layers.Dense(reduced_dim // 2, activation='relu')(robot2_state)
        
        # 添加優化版的跨機器人注意力
        robot1_enhanced, robot2_enhanced = CrossRobotAttention(
            d_model=reduced_dim // 2, 
            num_heads=1  # 減少頭數
        )([robot1_feat, robot2_feat])
        
        # 融合機器人特徵
        robot_features = layers.Concatenate()([robot1_enhanced, robot2_enhanced])
        
        # 處理時間維度特徵，全局平均池化減少參數量
        temporal_frontier_feature = layers.GlobalAveragePooling1D()(frontier_features)
        
        # 使用較小的特徵維度
        temporal_frontier_feature = layers.Dense(reduced_dim, activation='relu')(temporal_frontier_feature)
        
        # 添加優化版的時間注意力
        temporal_features = TemporalAttention(
            d_model=reduced_dim,
            memory_length=5,  # 減少記憶長度
            num_heads=1  # 減少頭數
        )(temporal_frontier_feature)
        
        # 使用較小的特徵維度
        map_features_reduced = layers.Dense(reduced_dim, activation='relu')(map_features)
        frontier_global = layers.GlobalAveragePooling1D()(frontier_features)
        frontier_global = layers.Dense(reduced_dim, activation='relu')(frontier_global)
        robot_features_reduced = layers.Dense(reduced_dim, activation='relu')(robot_features)
        
        # 使用優化版的自適應注意力融合
        fused_features = AdaptiveAttentionFusion(
            d_model=reduced_dim
        )([
            map_features_reduced,
            frontier_global,
            robot_features_reduced,
            temporal_features
        ])
        
        # 使用較小的隱藏層尺寸
        combined_features = layers.Concatenate()([
            map_features_reduced,
            fused_features,
            robot_features_reduced
        ])
        
        # 最終特徵處理，減少隱藏層大小
        x = layers.Dense(128, activation='relu')(combined_features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        return x
        
    def _build_policy_head(self, features, name_prefix):
        """構建策略輸出頭"""
        x = layers.Dense(64, activation='relu')(features)  # 從128減少到64
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)  # 從64減少到32
        x = layers.Dropout(self.dropout_rate)(x)
        policy = layers.Dense(
            self.max_frontiers,
            activation='softmax',
            name=name_prefix
        )(x)
        return policy

    def _build_value_head(self, features, name_prefix):
        """構建價值輸出頭"""
        x = layers.Dense(64, activation='relu')(features)  # 從128減少到64
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu')(x)  # 從64減少到32
        x = layers.Dropout(self.dropout_rate)(x)
        value = layers.Dense(1, name=name_prefix)(x)
        return value
        
    def _build_actor(self):
        """構建 Actor 網絡"""
        # 輸入層
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 特征提取
        map_features = self._build_perception_module(map_input)
        
        # 機器人狀態編碼
        robot1_state = layers.Concatenate()([robot1_pos, robot1_target])
        robot2_state = layers.Concatenate()([robot2_pos, robot2_target])
        
        # 共享特征提取
        shared_features = self._build_shared_features(
            map_features, 
            frontier_input, 
            robot1_state, 
            robot2_state
        )
        
        # 輸出層
        robot1_policy = self._build_policy_head(shared_features, 'robot1_policy')
        robot2_policy = self._build_policy_head(shared_features, 'robot2_policy')
        
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
                'robot2_policy': robot2_policy
            }
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer)
        
        return model
        
    def _build_critic(self):
        """構建 Critic 網絡"""
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 特征提取
        map_features = self._build_perception_module(map_input)
        
        # 機器人狀態編碼
        robot1_state = layers.Concatenate()([robot1_pos, robot1_target])
        robot2_state = layers.Concatenate()([robot2_pos, robot2_target])
        
        # 共享特征提取
        shared_features = self._build_shared_features(
            map_features, 
            frontier_input, 
            robot1_state, 
            robot2_state
        )
        
        # 輸出層
        robot1_value = self._build_value_head(shared_features, 'robot1_value')
        robot2_value = self._build_value_head(shared_features, 'robot2_value')
        
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
                'robot1_value': robot1_value,
                'robot2_value': robot2_value
            }
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model

    def predict_policy(self, state, frontiers, robot1_pos, robot2_pos, 
                      robot1_target, robot2_target):
        """預測動作概率分布"""
        return self.actor.predict({
            'map_input': state,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        }, verbose=0)
    
    def predict_value(self, state, frontiers, robot1_pos, robot2_pos, 
                     robot1_target, robot2_target):
        """預測狀態值"""
        return self.critic.predict({
            'map_input': state,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        }, verbose=0)
    
    def train_actor(self, states, frontiers, robot1_pos, robot2_pos,
                robot1_target, robot2_target, actions, advantages):
        """訓練 Actor 網路"""
        with tf.GradientTape() as tape:
            # 直接使用self.actor而不是self.model.actor
            policy_dict = self.actor({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            robot1_loss = self._compute_policy_loss(
                policy_dict['robot1_policy'],
                actions['robot1'],
                advantages['robot1']
            )
            robot2_loss = self._compute_policy_loss(
                policy_dict['robot2_policy'],
                actions['robot2'],
                advantages['robot2']
            )
            
            total_loss = robot1_loss + robot2_loss
            
        # 應用梯度裁剪
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.actor.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables))
            
        return total_loss
    
    def train_critic(self, states, frontiers, robot1_pos, robot2_pos, 
                    robot1_target, robot2_target, returns):
        """訓練 Critic 網路"""
        with tf.GradientTape() as tape:
            # 直接使用self.critic而不是self.model.critic
            values = self.critic({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            # 計算critic loss
            robot1_value_loss = tf.keras.losses.Huber()(
                returns['robot1'], values['robot1_value'])
            robot2_value_loss = tf.keras.losses.Huber()(
                returns['robot2'], values['robot2_value'])
            
            value_loss = robot1_value_loss + robot2_value_loss
            
        # 應用梯度裁剪
        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.critic.optimizer.apply_gradients(
            zip(grads, self.critic.trainable_variables))
            
        return value_loss
    
    def _compute_policy_loss(self, policy, actions, advantages):
        """計算策略損失
        
        Args:
            policy: 策略網絡輸出的動作概率分布
            actions: 實際執行的動作
            advantages: 計算出的優勢值
        
        Returns:
            policy_loss: 策略損失值
        """
        # 將動作轉換為 one-hot 編碼
        actions_one_hot = tf.one_hot(actions, self.max_frontiers)
        
        # 計算所選動作的對數概率
        log_prob = tf.math.log(tf.reduce_sum(policy * actions_one_hot, axis=1) + 1e-10)
        
        # 計算策略損失
        policy_loss = -tf.reduce_mean(log_prob * advantages)
        
        # 添加熵正則化以鼓勵探索，但減少熵權重
        entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1)
        entropy_bonus = 0.005 * tf.reduce_mean(entropy)  # 從0.01減少到0.005
        
        return policy_loss - entropy_bonus
    
    def save(self, filepath):
        """保存模型到 H5 格式"""
        print("保存模型到 H5 格式...")
        
        try:
            # 修改：使用 H5 格式保存
            actor_path = filepath + '_actor.h5'
            critic_path = filepath + '_critic.h5'
            
            print(f"保存 actor 模型到: {actor_path}")
            self.actor.save(actor_path, save_format='h5')
            
            print(f"保存 critic 模型到: {critic_path}")
            self.critic.save(critic_path, save_format='h5')
            
            # 保存配置不變
            config = {
                'input_shape': self.input_shape,
                'max_frontiers': self.max_frontiers,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'dff': self.dff,
                'dropout_rate': self.dropout_rate
            }
            
            with open(filepath + '_config.json', 'w') as f:
                json.dump(config, f, indent=4)
                
            print("H5 格式模型保存成功")
            return True
            
        except Exception as e:
            print(f"保存模型時出錯: {str(e)}")
            return False

    def load(self, filepath):
        """載入 H5 格式的模型"""
        print("載入 H5 格式模型...")
        
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
            
            # 定義自定義對象
            custom_objects = {
                'LayerNormalization': LayerNormalization,
                'MultiHeadAttention': MultiHeadAttention,
                'PositionalEncoding': PositionalEncoding,
                'FeedForward': FeedForward,
                'SpatialAttention': SpatialAttention,
                'EnhancedSpatialAttention': EnhancedSpatialAttention,
                'CrossRobotAttention': CrossRobotAttention,
                'TemporalAttention': TemporalAttention,
                'AdaptiveAttentionFusion': AdaptiveAttentionFusion
            }
            
            # 修改：載入 H5 格式模型
            actor_path = filepath + '_actor.h5'
            critic_path = filepath + '_critic.h5'
            
            print(f"載入 actor 模型: {actor_path}")
            self.actor = tf.keras.models.load_model(
                actor_path,
                custom_objects=custom_objects
            )
            
            print(f"載入 critic 模型: {critic_path}")
            self.critic = tf.keras.models.load_model(
                critic_path,
                custom_objects=custom_objects
            )
            
            print("H5 格式模型載入成功")
            return True
            
        except Exception as e:
            print(f"載入模型時出錯: {str(e)}")
            return False

    def verify_model(self):
        """驗證模型"""
        print("驗證模型...")
        test_inputs = {
            'map_input': np.zeros((1, *self.input_shape)),
            'frontier_input': np.zeros((1, self.max_frontiers, 2)),
            'robot1_pos_input': np.zeros((1, 2)),
            'robot2_pos_input': np.zeros((1, 2)),
            'robot1_target_input': np.zeros((1, 2)),
            'robot2_target_input': np.zeros((1, 2))
        }
        
        try:
            # 測試前向傳播
            actor_outputs = self.actor.predict(test_inputs, verbose=0)
            critic_outputs = self.critic.predict(test_inputs, verbose=0)
            
            # 檢查輸出形狀
            assert actor_outputs['robot1_policy'].shape == (1, self.max_frontiers)
            assert actor_outputs['robot2_policy'].shape == (1, self.max_frontiers)
            assert critic_outputs['robot1_value'].shape == (1, 1)
            assert critic_outputs['robot2_value'].shape == (1, 1)
            
            print("模型驗證通過")
            return True
            
        except Exception as e:
            print(f"模型驗證失敗: {str(e)}")
            return False