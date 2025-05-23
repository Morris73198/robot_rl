import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers
import json
import os

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
    """多機器人 Actor-Critic 模型 - 簡化版（參考 network2 的架構）"""
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.d_model = 128  # 參考 network2
        self.num_heads = 4  # 參考 network2
        self.dff = 256  # 參考 network2
        self.dropout_rate = 0.1
        
        # 創建 Actor 和 Critic 網絡
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # 打印模型參數統計
        self._print_model_summary()
    
    def _print_model_summary(self):
        """打印模型參數統計"""
        actor_params = self.actor.count_params()
        critic_params = self.critic.count_params()
        total_params = actor_params + critic_params
        
        print("\n===== 模型參數統計 =====")
        print(f"Actor 參數: {actor_params:,}")
        print(f"Critic 參數: {critic_params:,}")
        print(f"總參數數量: {total_params:,}")
        print(f"預期權重檔案大小: {total_params * 4 / (1024*1024):.2f} MB")
        print("========================\n")
        
    def _build_actor(self):
        """構建 Actor 網絡 - 完全參考 network2 的簡化架構"""
        # 輸入層
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 處理地圖輸入 - 完全參考 network2 的簡單 CNN
        x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(map_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        map_features = layers.Flatten()(x)
        map_features = layers.Dense(256, activation='relu')(map_features)
        
        # 處理前沿點 - 參考 network2
        f = layers.Dense(64, activation='relu')(frontier_input)
        f = layers.TimeDistributed(layers.Dense(32, activation='relu'))(f)
        f = layers.GlobalAveragePooling1D()(f)
        frontiers_features = layers.Dense(128, activation='relu')(f)
        
        # 處理機器人狀態 - 參考 network2
        robot1_state = layers.Concatenate()([robot1_pos, robot1_target])
        robot2_state = layers.Concatenate()([robot2_pos, robot2_target])
        robot1_feat = layers.Dense(64, activation='relu')(robot1_state)
        robot2_feat = layers.Dense(64, activation='relu')(robot2_state)
        robot_features = layers.Concatenate()([robot1_feat, robot2_feat])
        robot_features = layers.Dense(128, activation='relu')(robot_features)
        
        # 合併所有特徵 - 參考 network2 的直接連接
        combined = layers.Concatenate()([map_features, frontiers_features, robot_features])
        combined = layers.Dense(512, activation='relu')(combined)
        combined = layers.Dropout(0.1)(combined)
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.Dropout(0.1)(combined)
        combined = layers.Dense(128, activation='relu')(combined)
        
        # 分成兩支路 - 參考 network2
        robot1_branch = layers.Dense(128, activation='relu')(combined)
        robot2_branch = layers.Dense(128, activation='relu')(combined)
        
        # 輸出層
        robot1_policy = layers.Dense(self.max_frontiers, activation='softmax', name='robot1_policy')(robot1_branch)
        robot2_policy = layers.Dense(self.max_frontiers, activation='softmax', name='robot2_policy')(robot2_branch)
        
        # 創建模型
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
        
        # 設置優化器
        model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        return model
        
    def _build_critic(self):
        """構建 Critic 網絡 - 完全參考 network2 的簡化架構"""
        # 輸入層 - 與Actor共享
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 處理地圖輸入 - 完全參考 network2 的簡單 CNN
        x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(map_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        map_features = layers.Flatten()(x)
        map_features = layers.Dense(256, activation='relu')(map_features)
        
        # 處理前沿點 - 參考 network2
        f = layers.Dense(64, activation='relu')(frontier_input)
        f = layers.TimeDistributed(layers.Dense(32, activation='relu'))(f)
        f = layers.GlobalAveragePooling1D()(f)
        frontiers_features = layers.Dense(128, activation='relu')(f)
        
        # 處理機器人狀態 - 參考 network2
        robot1_state = layers.Concatenate()([robot1_pos, robot1_target])
        robot2_state = layers.Concatenate()([robot2_pos, robot2_target])
        robot1_feat = layers.Dense(64, activation='relu')(robot1_state)
        robot2_feat = layers.Dense(64, activation='relu')(robot2_state)
        robot_features = layers.Concatenate()([robot1_feat, robot2_feat])
        robot_features = layers.Dense(128, activation='relu')(robot_features)
        
        # 合併所有特徵 - 參考 network2 的直接連接
        combined = layers.Concatenate()([map_features, frontiers_features, robot_features])
        combined = layers.Dense(512, activation='relu')(combined)
        combined = layers.Dropout(0.1)(combined)
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.Dropout(0.1)(combined)
        combined = layers.Dense(128, activation='relu')(combined)
        
        # 分成兩支路 - 參考 network2
        robot1_branch = layers.Dense(128, activation='relu')(combined)
        robot2_branch = layers.Dense(128, activation='relu')(combined)
        
        # 輸出層 - 價值函數
        robot1_value = layers.Dense(1, name='robot1_value')(robot1_branch)
        robot2_value = layers.Dense(1, name='robot2_value')(robot2_branch)
        
        # 創建模型
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
        
        # 設置優化器和編譯
        model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=model.optimizer, loss='mse')
        
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
        """預測狀態價值"""
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
        """訓練 Actor 網絡"""
        with tf.GradientTape() as tape:
            # 取得策略預測
            policy_dict = self.actor({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            # 計算 Robot1 的策略損失
            robot1_actions = actions['robot1']
            robot1_advantages = advantages['robot1']
            robot1_actions_one_hot = tf.one_hot(robot1_actions, self.max_frontiers)
            
            # 添加數值穩定性處理
            robot1_policy = tf.clip_by_value(policy_dict['robot1_policy'], 1e-8, 1.0)
            robot1_policy = robot1_policy / tf.reduce_sum(robot1_policy, axis=1, keepdims=True)
            
            robot1_probs = tf.reduce_sum(robot1_policy * robot1_actions_one_hot, axis=1)
            robot1_log_probs = tf.math.log(robot1_probs)
            robot1_loss = -tf.reduce_mean(robot1_log_probs * robot1_advantages)
            
            # 計算 Robot2 的策略損失
            robot2_actions = actions['robot2']
            robot2_advantages = advantages['robot2']
            robot2_actions_one_hot = tf.one_hot(robot2_actions, self.max_frontiers)
            
            # 添加數值穩定性處理
            robot2_policy = tf.clip_by_value(policy_dict['robot2_policy'], 1e-8, 1.0)
            robot2_policy = robot2_policy / tf.reduce_sum(robot2_policy, axis=1, keepdims=True)
            
            robot2_probs = tf.reduce_sum(robot2_policy * robot2_actions_one_hot, axis=1)
            robot2_log_probs = tf.math.log(robot2_probs)
            robot2_loss = -tf.reduce_mean(robot2_log_probs * robot2_advantages)
            
            # 添加熵正則化
            entropy_coef = 0.005
            robot1_entropy = -tf.reduce_mean(tf.reduce_sum(
                robot1_policy * tf.math.log(robot1_policy), 
                axis=1))
            robot2_entropy = -tf.reduce_mean(tf.reduce_sum(
                robot2_policy * tf.math.log(robot2_policy), 
                axis=1))
            entropy_reward = entropy_coef * (robot1_entropy + robot2_entropy)
            
            # 計算協調損失 - 鼓勵不同的策略
            coordination_coef = 0.2
            similarity = tf.reduce_mean(tf.reduce_sum(
                tf.sqrt(robot1_policy * robot2_policy), axis=1))
            coordination_loss = coordination_coef * similarity
            
            # 添加L2正則化以防止權重爆炸
            l2_reg = 0.001
            l2_loss = tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.actor.trainable_weights if 'kernel' in w.name]) * l2_reg
            
            # 總損失
            policy_loss = robot1_loss + robot2_loss
            total_loss = policy_loss - entropy_reward + coordination_loss + l2_loss
        
        # 計算梯度並更新
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        
        # 梯度裁剪 - 使用更嚴格的值
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        
        # 檢查梯度是否包含NaN
        if tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None]):
            print("警告: Actor訓練中檢測到NaN梯度")
            # 返回一個固定的損失值，不更新參數
            return 0.0
        
        # 檢查梯度範數，如果過大則跳過更新
        if tf.math.is_nan(grad_norm) or grad_norm > 10.0:
            print(f"警告: 檢測到較大的梯度範數: {grad_norm}, 跳過更新")
            return float(total_loss)
        
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        
        return float(total_loss)
    
    def train_critic(self, states, frontiers, robot1_pos, robot2_pos, 
                    robot1_target, robot2_target, returns):
        """訓練 Critic 網絡"""
        with tf.GradientTape() as tape:
            # 取得價值預測
            value_dict = self.critic({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            # 計算價值損失
            robot1_returns = returns['robot1']
            robot2_returns = returns['robot2']
            
            robot1_value_loss = tf.keras.losses.Huber()(
                robot1_returns, value_dict['robot1_value'])
            robot2_value_loss = tf.keras.losses.Huber()(
                robot2_returns, value_dict['robot2_value'])
            
            # 總損失
            value_loss = robot1_value_loss + robot2_value_loss
        
        # 計算梯度並更新
        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        # 梯度裁剪
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        
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
        """保存模型 - 僅使用 .h5 格式"""
        print("\n開始保存模型...")
        
        # 創建目錄（如果不存在）
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        try:
            # 1. 保存完整模型 (.h5 格式)
            actor_path = filepath + '_actor.h5'
            critic_path = filepath + '_critic.h5'
            
            print(f"保存 Actor 模型到: {actor_path}")
            self.actor.save(actor_path, save_format='h5')
            
            print(f"保存 Critic 模型到: {critic_path}")
            self.critic.save(critic_path, save_format='h5')
            
            # 2. 保存配置 (小型 JSON 檔案)
            config = {
                'input_shape': self.input_shape,
                'max_frontiers': self.max_frontiers,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'dff': self.dff,
                'dropout_rate': self.dropout_rate,
                'actor_params': self.actor.count_params(),
                'critic_params': self.critic.count_params(),
                'save_time': str(np.datetime64('now'))
            }
            
            config_path = filepath + '_config.json'
            print(f"保存配置到: {config_path}")
            
            with open(config_path, 'w') as f:
                # 將元組轉換為列表以便 JSON 序列化
                config['input_shape'] = list(config['input_shape'])
                json.dump(config, f, indent=4)
            
            # 3. 驗證保存的檔案
            actor_size = os.path.getsize(actor_path) / (1024 * 1024)
            critic_size = os.path.getsize(critic_path) / (1024 * 1024)
            total_size = actor_size + critic_size
            
            print("\n保存的檔案大小:")
            print(f"Actor 模型: {actor_size:.2f} MB")
            print(f"Critic 模型: {critic_size:.2f} MB")
            print(f"總計: {total_size:.2f} MB")
            
            # 4. 驗證檔案大小是否合理
            expected_size = (self.actor.count_params() + self.critic.count_params()) * 4 / (1024 * 1024)
            print(f"預期大小 (基於參數數量): {expected_size:.2f} MB")
            
            if total_size < expected_size * 0.5:
                print("\n警告: 保存檔案大小顯著小於預期!")
            else:
                print("\n保存成功: 檔案大小合理")
            
            return True
        
        except Exception as e:
            print(f"保存模型時出錯: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def load(self, filepath):
        """載入模型 - 僅使用 .h5 格式"""
        print("\n開始載入模型...")
        
        actor_path = filepath + '_actor.h5'
        critic_path = filepath + '_critic.h5'
        config_path = filepath + '_config.json'
        
        if not os.path.exists(actor_path) or not os.path.exists(critic_path):
            print(f"錯誤: 找不到模型檔案: {actor_path} 或 {critic_path}")
            return False
        
        try:
            # 1. 載入配置
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 更新模型配置
                self.input_shape = tuple(config['input_shape'])
                self.max_frontiers = config['max_frontiers']
                self.d_model = config.get('d_model', 128)
                self.num_heads = config.get('num_heads', 4)
                self.dff = config.get('dff', 256)
                self.dropout_rate = config.get('dropout_rate', 0.1)
                
                print(f"已載入配置，預期模型參數: Actor {config.get('actor_params', 'N/A'):,}, " + 
                    f"Critic {config.get('critic_params', 'N/A'):,}")
                
                if 'save_time' in config:
                    print(f"模型保存時間: {config['save_time']}")
            
            # 2. 載入模型
            print(f"載入 Actor 模型: {actor_path}")
            self.actor = tf.keras.models.load_model(actor_path)
            
            print(f"載入 Critic 模型: {critic_path}")
            self.critic = tf.keras.models.load_model(critic_path)
            
            # 3. 確保優化器設置
            if not hasattr(self.actor, 'optimizer') or self.actor.optimizer is None:
                self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            
            if not hasattr(self.critic, 'optimizer') or self.critic.optimizer is None:
                self.critic.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
                self.critic.compile(optimizer=self.critic.optimizer, loss='mse')
            
            # 4. 驗證載入結果
            actor_params = self.actor.count_params()
            critic_params = self.critic.count_params()
            total_params = actor_params + critic_params
            
            print(f"\n載入完成:")
            print(f"Actor 參數: {actor_params:,}")
            print(f"Critic 參數: {critic_params:,}")
            print(f"總參數: {total_params:,}")
            print(f"理論檔案大小: {total_params * 4 / (1024 * 1024):.2f} MB")
            
            if config.get('actor_params') and actor_params != config['actor_params']:
                print(f"警告: Actor 參數數量與配置不符! 預期: {config['actor_params']:,}, 實際: {actor_params:,}")
                
            if config.get('critic_params') and critic_params != config['critic_params']:
                print(f"警告: Critic 參數數量與配置不符! 預期: {config['critic_params']:,}, 實際: {critic_params:,}")
            
            print("\n模型載入成功")
            return True
        
        except Exception as e:
            print(f"載入模型時出錯: {str(e)}")
            import traceback
            traceback.print_exc()
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