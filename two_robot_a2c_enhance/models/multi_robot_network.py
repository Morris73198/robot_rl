import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers
import json
import os

# 在模塊開頭設置GPU記憶體增長
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"GPU memory growth enabled for {physical_devices[0]}")
except Exception as e:
    print(f"GPU setup warning: {e}")

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
    """空間注意力層 - 穩定版本"""
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

class MultiHeadAttention(layers.Layer):
    """多頭注意力層 - 記憶體優化版"""
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
    def build(self, input_shape):
        self.wq = layers.Dense(self.d_model, name='query_dense')
        self.wk = layers.Dense(self.d_model, name='key_dense')
        self.wv = layers.Dense(self.d_model, name='value_dense')
        self.dense = layers.Dense(self.d_model, name='output_dense')
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = LayerNormalization()
        super().build(input_shape)
        
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
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
            
        batch_size = tf.shape(q)[0]
        
        # 保存原始輸入用於殘差連接
        original_input = inputs if not isinstance(inputs, list) else inputs[0]
        
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
        
        # 修復殘差連接 - 確保維度匹配
        if original_input.shape[-1] == output.shape[-1]:
            output = self.layer_norm(output + original_input)
        else:
            # 如果維度不匹配，只使用LayerNorm而不加殘差
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

class CrossRobotAttention(layers.Layer):
    """跨機器人注意力層"""
    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.projection1 = layers.Dense(self.d_model, name='robot1_projection')
        self.projection2 = layers.Dense(self.d_model, name='robot2_projection')
        self.attention = layers.Attention(use_scale=True, dropout=self.dropout_rate)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate
        })
        return config
        
    def call(self, inputs):
        robot1_features, robot2_features = inputs
        
        robot1_proj = self.projection1(robot1_features)
        robot2_proj = self.projection2(robot2_features)
        
        robot1_expanded = tf.expand_dims(robot1_proj, axis=1)
        robot2_expanded = tf.expand_dims(robot2_proj, axis=1)
        
        r1_to_r2 = self.attention([robot1_expanded, robot2_expanded, robot2_expanded])
        r1_enhanced = self.layer_norm1(robot1_expanded + r1_to_r2)
        
        r2_to_r1 = self.attention([robot2_expanded, robot1_expanded, robot1_expanded])
        r2_enhanced = self.layer_norm2(robot2_expanded + r2_to_r1)
        
        return tf.squeeze(r1_enhanced, axis=1), tf.squeeze(r2_enhanced, axis=1)

class TemporalAttention(layers.Layer):
    """時間注意力層 - 簡化版本"""
    def __init__(self, d_model, memory_length=8, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.memory_length = memory_length
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # 使用可訓練的記憶體
        self.memory = self.add_weight(
            name="temporal_memory",
            shape=[1, self.memory_length, self.d_model],
            initializer='zeros',
            trainable=True
        )
        
        self.attention = layers.Attention(use_scale=True, dropout=self.dropout_rate)
        self.layer_norm = LayerNormalization()
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        if len(inputs.shape) > 2:
            inputs = tf.reduce_mean(inputs, axis=list(range(1, len(tf.shape(inputs))-1)))
        
        query = tf.expand_dims(inputs, axis=1)
        batch_size = tf.shape(query)[0]
        
        key_value = tf.repeat(self.memory, batch_size, axis=0)
        
        attended = self.attention([query, key_value, key_value])
        output = self.layer_norm(query + attended)
        
        return tf.squeeze(output, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'memory_length': self.memory_length,
            'dropout_rate': self.dropout_rate
        })
        return config

class AdaptiveAttentionFusion(layers.Layer):
    """自適應注意力融合層"""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        self.weight_net = layers.Dense(4, activation='softmax', name='fusion_weights')
        self.layer_norm = LayerNormalization()
        super().build(input_shape)
        
    def call(self, inputs):
        original, spatial_attn, cross_robot_attn, temporal_attn = inputs
        
        # 確保所有輸入都是2D
        if len(spatial_attn.shape) > 2:
            spatial_attn = tf.reduce_mean(spatial_attn, axis=list(range(1, len(tf.shape(spatial_attn))-1)))
        if len(cross_robot_attn.shape) > 2:
            cross_robot_attn = tf.reduce_mean(cross_robot_attn, axis=list(range(1, len(tf.shape(cross_robot_attn))-1)))
        if len(temporal_attn.shape) > 2:
            temporal_attn = tf.reduce_mean(temporal_attn, axis=list(range(1, len(tf.shape(temporal_attn))-1)))
        
        # 生成融合權重
        features_concat = tf.concat([
            spatial_attn, cross_robot_attn, temporal_attn, original
        ], axis=-1)
        
        weights = self.weight_net(features_concat)
        w1, w2, w3, w4 = tf.split(weights, 4, axis=-1)
        
        # 加權融合
        fused = (w1 * spatial_attn + w2 * cross_robot_attn + 
                w3 * temporal_attn + w4 * original)
        
        output = self.layer_norm(original + fused)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config

class MultiRobotACModel:
    """多機器人 Actor-Critic 模型 - 穩定版本"""
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        
        # 保持足夠大的模型以滿足>1MB要求
        self.d_model = 256  # 恢復較大的維度
        self.num_heads = 8
        self.dff = 512
        self.dropout_rate = 0.15  # 適中的dropout率
        
        # 創建 Actor 和 Critic 網絡
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # 打印模型信息
        self._print_model_info()
        
    def _print_model_info(self):
        """打印模型信息"""
        actor_params = self.actor.count_params()
        critic_params = self.critic.count_params()
        total_params = actor_params + critic_params
        
        print("\n===== 模型參數統計 =====")
        print(f"Actor 參數: {actor_params:,}")
        print(f"Critic 參數: {critic_params:,}")
        print(f"總參數數量: {total_params:,}")
        print(f"預期權重檔案大小: {total_params * 4 / (1024*1024):.2f} MB")
        print("=======================\n")
        
    def _build_perception_module(self, inputs):
        """構建感知模塊"""
        # 適度降低解析度
        x = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
        
        # 多尺度特徵提取
        conv_configs = [
            {'filters': 32, 'kernel_size': 3, 'strides': 2},
            {'filters': 32, 'kernel_size': 5, 'strides': 2},
            {'filters': 32, 'kernel_size': 7, 'strides': 2}
        ]
        
        features = []
        for i, config in enumerate(conv_configs):
            branch = layers.Conv2D(
                filters=config['filters'],
                kernel_size=config['kernel_size'],
                strides=config['strides'],
                padding='same',
                kernel_regularizer=regularizers.l2(0.01),
                name=f'conv_branch_{i}'
            )(x)
            branch = layers.BatchNormalization(name=f'bn_branch_{i}')(branch)
            branch = layers.Activation('relu', name=f'relu_branch_{i}')(branch)
            branch = SpatialAttention(name=f'spatial_attn_{i}')(branch)
            features.append(branch)
            
        # 合併特征
        x = layers.Add(name='feature_fusion')(features)
        x = layers.Conv2D(64, 1, padding='same', name='feature_compress')(x)
        x = layers.BatchNormalization(name='final_bn')(x)
        x = layers.Activation('relu', name='final_relu')(x)
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        return x
        
    def _build_shared_features(self, map_features, frontier_input, robot1_state, robot2_state):
        """構建共享特徵提取層"""
        # Frontier特徵處理 - 使用較大維度以確保模型大小
        frontier_features = layers.Dense(128, activation='relu', name='frontier_dense1')(frontier_input)
        frontier_features = layers.Dropout(self.dropout_rate, name='frontier_dropout1')(frontier_features)
        frontier_features = layers.Dense(self.d_model, activation='relu', name='frontier_dense2')(frontier_features)
        frontier_features = layers.Dropout(self.dropout_rate, name='frontier_dropout2')(frontier_features)
        
        # 多頭注意力處理frontier
        frontier_attention = MultiHeadAttention(
            d_model=self.d_model, 
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            name='frontier_attention'
        )(frontier_features)
        
        # 前饋網路
        ff_output = layers.Dense(self.dff, activation='relu', name='frontier_ff1')(frontier_attention)
        ff_output = layers.Dropout(self.dropout_rate, name='frontier_ff_dropout')(ff_output)
        ff_output = layers.Dense(self.d_model, name='frontier_ff2')(ff_output)
        
        # 殘差連接和層正規化
        frontier_features = layers.Add(name='frontier_residual')([frontier_attention, ff_output])
        frontier_features = LayerNormalization(name='frontier_norm')(frontier_features)
        
        # 處理機器人狀態 - 使用較大維度
        robot1_feat = layers.Dense(128, activation='relu', name='robot1_dense1')(robot1_state)
        robot1_feat = layers.Dropout(self.dropout_rate, name='robot1_dropout1')(robot1_feat)
        robot1_feat = layers.Dense(self.d_model, activation='relu', name='robot1_dense2')(robot1_feat)
        robot1_feat = layers.Dropout(self.dropout_rate, name='robot1_dropout2')(robot1_feat)
        
        robot2_feat = layers.Dense(128, activation='relu', name='robot2_dense1')(robot2_state)
        robot2_feat = layers.Dropout(self.dropout_rate, name='robot2_dropout1')(robot2_feat)
        robot2_feat = layers.Dense(self.d_model, activation='relu', name='robot2_dense2')(robot2_feat)
        robot2_feat = layers.Dropout(self.dropout_rate, name='robot2_dropout2')(robot2_feat)
        
        # 跨機器人注意力
        robot1_enhanced, robot2_enhanced = CrossRobotAttention(
            d_model=self.d_model,
            dropout_rate=self.dropout_rate,
            name='cross_robot_attention'
        )([robot1_feat, robot2_feat])
        
        # 融合機器人特徵
        robot_features = layers.Concatenate(name='robot_concat')([robot1_enhanced, robot2_enhanced])
        robot_features = layers.Dense(self.d_model, activation='relu', name='robot_fusion')(robot_features)
        robot_features = layers.Dropout(self.dropout_rate, name='robot_fusion_dropout')(robot_features)
        
        # 處理時間維度特徵
        temporal_frontier_feature = layers.GlobalAveragePooling1D(name='temporal_pool')(frontier_features)
        temporal_frontier_feature = layers.Dense(self.d_model, activation='relu', name='temporal_dense')(temporal_frontier_feature)
        temporal_frontier_feature = layers.Dropout(self.dropout_rate, name='temporal_dropout')(temporal_frontier_feature)
        
        # 時間注意力
        temporal_features = TemporalAttention(
            d_model=self.d_model,
            memory_length=8,
            dropout_rate=self.dropout_rate,
            name='temporal_attention'
        )(temporal_frontier_feature)
        
        # 處理地圖特徵
        map_features_enhanced = layers.Dense(self.d_model, activation='relu', name='map_dense1')(map_features)
        map_features_enhanced = layers.Dropout(self.dropout_rate, name='map_dropout1')(map_features_enhanced)
        map_features_enhanced = layers.Dense(self.d_model, activation='relu', name='map_dense2')(map_features_enhanced)
        map_features_enhanced = layers.Dropout(self.dropout_rate, name='map_dropout2')(map_features_enhanced)
        
        frontier_global = layers.GlobalAveragePooling1D(name='frontier_global_pool')(frontier_features)
        frontier_global = layers.Dense(self.d_model, activation='relu', name='frontier_global_dense')(frontier_global)
        frontier_global = layers.Dropout(self.dropout_rate, name='frontier_global_dropout')(frontier_global)
        
        # 自適應注意力融合
        fused_features = AdaptiveAttentionFusion(
            d_model=self.d_model,
            name='adaptive_fusion'
        )([
            map_features_enhanced,
            frontier_global,
            robot_features,
            temporal_features
        ])
        
        # 最終特徵處理 - 添加更多層以增加參數量
        combined_features = layers.Concatenate(name='final_concat')([
            map_features_enhanced,
            fused_features,
            robot_features
        ])
        
        x = layers.Dense(self.dff, activation='relu', name='final_dense1')(combined_features)
        x = layers.Dropout(self.dropout_rate, name='final_dropout1')(x)
        x = layers.Dense(self.d_model, activation='relu', name='final_dense2')(x)
        x = layers.Dropout(self.dropout_rate, name='final_dropout2')(x)
        x = layers.Dense(self.d_model//2, activation='relu', name='final_dense3')(x)
        x = layers.Dropout(self.dropout_rate, name='final_dropout3')(x)
        
        return x
        
    def _build_policy_head(self, features, name_prefix):
        """構建策略輸出頭"""
        x = layers.Dense(256, activation='relu', name=f'{name_prefix}_policy_dense1')(features)
        x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_policy_dropout1')(x)
        x = layers.Dense(128, activation='relu', name=f'{name_prefix}_policy_dense2')(x)
        x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_policy_dropout2')(x)
        x = layers.Dense(64, activation='relu', name=f'{name_prefix}_policy_dense3')(x)
        x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_policy_dropout3')(x)
        policy = layers.Dense(
            self.max_frontiers,
            activation='softmax',
            name=name_prefix
        )(x)
        return policy

    def _build_value_head(self, features, name_prefix):
        """構建價值輸出頭"""
        x = layers.Dense(256, activation='relu', name=f'{name_prefix}_value_dense1')(features)
        x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_value_dropout1')(x)
        x = layers.Dense(128, activation='relu', name=f'{name_prefix}_value_dense2')(x)
        x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_value_dropout2')(x)
        x = layers.Dense(64, activation='relu', name=f'{name_prefix}_value_dense3')(x)
        x = layers.Dropout(self.dropout_rate, name=f'{name_prefix}_value_dropout3')(x)
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
        robot1_state = layers.Concatenate(name='robot1_state_concat')([robot1_pos, robot1_target])
        robot2_state = layers.Concatenate(name='robot2_state_concat')([robot2_pos, robot2_target])
        
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
        
        # 使用較小的學習率提高穩定性
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
        model.compile(optimizer=optimizer)
        
        return model
        
    def _build_critic(self):
        """構建 Critic 網絡"""
        # 與Actor相同的結構，但輸出為價值函數
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 特征提取
        map_features = self._build_perception_module(map_input)
        
        # 機器人狀態編碼
        robot1_state = layers.Concatenate(name='robot1_state_concat')([robot1_pos, robot1_target])
        robot2_state = layers.Concatenate(name='robot2_state_concat')([robot2_pos, robot2_target])
        
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
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
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
            
        # 梯度裁剪
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.actor.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables))
            
        return total_loss
    
    def train_critic(self, states, frontiers, robot1_pos, robot2_pos, 
                    robot1_target, robot2_target, returns):
        """訓練 Critic 網路"""
        with tf.GradientTape() as tape:
            values = self.critic({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            robot1_value_loss = tf.keras.losses.Huber()(
                returns['robot1'], values['robot1_value'])
            robot2_value_loss = tf.keras.losses.Huber()(
                returns['robot2'], values['robot2_value'])
            
            value_loss = robot1_value_loss + robot2_value_loss
            
        # 梯度裁剪
        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.critic.optimizer.apply_gradients(
            zip(grads, self.critic.trainable_variables))
            
        return value_loss
    
    def _compute_policy_loss(self, policy, actions, advantages):
        """計算策略損失"""
        actions_one_hot = tf.one_hot(actions, self.max_frontiers)
        log_prob = tf.math.log(tf.reduce_sum(policy * actions_one_hot, axis=1) + 1e-10)
        policy_loss = -tf.reduce_mean(log_prob * advantages)
        
        # 熵正則化
        entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1)
        entropy_bonus = 0.01 * tf.reduce_mean(entropy)
        
        return policy_loss - entropy_bonus
    
    def save(self, filepath):
        """保存模型為.h5格式"""
        print("保存模型為.h5格式...")
        
        try:
            # 保存 Actor 為 .h5 格式
            actor_path = filepath + '_actor.h5'
            print(f"保存 actor 模型到: {actor_path}")
            self.actor.save(actor_path, save_format='h5')
            
            # 保存 Critic 為 .h5 格式
            critic_path = filepath + '_critic.h5'
            print(f"保存 critic 模型到: {critic_path}")
            self.critic.save(critic_path, save_format='h5')
            
            # 保存配置
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
                
            print("模型(.h5格式)保存成功")
            return True
            
        except Exception as e:
            print(f"保存模型時出錯: {str(e)}")
            return False
    
    def load(self, filepath):
        """載入.h5格式的模型"""
        print("載入.h5格式模型...")
        
        try:
            # 載入配置
            config_path = filepath + '_config.json'
            if not os.path.exists(config_path):
                print(f"配置文件不存在: {config_path}")
                return False
                
            with open(config_path, 'r') as f:
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
                'SpatialAttention': SpatialAttention,
                'MultiHeadAttention': MultiHeadAttention,
                'CrossRobotAttention': CrossRobotAttention,
                'TemporalAttention': TemporalAttention,
                'AdaptiveAttentionFusion': AdaptiveAttentionFusion
            }
            
            # 載入模型
            actor_path = filepath + '_actor.h5'
            critic_path = filepath + '_critic.h5'
            
            if not os.path.exists(actor_path):
                print(f"Actor模型文件不存在: {actor_path}")
                return False
            if not os.path.exists(critic_path):
                print(f"Critic模型文件不存在: {critic_path}")
                return False
                
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
            
            print("模型(.h5格式)載入成功")
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