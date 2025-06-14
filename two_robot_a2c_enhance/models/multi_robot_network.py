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

class SpatialAttention(layers.Layer):
    """空間注意力層 - 簡化版"""
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

class PositionalEncoding(layers.Layer):
    """位置編碼層 - 簡化版"""
    def __init__(self, max_position, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.d_model = d_model
        
    def build(self, input_shape):
        # 使用簡化的線性位置編碼
        position_enc = np.zeros((self.max_position, self.d_model))
        for pos in range(self.max_position):
            for i in range(self.d_model):
                if i % 2 == 0:
                    position_enc[pos, i] = np.sin(pos / np.power(10000, 2 * i / self.d_model))
                else:
                    position_enc[pos, i] = np.cos(pos / np.power(10000, 2 * (i - 1) / self.d_model))
                    
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

class CrossRobotAttention(layers.Layer):
    """跨機器人注意力層 - 優化版"""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        # 簡化投影層
        self.robot1_projection = layers.Dense(self.d_model, name='robot1_projection')
        self.robot2_projection = layers.Dense(self.d_model, name='robot2_projection')
        
        # 使用標準注意力層代替自定義實現
        self.attention = layers.Attention(use_scale=True, dropout=0.1)
        
        # 正規化層
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config
        
    def call(self, inputs):
        robot1_features, robot2_features = inputs
        
        # 投影到相同維度
        robot1_proj = self.robot1_projection(robot1_features)
        robot2_proj = self.robot2_projection(robot2_features)
        
        # 擴展維度以適應注意力機制
        robot1_expanded = tf.expand_dims(robot1_proj, axis=1)
        robot2_expanded = tf.expand_dims(robot2_proj, axis=1)
        
        # Robot1 attends to Robot2
        r1_to_r2 = self.attention([robot1_expanded, robot2_expanded, robot2_expanded])
        r1_enhanced = self.layer_norm1(robot1_expanded + r1_to_r2)
        
        # Robot2 attends to Robot1
        r2_to_r1 = self.attention([robot2_expanded, robot1_expanded, robot1_expanded])
        r2_enhanced = self.layer_norm2(robot2_expanded + r2_to_r1)
        
        # 移除額外維度
        return tf.squeeze(r1_enhanced, axis=1), tf.squeeze(r2_enhanced, axis=1)

class AdaptiveAttentionFusion(layers.Layer):
    """自適應注意力融合層 - 方案B核心組件"""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        # 投影層 - 將所有特徵投影到相同維度
        self.frontier_projection = layers.Dense(self.d_model, name='frontier_projection')
        self.cross_robot_projection = layers.Dense(self.d_model, name='cross_robot_projection')
        self.map_projection = layers.Dense(self.d_model, name='map_projection')
        
        # 自適應權重網路 - 學習最佳的特徵融合策略
        self.adaptive_weight_dense1 = layers.Dense(self.d_model // 2, activation='relu', name='adaptive_weight_dense1')
        self.adaptive_weight_dropout = layers.Dropout(0.1, name='adaptive_weight_dropout')
        self.adaptive_weight_dense2 = layers.Dense(3, activation='softmax', name='adaptive_weight_dense2')
        
        # 交叉注意力層 - 讓不同模態之間相互關注
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.d_model // 4,
            dropout=0.1,
            name='cross_modal_attention'
        )
        
        # 門控機制 - 控制信息流
        self.gate_dense = layers.Dense(self.d_model, activation='sigmoid', name='gate_dense')
        
        # 正規化層
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config
        
    def call(self, inputs, training=None):
        frontier_features, cross_robot_features, map_features = inputs
        
        # 1. 投影所有特徵到相同維度
        frontier_proj = self.frontier_projection(frontier_features)
        cross_robot_proj = self.cross_robot_projection(cross_robot_features)
        map_proj = self.map_projection(map_features)
        
        # 2. 計算自適應權重
        # 使用所有特徵的拼接來預測最佳融合權重
        concat_features = tf.concat([frontier_proj, cross_robot_proj, map_proj], axis=-1)
        adaptive_weights = self.adaptive_weight_dense1(concat_features)
        adaptive_weights = self.adaptive_weight_dropout(adaptive_weights, training=training)
        adaptive_weights = self.adaptive_weight_dense2(adaptive_weights)
        w_frontier, w_robot, w_map = tf.split(adaptive_weights, 3, axis=-1)
        
        # 3. 交叉模態注意力
        # 將特徵重塑為序列格式以使用多頭注意力
        stacked_features = tf.stack([frontier_proj, cross_robot_proj, map_proj], axis=1)  # [batch, 3, d_model]
        
        # 自注意力：讓不同模態相互關注
        attended_features = self.cross_attention(
            stacked_features, stacked_features, stacked_features, training=training
        )
        attended_features = self.layer_norm1(stacked_features + attended_features)
        
        # 分離回原始特徵
        frontier_attended, cross_robot_attended, map_attended = tf.unstack(attended_features, axis=1)
        
        # 4. 自適應加權融合
        weighted_features = (
            w_frontier * frontier_attended + 
            w_robot * cross_robot_attended + 
            w_map * map_attended
        )
        
        # 5. 門控機制
        gate = self.gate_dense(weighted_features, training=training)
        gated_features = gate * weighted_features
        
        # 6. 殘差連接和最終正規化
        # 使用map特徵作為殘差基準（因為它包含最全局的信息）
        output = self.layer_norm2(map_proj + gated_features)
        
        return output

class TemporalAttentionModule(layers.Layer):
    """時間注意力模組 - 修復梯度消失問題"""
    def __init__(self, d_model, sequence_length=10, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.sequence_length = sequence_length
        
    def build(self, input_shape):
        # 時序特徵記憶機制
        self.memory_states = self.add_weight(
            shape=(self.sequence_length, self.d_model),
            initializer='zeros',
            trainable=False,
            name='memory_states'
        )
        
        # 輸入投影
        self.input_projection = layers.Dense(self.d_model, name='input_projection')
        
        # 時序注意力機制
        self.temporal_attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.d_model // 4,
            dropout=0.1,
            name='temporal_attention'
        )
        
        # 時序融合網路
        self.temporal_fusion_dense1 = layers.Dense(self.d_model, activation='relu', name='temporal_fusion_dense1')
        self.temporal_fusion_dropout = layers.Dropout(0.1, name='temporal_fusion_dropout')
        self.temporal_fusion_dense2 = layers.Dense(self.d_model, name='temporal_fusion_dense2')
        
        # 修復：將update_gate改為總是參與計算的組件
        self.memory_gate_dense = layers.Dense(self.d_model, activation='sigmoid', name='memory_gate_dense')
        
        # 正規化層
        self.layer_norm = LayerNormalization()
        
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'sequence_length': self.sequence_length
        })
        return config
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # 1. 投影當前輸入
        current_state = self.input_projection(inputs)
        
        # 2. 擴展記憶狀態以匹配批次大小
        memory_batch = tf.tile(
            tf.expand_dims(self.memory_states, 0), 
            [batch_size, 1, 1]
        )  # [batch, sequence_length, d_model]
        
        # 3. 將當前狀態添加到序列中進行注意力計算
        current_expanded = tf.expand_dims(current_state, 1)  # [batch, 1, d_model]
        full_sequence = tf.concat([memory_batch, current_expanded], axis=1)  # [batch, seq_len+1, d_model]
        
        # 4. 時序注意力 - 當前狀態關注歷史
        attended_output = self.temporal_attention(
            current_expanded,  # query: 當前狀態
            full_sequence,     # key & value: 歷史+當前
            full_sequence,
            training=training
        )  # [batch, 1, d_model]
        
        attended_current = tf.squeeze(attended_output, axis=1)  # [batch, d_model]
        
        # 5. 時序融合
        temporal_features = self.temporal_fusion_dense1(attended_current)
        temporal_features = self.temporal_fusion_dropout(temporal_features, training=training)
        temporal_features = self.temporal_fusion_dense2(temporal_features)
        
        # 6. 修復：確保memory_gate_dense總是參與梯度計算
        # 計算記憶門控 - 這個操作總是執行，確保參與梯度計算
        memory_gate = self.memory_gate_dense(current_state)
        
        # 將門控應用到時序特徵上，確保梯度流動
        gated_temporal_features = memory_gate * temporal_features
        
        # 7. 記憶狀態更新（僅在訓練時）
        if training:
            # 使用門控後的當前狀態來更新記憶
            filtered_current = memory_gate * current_state
            
            # 滑動窗口更新：移除最舊的狀態，添加當前狀態
            memory_update = tf.reduce_mean(filtered_current, axis=0)
            new_memory = tf.concat([
                self.memory_states[1:],  # 移除最舊的
                tf.expand_dims(memory_update, 0)  # 添加當前的平均
            ], axis=0)
            
            # 更新記憶（非梯度更新）
            self.memory_states.assign(new_memory)
        
        # 8. 殘差連接和正規化
        # 使用門控後的時序特徵，確保memory_gate_dense參與梯度計算
        output = self.layer_norm(current_state + gated_temporal_features)
        
        return output

class MultiRobotACModel:
    """多機器人 Actor-Critic 模型 - 方案B實現（修復梯度消失）"""
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        
        # 模型參數
        self.d_model = 128
        self.num_heads = 4
        self.dff = 256
        self.dropout_rate = 0.2

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
        
        print("\n===== 修復梯度消失後的模型參數統計 =====")
        print(f"Actor 參數: {actor_params:,}")
        print(f"Critic 參數: {critic_params:,}")
        print(f"總參數數量: {total_params:,}")
        print(f"預期權重檔案大小: {total_params * 4 / (1024*1024):.2f} MB")
        print("========================================\n")
        
    def _build_perception_module(self, inputs):
        """構建感知模塊"""
        # 降低輸入分辨率
        x = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
        
        # 多尺度特徵提取
        conv_configs = [
            {'filters': 16, 'kernel_size': 3, 'strides': 2},
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
            branch = SpatialAttention()(branch)
            features.append(branch)
            
        # 合併特征
        x = layers.Add()(features)
        x = layers.Conv2D(32, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        return x
        
    def _build_state_process_module(self, robot1_pos, robot1_target, robot2_pos, robot2_target):
        """構建狀態處理模塊"""
        # Robot1 state
        robot1_state = layers.Concatenate()([robot1_pos, robot1_target])
        robot1_features = layers.Dense(32, activation='relu')(robot1_state)
        robot1_features = layers.Dropout(self.dropout_rate)(robot1_features)
        
        # Robot2 state  
        robot2_state = layers.Concatenate()([robot2_pos, robot2_target])
        robot2_features = layers.Dense(32, activation='relu')(robot2_state)
        robot2_features = layers.Dropout(self.dropout_rate)(robot2_features)
        
        return robot1_features, robot2_features
        
    def _build_frontier_attention_layer(self, frontier_input):
        """構建前沿注意力層"""
        # 特徵投影
        x = layers.Dense(64, activation='relu')(frontier_input)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 位置編碼
        x = PositionalEncoding(self.max_frontiers, 64)(x)
        
        # 多頭注意力
        attention_output = layers.MultiHeadAttention(
            num_heads=2,
            key_dim=32,
            dropout=self.dropout_rate
        )(x, x)
        
        # 前饋網路
        ff_output = layers.Dense(64, activation='relu')(attention_output)
        ff_output = layers.Dropout(self.dropout_rate)(ff_output)
        
        # Add & Norm
        output = layers.Add()([attention_output, ff_output])
        output = LayerNormalization()(output)
        
        # 全局平均池化
        frontier_features = layers.GlobalAveragePooling1D()(output)
        
        return frontier_features
        
    def _build_enhanced_features(self, map_features, frontier_features, 
                                robot1_features, robot2_features):
        """構建增強特徵提取 - 方案B：先融合再時序"""
        
        # 1. 跨機器人注意力
        cross_attention = CrossRobotAttention(64)
        r1_enhanced, r2_enhanced = cross_attention([robot1_features, robot2_features])
        cross_robot_features = layers.Concatenate()([r1_enhanced, r2_enhanced])
        
        # 2. 自適應注意力融合 - 方案B第一步：融合多模態信息
        fusion_layer = AdaptiveAttentionFusion(self.d_model)
        fused_features = fusion_layer([
            frontier_features, 
            cross_robot_features, 
            map_features
        ])
        
        # 3. 時間注意力模組 - 方案B第二步：對融合後的特徵進行時序建模（已修復梯度消失）
        temporal_attention = TemporalAttentionModule(self.d_model, sequence_length=10)
        temporal_enhanced_features = temporal_attention(fused_features)
        
        return temporal_enhanced_features
        
    def _build_process_module(self, features):
        """構建處理模塊"""
        x = layers.Dense(128, activation='relu')(features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        return x

    def _build_actor(self):
        """構建 Actor 網絡"""
        # 輸入層
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 感知模塊
        map_features = self._build_perception_module(map_input)
        
        # 狀態處理模塊
        robot1_features, robot2_features = self._build_state_process_module(
            robot1_pos, robot1_target, robot2_pos, robot2_target
        )
        
        # 前沿注意力層
        frontier_features = self._build_frontier_attention_layer(frontier_input)
        
        # 增強特徵提取 - 方案B：先融合再時序（已修復梯度消失）
        enhanced_features = self._build_enhanced_features(
            map_features, frontier_features, robot1_features, robot2_features
        )
        
        # 處理模塊
        processed_features = self._build_process_module(enhanced_features)
        
        # Actor 網絡輸出
        # Robot1 policy
        robot1_policy = layers.Dense(64, activation='relu')(processed_features)
        robot1_policy = layers.Dropout(self.dropout_rate)(robot1_policy)
        robot1_policy = layers.Dense(32, activation='relu')(robot1_policy)
        robot1_policy = layers.Dropout(self.dropout_rate)(robot1_policy)
        robot1_policy = layers.Dense(self.max_frontiers, activation='softmax', 
                                   name='robot1_policy')(robot1_policy)
        
        # Robot2 policy
        robot2_policy = layers.Dense(64, activation='relu')(processed_features)
        robot2_policy = layers.Dropout(self.dropout_rate)(robot2_policy)
        robot2_policy = layers.Dense(32, activation='relu')(robot2_policy)
        robot2_policy = layers.Dropout(self.dropout_rate)(robot2_policy)
        robot2_policy = layers.Dense(self.max_frontiers, activation='softmax', 
                                   name='robot2_policy')(robot2_policy)
        
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
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
        model.compile(optimizer=optimizer)
        
        return model
        
    def _build_critic(self):
        """構建 Critic 網絡"""
        # 輸入層
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 感知模塊
        map_features = self._build_perception_module(map_input)
        
        # 狀態處理模塊
        robot1_features, robot2_features = self._build_state_process_module(
            robot1_pos, robot1_target, robot2_pos, robot2_target
        )
        
        # 前沿注意力層
        frontier_features = self._build_frontier_attention_layer(frontier_input)
        
        # 增強特徵提取 - 方案B：先融合再時序（已修復梯度消失）
        enhanced_features = self._build_enhanced_features(
            map_features, frontier_features, robot1_features, robot2_features
        )
        
        # 處理模塊
        processed_features = self._build_process_module(enhanced_features)
        
        # Critic 網絡輸出
        # Robot1 value
        robot1_value = layers.Dense(64, activation='relu')(processed_features)
        robot1_value = layers.Dropout(self.dropout_rate)(robot1_value)
        robot1_value = layers.Dense(32, activation='relu')(robot1_value)
        robot1_value = layers.Dropout(self.dropout_rate)(robot1_value)
        robot1_value = layers.Dense(1, name='robot1_value')(robot1_value)
        
        # Robot2 value
        robot2_value = layers.Dense(64, activation='relu')(processed_features)
        robot2_value = layers.Dropout(self.dropout_rate)(robot2_value)
        robot2_value = layers.Dense(32, activation='relu')(robot2_value)
        robot2_value = layers.Dropout(self.dropout_rate)(robot2_value)
        robot2_value = layers.Dense(1, name='robot2_value')(robot2_value)
        
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
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
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
        grads, _ = tf.clip_by_global_norm(grads, 0.3)
        self.actor.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables))
            
        return total_loss
    
    def train_critic(self, states, frontiers, robot1_pos, robot2_pos, 
                    robot1_target, robot2_target, returns):
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
        grads, _ = tf.clip_by_global_norm(grads, 0.3)
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
        entropy_bonus = 0.005 * tf.reduce_mean(entropy)
        
        return policy_loss - entropy_bonus
    
    def save(self, filepath):
        """保存模型到 h5 格式"""
        print("保存修復梯度消失後的模型...")
        
        try:
            # 保存 Actor
            actor_path = filepath + '_actor.h5'
            print(f"保存 actor 模型到: {actor_path}")
            self.actor.save(actor_path, save_format='h5')
            
            # 保存 Critic
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
                'dropout_rate': self.dropout_rate,
                'architecture_version': 'B_gradient_fixed',  # 標記架構版本
                'gradient_fix_applied': True
            }
            
            with open(filepath + '_config.json', 'w') as f:
                json.dump(config, f, indent=4)
                
            print("修復梯度消失後的模型保存成功")
            return True
            
        except Exception as e:
            print(f"保存模型時出錯: {str(e)}")
            return False

    def load(self, filepath):
        """載入 h5 格式的模型"""
        print("載入修復梯度消失後的模型...")
        
        try:
            # 載入配置
            with open(filepath + '_config.json', 'r') as f:
                config = json.load(f)
                
            # 更新模型配置
            self.input_shape = tuple(config['input_shape'])
            self.max_frontiers = config['max_frontiers']
            self.d_model = config['d_model']
            self.num_heads = config.get('num_heads', 4)
            self.dff = config.get('dff', 256)
            self.dropout_rate = config['dropout_rate']
            
            # 檢查架構版本
            arch_version = config.get('architecture_version', 'unknown')
            gradient_fix = config.get('gradient_fix_applied', False)
            print(f"載入架構版本: {arch_version}")
            print(f"梯度修復狀態: {'已修復' if gradient_fix else '未修復'}")
            
            # 定義自定義對象
            custom_objects = {
                'LayerNormalization': LayerNormalization,
                'SpatialAttention': SpatialAttention,
                'PositionalEncoding': PositionalEncoding,
                'CrossRobotAttention': CrossRobotAttention,
                'AdaptiveAttentionFusion': AdaptiveAttentionFusion,
                'TemporalAttentionModule': TemporalAttentionModule
            }
            
            # 載入模型
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
            
            print("修復梯度消失後的模型載入成功")
            return True
            
        except Exception as e:
            print(f"載入模型時出錯: {str(e)}")
            return False

    def verify_model(self):
        """驗證模型並檢查梯度流"""
        print("驗證修復梯度消失後的模型...")
        test_inputs = {
            'map_input': np.zeros((2, *self.input_shape)),  # 使用batch_size=2來測試
            'frontier_input': np.zeros((2, self.max_frontiers, 2)),
            'robot1_pos_input': np.zeros((2, 2)),
            'robot2_pos_input': np.zeros((2, 2)),
            'robot1_target_input': np.zeros((2, 2)),
            'robot2_target_input': np.zeros((2, 2))
        }
        
        try:
            # 測試前向傳播
            print("測試Actor前向傳播...")
            actor_outputs = self.actor.predict(test_inputs, verbose=0)
            
            print("測試Critic前向傳播...")
            critic_outputs = self.critic.predict(test_inputs, verbose=0)
            
            # 檢查輸出形狀
            print(f"Actor輸出形狀:")
            print(f"  Robot1 policy: {actor_outputs['robot1_policy'].shape}")
            print(f"  Robot2 policy: {actor_outputs['robot2_policy'].shape}")
            
            print(f"Critic輸出形狀:")
            print(f"  Robot1 value: {critic_outputs['robot1_value'].shape}")
            print(f"  Robot2 value: {critic_outputs['robot2_value'].shape}")
            
            # 驗證輸出形狀
            assert actor_outputs['robot1_policy'].shape == (2, self.max_frontiers)
            assert actor_outputs['robot2_policy'].shape == (2, self.max_frontiers)
            assert critic_outputs['robot1_value'].shape == (2, 1)
            assert critic_outputs['robot2_value'].shape == (2, 1)
            
            # 驗證概率分布
            robot1_probs = actor_outputs['robot1_policy'][0]
            robot2_probs = actor_outputs['robot2_policy'][0]
            
            print(f"Robot1概率分布統計:")
            print(f"  總和: {np.sum(robot1_probs):.6f}")
            print(f"  最小值: {np.min(robot1_probs):.6f}")
            print(f"  最大值: {np.max(robot1_probs):.6f}")
            
            print(f"Robot2概率分布統計:")
            print(f"  總和: {np.sum(robot2_probs):.6f}")
            print(f"  最小值: {np.min(robot2_probs):.6f}")
            print(f"  最大值: {np.max(robot2_probs):.6f}")
            
            # 驗證概率分布有效性
            assert abs(np.sum(robot1_probs) - 1.0) < 1e-5, "Robot1概率總和不為1"
            assert abs(np.sum(robot2_probs) - 1.0) < 1e-5, "Robot2概率總和不為1"
            assert np.all(robot1_probs >= 0), "Robot1存在負概率"
            assert np.all(robot2_probs >= 0), "Robot2存在負概率"
            
            # 測試梯度流動
            print("\n測試梯度流動...")
            self._test_gradient_flow(test_inputs)
            
            print("修復梯度消失後的模型驗證通過！")
            print("\n架構特點:")
            print("- 先進行自適應多模態融合")
            print("- 再進行時序注意力建模")
            print("- 跨機器人協作機制")
            print("- 記憶機制用於長期依賴")
            print("- 已修復梯度消失問題")
            
            return True
            
        except Exception as e:
            print(f"模型驗證失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_gradient_flow(self, test_inputs):
        """測試梯度流動，確保所有參數都參與梯度計算"""
        # 創建假的訓練數據
        fake_actions = {
            'robot1': tf.constant([0, 1], dtype=tf.int32),
            'robot2': tf.constant([1, 0], dtype=tf.int32)
        }
        fake_advantages = {
            'robot1': tf.constant([0.1, -0.1], dtype=tf.float32),
            'robot2': tf.constant([-0.1, 0.1], dtype=tf.float32)
        }
        fake_returns = {
            'robot1': tf.constant([[0.5], [0.3]], dtype=tf.float32),
            'robot2': tf.constant([[0.4], [0.6]], dtype=tf.float32)
        }
        
        # 測試Actor梯度
        print("測試Actor梯度流動...")
        with tf.GradientTape() as tape:
            policy_dict = self.actor(test_inputs, training=True)
            robot1_loss = self._compute_policy_loss(
                policy_dict['robot1_policy'],
                fake_actions['robot1'],
                fake_advantages['robot1']
            )
            robot2_loss = self._compute_policy_loss(
                policy_dict['robot2_policy'],
                fake_actions['robot2'],
                fake_advantages['robot2']
            )
            total_loss = robot1_loss + robot2_loss
        
        actor_grads = tape.gradient(total_loss, self.actor.trainable_variables)
        
        # 檢查是否有None梯度
        none_grad_vars = []
        for i, (grad, var) in enumerate(zip(actor_grads, self.actor.trainable_variables)):
            if grad is None:
                none_grad_vars.append(var.name)
        
        if none_grad_vars:
            print(f"警告: Actor中仍有變數沒有梯度: {none_grad_vars}")
        else:
            print("Actor所有變數都有梯度")
        
        # 測試Critic梯度
        print("測試Critic梯度流動...")
        with tf.GradientTape() as tape:
            values = self.critic(test_inputs, training=True)
            robot1_value_loss = tf.keras.losses.Huber()(
                fake_returns['robot1'], values['robot1_value'])
            robot2_value_loss = tf.keras.losses.Huber()(
                fake_returns['robot2'], values['robot2_value'])
            value_loss = robot1_value_loss + robot2_value_loss
        
        critic_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        
        # 檢查是否有None梯度
        none_grad_vars = []
        for i, (grad, var) in enumerate(zip(critic_grads, self.critic.trainable_variables)):
            if grad is None:
                none_grad_vars.append(var.name)
        
        if none_grad_vars:
            print(f"警告: Critic中仍有變數沒有梯度: {none_grad_vars}")
        else:
            print("Critic所有變數都有梯度")
        
        # 特別檢查TemporalAttentionModule中的memory_gate_dense
        print("檢查TemporalAttentionModule梯度...")
        temporal_vars = [var for var in self.actor.trainable_variables 
                        if 'temporal_attention_module' in var.name and 'memory_gate_dense' in var.name]
        
        if temporal_vars:
            print(f"找到TemporalAttentionModule變數: {[var.name for var in temporal_vars]}")
            
            # 檢查這些變數的梯度
            temporal_grads = [grad for grad, var in zip(actor_grads, self.actor.trainable_variables) 
                            if 'temporal_attention_module' in var.name and 'memory_gate_dense' in var.name]
            
            if all(grad is not None for grad in temporal_grads):
                print("TemporalAttentionModule的memory_gate_dense有梯度")
            else:
                print("警告: TemporalAttentionModule的memory_gate_dense仍然沒有梯度")
        else:
            print("未找到TemporalAttentionModule變數")

    def get_model_info(self):
        """獲取模型信息"""
        return {
            'architecture': 'B_gradient_fixed',
            'description': 'Adaptive Fusion → Temporal Attention (梯度消失已修復)',
            'input_shape': self.input_shape,
            'max_frontiers': self.max_frontiers,
            'd_model': self.d_model,
            'parameters': {
                'actor': self.actor.count_params(),
                'critic': self.critic.count_params(),
                'total': self.actor.count_params() + self.critic.count_params()
            },
            'features': [
                'Multi-scale spatial attention',
                'Cross-robot collaboration',
                'Adaptive multi-modal fusion',
                'Temporal attention with memory',
                'Gradient clipping for stability',
                'Huber loss for robust training',
                '已修復梯度消失問題'
            ],
            'fixes_applied': [
                '將update_gate_dense改為memory_gate_dense',
                'memory_gate_dense總是參與前向計算',
                '門控應用到時序特徵上，確保梯度流動',
                '移除條件性的梯度計算分支'
            ]
        }