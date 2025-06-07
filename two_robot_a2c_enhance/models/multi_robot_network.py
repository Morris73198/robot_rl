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

class TemporalAttentionModule(layers.Layer):
    """時間注意力模組 - 大幅簡化版本"""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        # 簡化的處理層
        self.input_projection = layers.Dense(self.d_model, name='input_projection')
        self.temporal_dense = layers.Dense(self.d_model, activation='relu', name='temporal_dense')
        self.output_dense = layers.Dense(self.d_model, name='output_dense')
        
        # 正規化和dropout
        self.layer_norm = LayerNormalization()
        self.dropout = layers.Dropout(0.1)
        
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config
        
    def call(self, inputs, training=None):
        # 投影到目標維度
        projected = self.input_projection(inputs)
        
        # 簡單的前饋處理
        temporal_features = self.temporal_dense(projected)
        temporal_features = self.dropout(temporal_features, training=training)
        temporal_features = self.output_dense(temporal_features)
        
        # 殘差連接和正規化
        output = self.layer_norm(projected + temporal_features)
        
        return output

class AdaptiveAttentionFusion(layers.Layer):
    """自適應注意力融合層 - 簡化版"""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        # 投影層
        self.frontier_projection = layers.Dense(self.d_model, name='frontier_projection')
        self.cross_robot_projection = layers.Dense(self.d_model, name='cross_robot_projection')
        self.temporal_projection = layers.Dense(self.d_model, name='temporal_projection')
        self.map_projection = layers.Dense(self.d_model, name='map_projection')
        
        # 簡化的權重網路
        self.weight_network = layers.Dense(3, activation='softmax', name='weight_network')
        self.layer_norm = LayerNormalization()
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config
        
    def call(self, inputs):
        frontier_features, cross_robot_features, temporal_features, map_features = inputs
        
        # 投影到相同維度
        frontier_proj = self.frontier_projection(frontier_features)
        cross_robot_proj = self.cross_robot_projection(cross_robot_features)
        temporal_proj = self.temporal_projection(temporal_features)
        map_proj = self.map_projection(map_features)
        
        # 計算權重
        concatenated = tf.concat([frontier_proj, cross_robot_proj, temporal_proj], axis=-1)
        weights = self.weight_network(concatenated)
        w1, w2, w3 = tf.split(weights, 3, axis=-1)
        
        # 加權融合
        weighted_features = (
            w1 * frontier_proj + 
            w2 * cross_robot_proj + 
            w3 * temporal_proj
        )
        
        # 殘差連接
        output = weighted_features + map_proj
        output = self.layer_norm(output)
        
        return output

class MultiRobotACModel:
    """多機器人 Actor-Critic 模型 - 穩定優化版"""
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        
        # 大幅降低模型複雜度以提高穩定性
        self.d_model = 128  # 從 256 降到 128
        self.num_heads = 4  # 從 8 降到 4
        self.dff = 256      # 從 512 降到 256
        self.dropout_rate = 0.2  # 增加 dropout 率

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
        
        print("\n===== 優化後模型參數統計 =====")
        print(f"Actor 參數: {actor_params:,}")
        print(f"Critic 參數: {critic_params:,}")
        print(f"總參數數量: {total_params:,}")
        print(f"預期權重檔案大小: {total_params * 4 / (1024*1024):.2f} MB")
        print("===============================\n")
        
    def _build_perception_module(self, inputs):
        """構建感知模塊 - 降低複雜度"""
        # 降低輸入分辨率
        x = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
        
        # 大幅減少特徵數量
        conv_configs = [
            {'filters': 16, 'kernel_size': 3, 'strides': 2},  # 從32降到16
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
        x = layers.Conv2D(32, 1, padding='same')(x)  # 從64降到32
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        return x
        
    def _build_state_process_module(self, robot1_pos, robot1_target, robot2_pos, robot2_target):
        """構建狀態處理模塊 - 簡化版"""
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
        """構建前沿注意力層 - 簡化版"""
        # 減少Dense層大小
        x = layers.Dense(64, activation='relu')(frontier_input)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # 位置編碼
        x = PositionalEncoding(self.max_frontiers, 64)(x)
        
        # 使用更簡單的注意力機制
        attention_output = layers.MultiHeadAttention(
            num_heads=2,  # 從4降到2
            key_dim=32,   # 從16增加到32但總體更簡單
            dropout=self.dropout_rate
        )(x, x)
        
        # 簡化的前饋網路
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
        """構建增強特徵提取 - 大幅簡化"""
        # 跨機器人注意力 - 使用較小維度
        cross_attention = CrossRobotAttention(64)  # 從self.d_model降到64
        r1_enhanced, r2_enhanced = cross_attention([robot1_features, robot2_features])
        
        # 合併跨機器人特徵
        cross_robot_features = layers.Concatenate()([r1_enhanced, r2_enhanced])
        
        # 時間注意力模組 - 使用較小維度
        temporal_attention = TemporalAttentionModule(64)  # 從self.d_model降到64
        temporal_features = temporal_attention(frontier_features)
        
        # 自適應注意力融合 - 使用較小維度
        fusion_layer = AdaptiveAttentionFusion(64)  # 從self.d_model降到64
        fused_features = fusion_layer([
            frontier_features, 
            cross_robot_features, 
            temporal_features, 
            map_features
        ])
        
        return fused_features
        
    def _build_process_module(self, features):
        """構建處理模塊 - 減少複雜度"""
        # 減少層大小
        x = layers.Dense(128, activation='relu')(features)  # 從256降到128
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)  # 從128降到64
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
        
        # 增強特徵提取
        enhanced_features = self._build_enhanced_features(
            map_features, frontier_features, robot1_features, robot2_features
        )
        
        # 處理模塊
        processed_features = self._build_process_module(enhanced_features)
        
        # Actor 網絡輸出 - 減少層大小
        # Robot1 policy
        robot1_policy = layers.Dense(64, activation='relu')(processed_features)  # 從128降到64
        robot1_policy = layers.Dropout(self.dropout_rate)(robot1_policy)
        robot1_policy = layers.Dense(32, activation='relu')(robot1_policy)  # 從64降到32
        robot1_policy = layers.Dropout(self.dropout_rate)(robot1_policy)
        robot1_policy = layers.Dense(self.max_frontiers, activation='softmax', 
                                   name='robot1_policy')(robot1_policy)
        
        # Robot2 policy
        robot2_policy = layers.Dense(64, activation='relu')(processed_features)  # 從128降到64
        robot2_policy = layers.Dropout(self.dropout_rate)(robot2_policy)
        robot2_policy = layers.Dense(32, activation='relu')(robot2_policy)  # 從64降到32
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
        
        # 使用更小的學習率提高穩定性
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)  # 從0.0001降到0.00005
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
        
        # 增強特徵提取
        enhanced_features = self._build_enhanced_features(
            map_features, frontier_features, robot1_features, robot2_features
        )
        
        # 處理模塊
        processed_features = self._build_process_module(enhanced_features)
        
        # Critic 網絡輸出 - 減少層大小
        # Robot1 value
        robot1_value = layers.Dense(64, activation='relu')(processed_features)  # 從128降到64
        robot1_value = layers.Dropout(self.dropout_rate)(robot1_value)
        robot1_value = layers.Dense(32, activation='relu')(robot1_value)  # 從64降到32
        robot1_value = layers.Dropout(self.dropout_rate)(robot1_value)
        robot1_value = layers.Dense(1, name='robot1_value')(robot1_value)
        
        # Robot2 value
        robot2_value = layers.Dense(64, activation='relu')(processed_features)  # 從128降到64
        robot2_value = layers.Dropout(self.dropout_rate)(robot2_value)
        robot2_value = layers.Dense(32, activation='relu')(robot2_value)  # 從64降到32
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
        
        # 使用更小的學習率
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
            
        # 增強梯度裁剪
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.3)  # 從0.5降到0.3，更嚴格的裁剪
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
            
        # 增強梯度裁剪
        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 0.3)  # 從0.5降到0.3
        self.critic.optimizer.apply_gradients(
            zip(grads, self.critic.trainable_variables))
            
        return value_loss
    
    def _compute_policy_loss(self, policy, actions, advantages):
        """計算策略損失"""
        actions_one_hot = tf.one_hot(actions, self.max_frontiers)
        log_prob = tf.math.log(tf.reduce_sum(policy * actions_one_hot, axis=1) + 1e-10)
        policy_loss = -tf.reduce_mean(log_prob * advantages)
        
        # 降低熵正則化權重以提高穩定性
        entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1)
        entropy_bonus = 0.005 * tf.reduce_mean(entropy)  # 從0.01降到0.005
        
        return policy_loss - entropy_bonus
    
    def save(self, filepath):
        """保存模型到 h5 格式"""
        print("保存優化後模型...")
        
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
                'dropout_rate': self.dropout_rate
            }
            
            with open(filepath + '_config.json', 'w') as f:
                json.dump(config, f, indent=4)
                
            print("優化後模型保存成功")
            return True
            
        except Exception as e:
            print(f"保存模型時出錯: {str(e)}")
            return False

    def load(self, filepath):
        """載入 h5 格式的模型"""
        print("載入優化後模型...")
        
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
            
            # 定義自定義對象
            custom_objects = {
                'LayerNormalization': LayerNormalization,
                'SpatialAttention': SpatialAttention,
                'PositionalEncoding': PositionalEncoding,
                'CrossRobotAttention': CrossRobotAttention,
                'TemporalAttentionModule': TemporalAttentionModule,
                'AdaptiveAttentionFusion': AdaptiveAttentionFusion
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
            
            print("優化後模型載入成功")
            return True
            
        except Exception as e:
            print(f"載入模型時出錯: {str(e)}")
            return False

    def verify_model(self):
        """驗證模型"""
        print("驗證優化後模型...")
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
            
            print("優化後模型驗證通過")
            return True
            
        except Exception as e:
            print(f"模型驗證失敗: {str(e)}")
            return False