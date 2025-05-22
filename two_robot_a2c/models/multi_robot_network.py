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

class MultiRobotACModel:
    """多機器人 Actor-Critic 模型"""
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.d_model = 256
        self.num_heads = 8
        self.dff = 512
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
        
    def _build_perception_module(self, inputs):
        """構建共享的感知模塊"""
        # 降低輸入分辨率
        x = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
        
        # 使用不同大小的卷積核進行特征提取
        conv_configs = [
            {'filters': 32, 'kernel_size': 3, 'strides': 2},
            {'filters': 32, 'kernel_size': 5, 'strides': 2},
            {'filters': 32, 'kernel_size': 7, 'strides': 2}
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
        x = layers.Conv2D(64, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        return x
        
    def _build_shared_features(self, map_features, frontier_input, robot1_state, robot2_state):
        """構建共享特征提取層"""
        # 處理 frontier 特征
        frontier_features = layers.Dense(64, activation='relu')(frontier_input)
        frontier_features = layers.Dropout(self.dropout_rate)(frontier_features)
        
        # 添加位置編碼
        pos_encoding = PositionalEncoding(self.max_frontiers, 64)(frontier_features)
        
        # 對 frontier 序列使用多頭注意力
        attention_output = MultiHeadAttention(
            d_model=64,
            num_heads=4,
            dropout_rate=self.dropout_rate
        )([pos_encoding, pos_encoding, pos_encoding])
        
        frontier_features = layers.Add()([frontier_features, attention_output])
        frontier_features = FeedForward(64, 128)(frontier_features)
        
        # 處理機器人狀態
        robot_features = layers.Concatenate()([
            layers.Dense(32, activation='relu')(robot1_state),
            layers.Dense(32, activation='relu')(robot2_state)
        ])
        
        # 合併所有特征
        combined_features = layers.Concatenate()([
            map_features,
            layers.GlobalAveragePooling1D()(frontier_features),
            robot_features
        ])
        
        x = layers.Dense(256, activation='relu')(combined_features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        
        return x
        
    def _build_policy_head(self, features, name_prefix):
        """構建策略輸出頭"""
        x = layers.Dense(128, activation='relu')(features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        policy = layers.Dense(
            self.max_frontiers,
            activation='softmax',
            name=name_prefix
        )(x)
        return policy

    def _build_value_head(self, features, name_prefix):
        """構建價值輸出頭"""
        x = layers.Dense(128, activation='relu')(features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
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
        
        # 添加熵正則化以鼓勵探索
        entropy = -tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1)
        entropy_bonus = 0.01 * tf.reduce_mean(entropy)
        
        return policy_loss - entropy_bonus
    
    def save(self, filepath):
        """保存模型到 h5 格式"""
        print("保存模型...")
        
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
                
            print("模型保存成功")
            return True
            
        except Exception as e:
            print(f"保存模型時出錯: {str(e)}")
            return False

    def load(self, filepath):
        """載入 h5 格式的模型"""
        print("載入模型...")
        
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
                'SpatialAttention': SpatialAttention
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
            
            print("模型載入成功")
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