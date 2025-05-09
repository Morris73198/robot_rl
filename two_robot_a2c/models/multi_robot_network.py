import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers, optimizers
import json

# 保持原有的自定義層不變
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
    """前饋神經網絡層"""
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
        
    def get_config(self):
        return super().get_config()

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
        """構建感知模塊"""
        conv_configs = [
            {'filters': 32, 'kernel_size': 3, 'strides': 1},
            {'filters': 32, 'kernel_size': 5, 'strides': 1},
            {'filters': 32, 'kernel_size': 7, 'strides': 1}
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
            
        concat_features = layers.Concatenate()(features)
        x = layers.Conv2D(64, 1)(concat_features)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        return x

    def _build_coordination_module(self, robot1_state, robot2_state):
        """構建協調模塊"""
        robot1_expanded = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1)
        )(robot1_state)
        robot2_expanded = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1)
        )(robot2_state)
        
        combined_states = layers.Concatenate(axis=1)([
            robot1_expanded, robot2_expanded
        ])
        
        attention = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )([combined_states, combined_states, combined_states])
        
        ffn = FeedForward(
            d_model=self.d_model,
            dff=self.dff,
            dropout_rate=self.dropout_rate
        )(attention)
        
        robot1_coord = layers.Lambda(lambda x: x[:, 0, :])(ffn)
        robot2_coord = layers.Lambda(lambda x: x[:, 1, :])(ffn)
        
        return robot1_coord, robot2_coord
        
    def _build_frontier_module(self, frontier_input, robot_state):
        """構建frontier評估模塊"""
        x = layers.Dense(64, activation='relu')(frontier_input)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = PositionalEncoding(self.max_frontiers, 64)(x)
        
        attention = MultiHeadAttention(
            d_model=64,
            num_heads=4,
            dropout_rate=self.dropout_rate
        )([x, x, x])
        
        robot_state_expanded = layers.RepeatVector(self.max_frontiers)(robot_state)
        combined = layers.Concatenate()([attention, robot_state_expanded])
        
        x = layers.Bidirectional(layers.LSTM(
            32, 
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.01)
        ))(combined)
        
        return x

    def _build_actor_network(self, features, name_prefix):
        """構建Actor網絡"""
        # 共享特徵層
        shared = layers.Dense(256, activation='relu', name=f"{name_prefix}_actor_dense1")(features)
        shared = layers.Dropout(self.dropout_rate)(shared)
        shared = layers.Dense(128, activation='relu', name=f"{name_prefix}_actor_dense2")(shared)
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
        """構建Critic網絡"""
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

    # ----- 修改部分: 使用分離的Actor-Critic模型 -----
    
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
            kernel_regularizer=regularizers.l2(0.01)
        )(robot1_state)
        robot2_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(robot2_state)
        
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
        
        # 設置優化器但不編譯，我們將使用自定義訓練函數
        actor_model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
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
            kernel_regularizer=regularizers.l2(0.01)
        )(robot1_state)
        robot2_features = layers.Dense(
            self.d_model,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(robot2_state)
        
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
        
        # 設置優化器並編譯critic模型
        critic_model.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        critic_model.compile(optimizer=critic_model.optimizer, loss='mse')
        
        return critic_model

    # ----- 以下是新的loss計算方法和訓練函數 -----
    
    def _compute_policy_loss(self, policy, actions, advantages):
        """計算策略損失
        
        Args:
            policy: 策略網絡輸出的動作概率分佈
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
        entropy_bonus = self.entropy_beta * tf.reduce_mean(entropy)
        
        return policy_loss - entropy_bonus
    
    def train_actor(self, states, frontiers, robot1_pos, robot2_pos, 
                   robot1_target, robot2_target, 
                   robot1_actions, robot2_actions, 
                   robot1_advantages, robot2_advantages):
        """訓練Actor網絡
        
        Args:
            states: 狀態批次
            frontiers: frontier點批次
            robot1_pos, robot2_pos: 機器人位置
            robot1_target, robot2_target: 機器人目標
            robot1_actions, robot2_actions: 實際動作
            robot1_advantages, robot2_advantages: 計算出的優勢值
            
        Returns:
            loss: 訓練損失
        """
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
            
            # 計算機器人1的策略損失
            robot1_loss = self._compute_policy_loss(
                policy_dict['robot1_policy'],
                robot1_actions,
                robot1_advantages
            )
            
            # 計算機器人2的策略損失
            robot2_loss = self._compute_policy_loss(
                policy_dict['robot2_policy'],
                robot2_actions,
                robot2_advantages
            )
            
            # 總策略損失
            total_loss = robot1_loss + robot2_loss
            
        # 計算梯度並應用
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        
        # 梯度裁剪
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        
        # 應用梯度
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        
        # 返回訓練指標
        return {
            'total_policy_loss': total_loss,
            'robot1_policy_loss': robot1_loss,
            'robot2_policy_loss': robot2_loss
        }
    
    def train_critic(self, states, frontiers, robot1_pos, robot2_pos, 
                    robot1_target, robot2_target, 
                    robot1_returns, robot2_returns):
        """訓練Critic網絡
        
        Args:
            states: 狀態批次
            frontiers: frontier點批次
            robot1_pos, robot2_pos: 機器人位置
            robot1_target, robot2_target: 機器人目標
            robot1_returns, robot2_returns: 實際回報
            
        Returns:
            loss: 訓練損失
        """
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
            
            # 計算機器人1的價值損失
            robot1_value_loss = tf.keras.losses.Huber()(
                robot1_returns, values['robot1_value'])
            
            # 計算機器人2的價值損失
            robot2_value_loss = tf.keras.losses.Huber()(
                robot2_returns, values['robot2_value'])
            
            # 總價值損失
            value_loss = robot1_value_loss + robot2_value_loss
            
        # 計算梯度並應用
        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        
        # 梯度裁剪
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        
        # 應用梯度
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        
        # 返回訓練指標
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
        """訓練一個批次的數據
        
        注意：雖然實現了network2的形式，但保留了原始API的參數列表，
        robot1_old_values, robot2_old_values, robot1_old_logits, robot2_old_logits 
        參數在新實現中沒有使用
        """
        # 訓練Actor
        actor_metrics = self.train_actor(
            states, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target,
            robot1_actions, robot2_actions, 
            robot1_advantages, robot2_advantages
        )
        
        # 訓練Critic
        critic_metrics = self.train_critic(
            states, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target,
            robot1_returns, robot2_returns
        )
        
        # 計算熵
        policy_dict = self.actor.predict({
            'map_input': states,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        }, verbose=0)
        
        # 計算熵
        robot1_entropy = -tf.reduce_mean(tf.reduce_sum(
            policy_dict['robot1_policy'] * tf.math.log(policy_dict['robot1_policy'] + 1e-10), 
            axis=1
        ))
        robot2_entropy = -tf.reduce_mean(tf.reduce_sum(
            policy_dict['robot2_policy'] * tf.math.log(policy_dict['robot2_policy'] + 1e-10), 
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
            'gradient_norm': 0.0  # 保持接口一致，但簡化實現
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
        
        # 合併結果
        return {
            'robot1_policy': actor_outputs['robot1_policy'],
            'robot2_policy': actor_outputs['robot2_policy'],
            'robot1_value': critic_outputs['robot1_value'],
            'robot2_value': critic_outputs['robot2_value'],
            'robot1_logits': actor_outputs['robot1_logits'],
            'robot2_logits': actor_outputs['robot2_logits']
        }
    
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
            'SpatialAttention': SpatialAttention
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
                self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            
            if not hasattr(self.critic, 'optimizer') or self.critic.optimizer is None:
                self.critic.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
                self.critic.compile(optimizer=self.critic.optimizer, loss='mse')
            
            print("模型載入成功")
            return True
            
        except Exception as e:
            print(f"載入模型時出錯: {str(e)}")
            return False