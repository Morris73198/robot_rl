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

class MultiRobotNetworkModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        """初始化多機器人網路模型
        
        Args:
            input_shape: 輸入地圖的形狀，默認(84, 84, 1)
            max_frontiers: 最大frontier點數量，默認50
        """
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.dropout_rate = 0.1
        
        # 構建並編譯模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        
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
        
    def _build_perception_module(self, inputs):
        """構建感知模塊，包含空間注意力 (A)"""
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
            
            # 包含空間注意力 (A)
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

    def _build_simple_robot_features(self, robot_pos, robot_target):
        """簡單的機器人特徵處理 (移除B)"""
        robot_state = layers.Concatenate()([robot_pos, robot_target])
        robot_features = layers.Dense(
            128,  # 較小的維度，因為沒有複雜的協調機制
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        )(robot_state)
        
        return robot_features
        
    def _build_simple_frontier_module(self, frontier_input, robot_features):
        """簡單的frontier處理模塊 (移除C)"""
        # 簡單地處理frontier輸入
        x = layers.Dense(64, activation='relu')(frontier_input)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Flatten()(x)
        
        # 與機器人特徵連接
        combined = layers.Concatenate()([x, robot_features])
        combined = layers.Dense(128, activation='relu')(combined)
        
        return combined

    def _build_model(self):
        """構建完整的模型，使用Dueling DQN架構"""
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
        
        # 1. 地圖感知處理，包含空間注意力 (A)
        map_features = self._build_perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features)
        
        # 2. 簡單的機器人特徵處理 (移除B)
        robot1_features = self._build_simple_robot_features(robot1_pos, robot1_target)
        robot2_features = self._build_simple_robot_features(robot2_pos, robot2_target)
        
        # 3. 簡單的Frontier處理 (移除C)
        robot1_frontier = self._build_simple_frontier_module(frontier_input, robot1_features)
        robot2_frontier = self._build_simple_frontier_module(frontier_input, robot2_features)
        
        # 4. Dueling DQN架構 (D)
        def build_dueling_streams(features, name_prefix):
            # 共享特徵層
            shared = layers.Dense(512, activation='relu')(features)
            shared = layers.Dropout(self.dropout_rate)(shared)
            shared = layers.Dense(256, activation='relu')(shared)
            shared = layers.Dropout(self.dropout_rate)(shared)
            
            # 價值流 (Value Stream)
            value_stream = layers.Dense(128, activation='relu')(shared)
            value = layers.Dense(1, name=f'{name_prefix}_value')(value_stream)
            
            # 優勢流 (Advantage Stream)
            advantage_stream = layers.Dense(128, activation='relu')(shared)
            advantage = layers.Dense(
                self.max_frontiers, 
                name=f'{name_prefix}_advantage'
            )(advantage_stream)
            
            # 組合價值和優勢
            mean_advantage = layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=1, keepdims=True)
            )(advantage)
            
            q_values = layers.Add(name=name_prefix)([
                value,
                layers.Subtract()([advantage, mean_advantage])
            ])
            
            return q_values
        
        # Robot 1的Dueling網絡 (D)
        robot1_features = layers.Concatenate()([
            robot1_frontier,
            map_features_flat
        ])
        robot1_output = build_dueling_streams(robot1_features, 'robot1')
        
        # Robot 2的Dueling網絡 (D)
        robot2_features = layers.Concatenate()([
            robot2_frontier,
            map_features_flat
        ])
        robot2_output = build_dueling_streams(robot2_features, 'robot2')
        
        # 構建最終模型
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
                'robot1': robot1_output,
                'robot2': robot2_output
            }
        )
        
        # 使用自定義的學習率調度器
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0005,
            decay_steps=2000,
            decay_rate=0.95
        )
        
        # 編譯模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr_schedule,
                clipnorm=0.5,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss={
                'robot1': self._huber_loss,
                'robot2': self._huber_loss
            }
        )
        
        return model
        
    def _huber_loss(self, y_true, y_pred):
        return tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)

    def update_target_model(self):
        """更新目標網路"""
        tau = 0.001  # 軟更新係數
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = tau * weights[i] + (1 - tau) * target_weights[i]
        self.target_model.set_weights(target_weights)
    
    def predict(self, state, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target):
        """預測動作值"""
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
    
    def train_on_batch(self, states, frontiers, robot1_pos, robot2_pos, 
                      robot1_target, robot2_target,
                      robot1_targets, robot2_targets):
        """訓練一個批次"""
        history = self.model.train_on_batch(
            {
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            },
            {
                'robot1': robot1_targets,
                'robot2': robot2_targets
            }
        )
        return history
    
    def save(self, filepath):
        """保存模型"""
        # 保存模型架構和權重
        self.model.save(filepath)
        # 保存額外的配置信息
        config = {
            'input_shape': self.input_shape,
            'max_frontiers': self.max_frontiers,
            'dropout_rate': self.dropout_rate
        }
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f)
    
    def load(self, filepath):
        """載入模型"""
        # 創建自定義對象字典
        custom_objects = {
            'LayerNormalization': LayerNormalization,
            'SpatialAttention': SpatialAttention,
            '_huber_loss': self._huber_loss
        }
        
        # 載入模型
        self.model = models.load_model(
            filepath,
            custom_objects=custom_objects
        )
        self.target_model = models.load_model(
            filepath,
            custom_objects=custom_objects
        )
