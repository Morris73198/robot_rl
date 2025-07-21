import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.models import Sequential
import numpy as np


class AttentionLayer(layers.Layer):
    def __init__(self, attention_dim=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        # input_shape: [batch_size, num_robots, feature_dim]
        self.W_q = self.add_weight(
            name='query_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='random_normal',
            trainable=True
        )
        self.W_k = self.add_weight(
            name='key_weight', 
            shape=(input_shape[-1], self.attention_dim),
            initializer='random_normal',
            trainable=True
        )
        self.W_v = self.add_weight(
            name='value_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='random_normal',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs, mask=None):
        # inputs: [batch_size, num_robots, feature_dim]
        queries = tf.matmul(inputs, self.W_q)  # [batch_size, num_robots, attention_dim]
        keys = tf.matmul(inputs, self.W_k)     # [batch_size, num_robots, attention_dim]  
        values = tf.matmul(inputs, self.W_v)   # [batch_size, num_robots, attention_dim]
        
        # 計算注意力權重
        attention_scores = tf.matmul(queries, keys, transpose_b=True)  # [batch_size, num_robots, num_robots]
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.attention_dim, tf.float32))
        
        if mask is not None:
            attention_scores += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # 應用注意力權重
        attended_values = tf.matmul(attention_weights, values)  # [batch_size, num_robots, attention_dim]
        
        return attended_values, attention_weights


class MultiRobotDQNNetwork:
    def __init__(self, num_robots=2, state_size=(84, 84, 1), num_actions=8, 
                 learning_rate=0.0001, use_attention=True):
        """
        多機器人DQN網絡
        
        Args:
            num_robots: 機器人數量
            state_size: 單個機器人的狀態空間大小
            num_actions: 動作空間大小
            learning_rate: 學習率
            use_attention: 是否使用注意力機制
        """
        self.num_robots = num_robots
        self.state_size = state_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.use_attention = use_attention
        
        # 建構網絡
        self.model = self._build_model()
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

    def _build_cnn_feature_extractor(self):
        """建構CNN特徵提取器"""
        cnn = Sequential([
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', 
                         input_shape=self.state_size, name='conv1'),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu', name='conv2'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv3'),
            layers.Flatten(name='flatten'),
            layers.Dense(512, activation='relu', name='dense_features')
        ], name='cnn_feature_extractor')
        return cnn

    def _build_model(self):
        """建構完整的多機器人DQN網絡"""
        # 輸入層 - 所有機器人的狀態
        state_inputs = layers.Input(shape=(self.num_robots,) + self.state_size, 
                                   name='state_inputs')
        
        # CNN特徵提取器
        cnn_extractor = self._build_cnn_feature_extractor()
        
        # 為每個機器人提取特徵
        robot_features = []
        for i in range(self.num_robots):
            robot_state = layers.Lambda(lambda x, idx=i: x[:, idx], 
                                      name=f'robot_{i}_state')(state_inputs)
            robot_feature = cnn_extractor(robot_state)
            robot_features.append(robot_feature)
        
        # 堆疊所有機器人的特徵
        stacked_features = layers.Lambda(
            lambda x: tf.stack(x, axis=1), 
            name='stacked_features'
        )(robot_features)  # [batch_size, num_robots, feature_dim]
        
        if self.use_attention:
            # 應用注意力機制
            attended_features, attention_weights = AttentionLayer(
                attention_dim=128, name='attention_layer'
            )(stacked_features)
            
            # 結合原始特徵和注意力特徵
            combined_features = layers.Add(name='combine_features')([
                stacked_features, attended_features
            ])
        else:
            combined_features = stacked_features
        
        # 為每個機器人生成Q值
        q_values_list = []
        for i in range(self.num_robots):
            # 提取單個機器人的特徵
            robot_combined_feature = layers.Lambda(
                lambda x, idx=i: x[:, idx], 
                name=f'robot_{i}_combined_feature'
            )(combined_features)
            
            # 全局特徵 (所有機器人特徵的平均)
            global_feature = layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=1), 
                name='global_feature'
            )(combined_features)
            
            # 結合局部和全局特徵
            robot_final_feature = layers.Concatenate(
                name=f'robot_{i}_final_feature'
            )([robot_combined_feature, global_feature])
            
            # Q值預測
            robot_q_values = layers.Dense(
                256, activation='relu', 
                name=f'robot_{i}_dense1'
            )(robot_final_feature)
            robot_q_values = layers.Dropout(0.3, name=f'robot_{i}_dropout')(robot_q_values)
            robot_q_values = layers.Dense(
                128, activation='relu', 
                name=f'robot_{i}_dense2'
            )(robot_q_values)
            robot_q_values = layers.Dense(
                self.num_actions, activation='linear',
                name=f'robot_{i}_q_values'
            )(robot_q_values)
            
            q_values_list.append(robot_q_values)
        
        # 堆疊所有機器人的Q值
        all_q_values = layers.Lambda(
            lambda x: tf.stack(x, axis=1), 
            name='all_q_values'
        )(q_values_list)  # [batch_size, num_robots, num_actions]
        
        model = Model(inputs=state_inputs, outputs=all_q_values, name='multi_robot_dqn')
        return model

    def predict(self, states, robot_positions=None):
        """
        預測Q值
        
        Args:
            states: 狀態 [num_robots, height, width, channels]
            robot_positions: 機器人位置 [num_robots, 2] (可選)
            
        Returns:
            q_values: Q值 [num_robots, num_actions]
        """
        # 增加批次維度
        batch_states = np.expand_dims(states, axis=0)
        q_values = self.model(batch_states, training=False)
        return q_values.numpy()[0]  # 移除批次維度

    def predict_batch(self, batch_states):
        """
        批次預測Q值
        
        Args:
            batch_states: 批次狀態 [batch_size, num_robots, height, width, channels]
            
        Returns:
            q_values: Q值 [batch_size, num_robots, num_actions]
        """
        return self.model(batch_states, training=False).numpy()

    def train_step(self, batch_states, batch_targets):
        """
        執行一個訓練步驟
        
        Args:
            batch_states: 批次狀態 [batch_size, num_robots, height, width, channels]
            batch_targets: 目標Q值 [batch_size, num_robots, num_actions]
            
        Returns:
            loss: 損失值
        """
        with tf.GradientTape() as tape:
            predictions = self.model(batch_states, training=True)
            loss = tf.reduce_mean(tf.square(batch_targets - predictions))
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss.numpy()

    def get_attention_weights(self, states):
        """
        獲取注意力權重（如果使用注意力機制）
        
        Args:
            states: 狀態 [num_robots, height, width, channels]
            
        Returns:
            attention_weights: 注意力權重 [num_robots, num_robots]
        """
        if not self.use_attention:
            return None
            
        # 創建一個子模型來獲取注意力權重
        attention_layer = None
        for layer in self.model.layers:
            if isinstance(layer, AttentionLayer):
                attention_layer = layer
                break
                
        if attention_layer is None:
            return None
            
        # 這裡需要更複雜的實現來提取注意力權重
        # 簡化版本，返回None
        return None

    def summary(self):
        """打印模型摘要"""
        self.model.summary()

    def save_weights(self, filepath):
        """保存權重"""
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        """加載權重"""
        self.model.load_weights(filepath)


class DuelingDQN(MultiRobotDQNNetwork):
    """Dueling DQN變體，支持多機器人"""
    
    def _build_model(self):
        """建構Dueling DQN網絡"""
        # 輸入層
        state_inputs = layers.Input(shape=(self.num_robots,) + self.state_size, 
                                   name='state_inputs')
        
        # CNN特徵提取器
        cnn_extractor = self._build_cnn_feature_extractor()
        
        # 為每個機器人提取特徵
        robot_features = []
        for i in range(self.num_robots):
            robot_state = layers.Lambda(lambda x, idx=i: x[:, idx], 
                                      name=f'robot_{i}_state')(state_inputs)
            robot_feature = cnn_extractor(robot_state)
            robot_features.append(robot_feature)
        
        # 堆疊特徵
        stacked_features = layers.Lambda(
            lambda x: tf.stack(x, axis=1), 
            name='stacked_features'
        )(robot_features)
        
        if self.use_attention:
            attended_features, _ = AttentionLayer(
                attention_dim=128, name='attention_layer'
            )(stacked_features)
            combined_features = layers.Add(name='combine_features')([
                stacked_features, attended_features
            ])
        else:
            combined_features = stacked_features
        
        # Dueling結構：為每個機器人分別計算Value和Advantage
        q_values_list = []
        for i in range(self.num_robots):
            robot_feature = layers.Lambda(
                lambda x, idx=i: x[:, idx], 
                name=f'robot_{i}_feature'
            )(combined_features)
            
            # 全局特徵
            global_feature = layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=1), 
                name='global_feature'
            )(combined_features)
            
            robot_final_feature = layers.Concatenate(
                name=f'robot_{i}_final_feature'
            )([robot_feature, global_feature])
            
            # Value stream
            value_stream = layers.Dense(256, activation='relu', 
                                      name=f'robot_{i}_value_dense1')(robot_final_feature)
            value_stream = layers.Dense(128, activation='relu',
                                      name=f'robot_{i}_value_dense2')(value_stream)
            value = layers.Dense(1, activation='linear', 
                               name=f'robot_{i}_value')(value_stream)
            
            # Advantage stream  
            advantage_stream = layers.Dense(256, activation='relu',
                                          name=f'robot_{i}_adv_dense1')(robot_final_feature)
            advantage_stream = layers.Dense(128, activation='relu',
                                          name=f'robot_{i}_adv_dense2')(advantage_stream)
            advantage = layers.Dense(self.num_actions, activation='linear',
                                   name=f'robot_{i}_advantage')(advantage_stream)
            
            # 組合Value和Advantage
            advantage_mean = layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=1, keepdims=True),
                name=f'robot_{i}_adv_mean'
            )(advantage)
            
            q_values = layers.Add(name=f'robot_{i}_q_combine')([
                value, 
                layers.Subtract(name=f'robot_{i}_adv_subtract')([advantage, advantage_mean])
            ])
            
            q_values_list.append(q_values)
        
        # 堆疊所有Q值
        all_q_values = layers.Lambda(
            lambda x: tf.stack(x, axis=1), 
            name='all_q_values'
        )(q_values_list)
        
        model = Model(inputs=state_inputs, outputs=all_q_values, name='multi_robot_dueling_dqn')
        return model