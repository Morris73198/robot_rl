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
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """計算注意力權重"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
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
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        output = self.dropout(output, training=training)
        output = self.layer_norm(output)
        
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
        # 計算空間注意力圖
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention_map = self.conv1(concat)
        attention_map = tf.sigmoid(attention_map)
        
        # 應用注意力
        output = inputs * attention_map
        output = self.norm(output)
        return output

class MultiRobotActorCriticModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        """初始化 Actor-Critic 模型
        
        Args:
            input_shape: 輸入地圖的形狀，默認(84, 84, 1)
            max_frontiers: 最大 frontier 點數量，默認50
        """
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.d_model = 256
        self.num_heads = 8
        self.dff = 512
        self.dropout_rate = 0.1
        
        # 創建 Actor 和 Critic 網路
        self.actor_model = self._build_actor()
        self.critic_model = self._build_critic()
        
    def _build_perception_module(self, inputs):
        """構建感知模塊，用於處理地圖輸入
        
        Args:
            inputs: 地圖輸入張量
            
        Returns:
            處理後的特徵張量
        """
        # 多尺度特徵提取
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
            
            # 添加空間注意力
            x = SpatialAttention()(x)
            
            x = layers.Conv2D(
                filters=config['filters'],
                kernel_size=config['kernel_size'],
                strides=2,  # 降採樣
                padding='same',
                kernel_regularizer=regularizers.l2(0.01)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            features.append(x)
            
        # 特徵融合
        concat_features = layers.Concatenate()(features)
        x = layers.Conv2D(64, 1)(concat_features)  # 1x1 卷積融合通道
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        return x

    def _build_shared_layers(self, inputs):
        """構建共享的特徵提取層
        
        Args:
            inputs: 輸入字典，包含所有輸入張量
            
        Returns:
            共享特徵張量
        """
        # 1. 地圖特徵提取
        map_features = self._build_perception_module(inputs['map_input'])
        map_features = layers.Flatten()(map_features)
        
        # 2. Frontier 特徵提取
        frontier_input = inputs['frontier_input']
        x = layers.Dense(64, activation='relu')(frontier_input)
        x = layers.LSTM(64, return_sequences=True)(x)
        frontier_features = layers.GlobalAveragePooling1D()(x)
        
        # 3. 機器人位置和目標處理
        robot_state = layers.Concatenate()([
            inputs['robot1_pos_input'],
            inputs['robot2_pos_input'],
            inputs['robot1_target_input'],
            inputs['robot2_target_input']
        ])
        robot_features = layers.Dense(128, activation='relu')(robot_state)
        
        # 4. 特徵融合
        combined = layers.Concatenate()([
            map_features,
            frontier_features,
            robot_features
        ])
        
        x = layers.Dense(512, activation='relu')(combined)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        return x
        
    def _build_actor(self):
        """構建 Actor 網路"""
        # 定義輸入
        inputs = {
            'map_input': layers.Input(shape=self.input_shape),
            'frontier_input': layers.Input(shape=(self.max_frontiers, 2)),
            'robot1_pos_input': layers.Input(shape=(2,)),
            'robot2_pos_input': layers.Input(shape=(2,)),
            'robot1_target_input': layers.Input(shape=(2,)),
            'robot2_target_input': layers.Input(shape=(2,))
        }
        
        # 使用共享層
        shared_features = self._build_shared_layers(inputs)
        
        # 分別為兩個機器人構建動作輸出
        robot1_hidden = layers.Dense(128, activation='relu')(shared_features)
        robot2_hidden = layers.Dense(128, activation='relu')(shared_features)
        
        # 添加注意力機制
        robot1_attention = layers.Dense(128, activation='tanh')(robot1_hidden)
        robot1_attention = layers.Dense(1, activation='sigmoid')(robot1_attention)
        robot1_features = layers.Multiply()([robot1_hidden, robot1_attention])
        
        robot2_attention = layers.Dense(128, activation='tanh')(robot2_hidden)
        robot2_attention = layers.Dense(1, activation='sigmoid')(robot2_attention)
        robot2_features = layers.Multiply()([robot2_hidden, robot2_attention])
        
        # 動作概率輸出
        robot1_logits = layers.Dense(self.max_frontiers)(robot1_features)
        robot2_logits = layers.Dense(self.max_frontiers)(robot2_features)
        
        robot1_actions = layers.Softmax(name='robot1')(robot1_logits)
        robot2_actions = layers.Softmax(name='robot2')(robot2_logits)
        
        return models.Model(
            inputs=inputs,
            outputs={'robot1': robot1_actions, 'robot2': robot2_actions}
        )
        
    def _build_critic(self):
        """構建 Critic 網路"""
        # 定義輸入
        inputs = {
            'map_input': layers.Input(shape=self.input_shape),
            'frontier_input': layers.Input(shape=(self.max_frontiers, 2)),
            'robot1_pos_input': layers.Input(shape=(2,)),
            'robot2_pos_input': layers.Input(shape=(2,)),
            'robot1_target_input': layers.Input(shape=(2,)),
            'robot2_target_input': layers.Input(shape=(2,))
        }
        
        # 使用共享層
        shared_features = self._build_shared_layers(inputs)
        
        # 分別為兩個機器人構建狀態值輸出
        robot1_hidden = layers.Dense(128, activation='relu')(shared_features)
        robot2_hidden = layers.Dense(128, activation='relu')(shared_features)
        
        # 添加注意力機制
        robot1_attention = layers.Dense(128, activation='tanh')(robot1_hidden)
        robot1_attention = layers.Dense(1, activation='sigmoid')(robot1_attention)
        robot1_features = layers.Multiply()([robot1_hidden, robot1_attention])
        
        robot2_attention = layers.Dense(128, activation='tanh')(robot2_hidden)
        robot2_attention = layers.Dense(1, activation='sigmoid')(robot2_attention)
        robot2_features = layers.Multiply()([robot2_hidden, robot2_attention])
        
        # 狀態值輸出
        robot1_value = layers.Dense(1, name='robot1')(robot1_features)
        robot2_value = layers.Dense(1, name='robot2')(robot2_features)
        
        return models.Model(
            inputs=inputs,
            outputs={'robot1': robot1_value, 'robot2': robot2_value}
        )
    
    def _actor_loss(self, advantages, predicted_actions):
        """Actor 損失函數
        
        Args:
            advantages: 優勢值
            predicted_actions: 預測的動作概率分布
            
        Returns:
            損失值
        """
        # 計算策略梯度損失
        actions = tf.stop_gradient(predicted_actions)
        log_probs = tf.math.log(predicted_actions + 1e-10)
        policy_loss = -tf.reduce_mean(advantages * log_probs)
        
        # 添加熵正則化以鼓勵探索
        entropy = -tf.reduce_mean(predicted_actions * log_probs)
        entropy_coef = 0.01
        
        return policy_loss - entropy_coef * entropy
        
    def compile(self, actor_lr=0.0003, critic_lr=0.001):
        """編譯 Actor 和 Critic 網路
        
        Args:
            actor_lr: Actor 學習率
            critic_lr: Critic 學習率
        """
        # 為Actor網路配置優化器和損失函數
        self.actor_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=actor_lr,
                clipnorm=1.0  # 梯度裁剪
            ),
            loss={
                'robot1': self._actor_loss,
                'robot2': self._actor_loss
            }
        )
        
        # 為Critic網路配置優化器和損失函數
        self.critic_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=critic_lr,
                clipnorm=1.0  # 梯度裁剪
            ),
            loss={
                'robot1': 'mse',  # 使用均方誤差
                'robot2': 'mse'
            }
        )
    
    def get_actions(self, state, frontiers, robot1_pos, robot2_pos, 
                   robot1_target, robot2_target):
        """獲取動作,並添加形狀檢查和除錯資訊"""
        try:
            print("\n除錯 get_actions:")
            print(f"狀態形狀: {state.shape}")
            print(f"Robot1 位置形狀: {np.array(robot1_pos).shape}")
            print(f"Robot2 位置形狀: {np.array(robot2_pos).shape}")
            print(f"Robot1 目標形狀: {np.array(robot1_target).shape}")
            print(f"Robot2 目標形狀: {np.array(robot2_target).shape}")
            
            # 確保狀態有批次維度
            if len(state.shape) == 3:
                state = np.expand_dims(state, 0)
            print(f"添加批次後的狀態形狀: {state.shape}")
            
            # 處理和驗證 frontiers
            padded_frontiers = self.pad_frontiers(frontiers)
            padded_frontiers = np.expand_dims(padded_frontiers, 0)
            print(f"填充後的 frontiers 形狀: {padded_frontiers.shape}")
            
            # 處理位置和目標
            robot1_pos = np.array(robot1_pos, dtype=np.float32).reshape(1, 2)
            robot2_pos = np.array(robot2_pos, dtype=np.float32).reshape(1, 2)
            robot1_target = np.array(robot1_target, dtype=np.float32).reshape(1, 2)
            robot2_target = np.array(robot2_target, dtype=np.float32).reshape(1, 2)
            
            # 取得模型預測
            action_probs = self.actor_model.predict(
                {
                    'map_input': state,
                    'frontier_input': padded_frontiers,
                    'robot1_pos_input': robot1_pos,
                    'robot2_pos_input': robot2_pos,
                    'robot1_target_input': robot1_target,
                    'robot2_target_input': robot2_target
                },
                verbose=0
            )
            
            valid_frontiers = min(self.max_frontiers, len(frontiers))
            print(f"有效的 frontiers 數量: {valid_frontiers}")
            
            # 處理機率分布
            robot1_probs = action_probs['robot1'][0, :valid_frontiers]
            robot2_probs = action_probs['robot2'][0, :valid_frontiers]
            
            # 正規化機率
            robot1_probs = np.clip(robot1_probs, 1e-6, None)
            robot2_probs = np.clip(robot2_probs, 1e-6, None)
            robot1_probs = robot1_probs / np.sum(robot1_probs)
            robot2_probs = robot2_probs / np.sum(robot2_probs)
            
            return {
                'robot1': np.expand_dims(robot1_probs, 0),
                'robot2': np.expand_dims(robot2_probs, 0)
            }
            
        except Exception as e:
            print(f"get_actions 發生錯誤: {str(e)}")
            print(f"完整錯誤追蹤:")
            import traceback
            traceback.print_exc()
            raise

    
    def get_values(self, state, frontiers, robot1_pos, robot2_pos, 
                  robot1_target, robot2_target):
        """獲取狀態值估計
        
        Args:
            state: 環境狀態
            frontiers: frontier點列表
            robot1_pos: 機器人1位置
            robot2_pos: 機器人2位置
            robot1_target: 機器人1目標
            robot2_target: 機器人2目標
            
        Returns:
            包含兩個機器人狀態值的字典
        """
        return self.critic_model.predict(
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
    
    def save(self, filepath):
        """保存模型
        
        Args:
            filepath: 保存路徑
        """
        self.actor_model.save(filepath + '_actor')
        self.critic_model.save(filepath + '_critic')
        
        # 保存模型配置
        config = {
            'input_shape': self.input_shape,
            'max_frontiers': self.max_frontiers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        }
        
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f)
    
    def load(self, filepath):
        """載入模型
        
        Args:
            filepath: 模型路徑
        """
        # 載入自定義組件
        custom_objects = {
            'LayerNormalization': LayerNormalization,
            'MultiHeadAttention': MultiHeadAttention,
            'SpatialAttention': SpatialAttention,
            '_actor_loss': self._actor_loss
        }
        
        # 載入模型
        self.actor_model = models.load_model(
            filepath + '_actor',
            custom_objects=custom_objects
        )
        self.critic_model = models.load_model(
            filepath + '_critic',
            custom_objects=custom_objects
        )
        
        # 載入配置
        try:
            with open(filepath + '_config.json', 'r') as f:
                config = json.load(f)
                self.input_shape = config['input_shape']
                self.max_frontiers = config['max_frontiers']
                self.d_model = config['d_model']
                self.num_heads = config['num_heads']
                self.dff = config['dff']
                self.dropout_rate = config['dropout_rate']
        except FileNotFoundError:
            print("未找到配置文件，使用默認配置")
            
    def pad_frontiers(self, frontiers):
        """補充 frontier 點到固定長度,並添加詳細的形狀檢查
        
        Args:
            frontiers: frontier 點列表或 numpy 陣列
            
        Returns:
            形狀為 (max_frontiers, 2) 的填充陣列
        """
        try:
            print(f"\n除錯 pad_frontiers:")
            print(f"輸入 frontiers 類型: {type(frontiers)}")
            if isinstance(frontiers, np.ndarray):
                print(f"輸入 frontiers 形狀: {frontiers.shape}")
            else:
                print(f"輸入 frontiers 長度: {len(frontiers)}")
            
            # 先創建零陣列
            padded = np.zeros((self.max_frontiers, 2))
            print(f"創建填充陣列,形狀為: {padded.shape}")
            
            if len(frontiers) > 0:
                # 轉換為 numpy 陣列
                if not isinstance(frontiers, np.ndarray):
                    frontiers = np.array(frontiers)
                print(f"轉換後的 frontiers 形狀: {frontiers.shape}")
                
                # 確保是二維陣列
                if len(frontiers.shape) == 1:
                    if len(frontiers) == 2:  # 單點
                        frontiers = frontiers.reshape(1, 2)
                    else:  # 多點被壓平
                        frontiers = frontiers.reshape(-1, 2)
                print(f"重整形狀後的 frontiers: {frontiers.shape}")
                
                # 計算要包含的 frontier 數量
                n_frontiers = min(len(frontiers), self.max_frontiers)
                print(f"將使用 {n_frontiers} 個 frontiers")
                
                # 複製資料
                padded[:n_frontiers] = frontiers[:n_frontiers]
                print(f"最終填充陣列形狀: {padded.shape}")
                
            return padded
            
        except Exception as e:
            print(f"pad_frontiers 發生錯誤: {str(e)}")
            print(f"Frontiers 資料: {frontiers}")
            # 返回安全的預設值
            return np.zeros((self.max_frontiers, 2))