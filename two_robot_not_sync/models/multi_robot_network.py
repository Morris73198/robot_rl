import tensorflow as tf
import numpy as np

class MultiRobotNetworkModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    def _build_model(self):
        # 共享地圖輸入
        map_input = tf.keras.layers.Input(shape=(84, 84, 1), name='map_input')
        
        # 兩個機器人的位置輸入
        robot1_pos_input = tf.keras.layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos_input = tf.keras.layers.Input(shape=(2,), name='robot2_pos_input')
        
        # CNN處理地圖 - 共享特徵提取器
        x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu')(map_input)
        x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
        map_features = tf.keras.layers.Flatten()(x)
        
        # Frontier輸入
        frontier_input = tf.keras.layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        
        # 處理frontier位置
        frontier_features = tf.keras.layers.Dense(128, activation='relu')(frontier_input)
        frontier_features = tf.keras.layers.Flatten()(frontier_features)
        
        # 處理兩個機器人位置
        robot1_features = tf.keras.layers.Dense(64, activation='relu')(robot1_pos_input)
        robot2_features = tf.keras.layers.Dense(64, activation='relu')(robot2_pos_input)
        
        # 特徵融合
        combined = tf.keras.layers.Concatenate()([
            map_features, 
            frontier_features, 
            robot1_features,
            robot2_features
        ])
        
        # 共享特徵處理
        shared = tf.keras.layers.Dense(512, activation='relu')(combined)
        shared = tf.keras.layers.Dense(256, activation='relu')(shared)
        
        # 分別為兩個機器人生成動作值
        robot1_output = tf.keras.layers.Dense(50, activation='linear', name='robot1_output')(shared)
        robot2_output = tf.keras.layers.Dense(50, activation='linear', name='robot2_output')(shared)
        
        model = tf.keras.Model(
            inputs={
                'map_input': map_input, 
                'frontier_input': frontier_input,
                'robot1_pos_input': robot1_pos_input,
                'robot2_pos_input': robot2_pos_input
            }, 
            outputs={
                'robot1': robot1_output,
                'robot2': robot2_output
            }
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss={
                'robot1': 'mse',
                'robot2': 'mse'
            }
        )
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def predict(self, state, frontiers, robot1_pos, robot2_pos):
        return self.model.predict({
            'map_input': state,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos
        })
        
    def train_on_batch(self, states, frontiers, robot1_pos, robot2_pos, robot1_targets, robot2_targets):
        return self.model.train_on_batch(
            {
                'map_input': states, 
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos
            }, 
            {
                'robot1': robot1_targets,
                'robot2': robot2_targets
            }
        )
        
    def save(self, path):
        self.model.save(path)
        
    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.target_model = tf.keras.models.load_model(path)
