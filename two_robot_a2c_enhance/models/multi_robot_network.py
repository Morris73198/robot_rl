import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers, optimizers
import json

# Helper layers
class LayerNormalization(layers.Layer):
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

class AdaptiveAttentionFusion(layers.Layer):
    """
    Implementation of the Adaptive Attention Fusion module as shown in image 1
    Fixed to handle dimension mismatches
    """
    def __init__(self, d_model=256, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        # Get feature dimensions from input shapes
        frontier_dim = input_shape[0][-1]
        cross_robot_dim = input_shape[1][-1]
        temporal_dim = input_shape[2][-1]
        map_dim = input_shape[3][-1]
        
        # Feature projection layers to ensure uniform dimensions
        # Project all features to d_model dimensions
        self.frontier_projection = layers.Dense(self.d_model)
        self.cross_robot_projection = layers.Dense(self.d_model)
        self.temporal_projection = layers.Dense(self.d_model)
        self.map_projection = layers.Dense(self.d_model)
        
        # Weight network for adaptive fusion
        self.weight_dense = layers.Dense(3, activation='softmax')
        self.layer_norm = LayerNormalization()
        
        super().build(input_shape)
        
    def call(self, inputs):
        # Inputs: [frontier_features, cross_robot_features, temporal_features, map_features]
        frontier_features, cross_robot_features, temporal_features, map_features = inputs
        
        # Project all features to the same dimension
        frontier_features_proj = self.frontier_projection(frontier_features)
        cross_robot_features_proj = self.cross_robot_projection(cross_robot_features)
        temporal_features_proj = self.temporal_projection(temporal_features)
        map_features_proj = self.map_projection(map_features)
        
        # Concatenate for weight calculation
        concat_features = tf.concat([frontier_features_proj, cross_robot_features_proj, temporal_features_proj], axis=-1)
        
        # Calculate adaptive weights
        weights = self.weight_dense(concat_features)  # [w1, w2, w3]
        
        # Extract individual weights
        w1 = tf.expand_dims(weights[:, 0], axis=-1)
        w2 = tf.expand_dims(weights[:, 1], axis=-1)
        w3 = tf.expand_dims(weights[:, 2], axis=-1)
        
        # Weighted features - all features now have same dimensions
        weighted_features = w1 * frontier_features_proj + w2 * cross_robot_features_proj + w3 * temporal_features_proj
        
        # Residual connection with map features
        output = weighted_features + map_features_proj
        
        # Normalization
        output = self.layer_norm(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model
        })
        return config

class TemporalAttentionModule(layers.Layer):
    """
    Implementation of the Temporal Attention module as shown in image 2
    """
    def __init__(self, d_model=256, memory_length=5, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.memory_length = memory_length
        
    def build(self, input_shape):
        # Memory tensor - will be built during the first call
        self.memory = None
        
        # Attention mechanism
        self.query_dense = layers.Dense(self.d_model)
        self.key_dense = layers.Dense(self.d_model)
        self.value_dense = layers.Dense(self.d_model)
        self.attention_dense = layers.Dense(self.d_model)
        
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        # Expand dimension if needed
        expanded_input = tf.expand_dims(inputs, axis=1) if len(tf.shape(inputs)) == 2 else inputs
        
        batch_size = tf.shape(expanded_input)[0]
        
        # Initialize memory if it's None or batch size changed
        if self.memory is None or tf.shape(self.memory)[0] != batch_size:
            self.memory = tf.zeros((batch_size, self.memory_length, self.d_model))
        
        if training:
            # Memory update (90% old + 10% new)
            # Keep the most recent memory entries and add the new one
            self.memory = tf.concat([self.memory[:, 1:], expanded_input], axis=1)
        
        # Attention mechanism
        q = self.query_dense(expanded_input)  # Query from current input
        k = self.key_dense(self.memory)       # Keys from memory
        v = self.value_dense(self.memory)     # Values from memory
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        # Final projection
        output = self.attention_dense(output)
        
        # Remove the extra dimension if input was 2D
        if len(tf.shape(inputs)) == 2:
            output = tf.squeeze(output, axis=1)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'memory_length': self.memory_length
        })
        return config

class PerceptionModule(layers.Layer):
    """
    Implementation of the Perception module as shown in image 3
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        # CNN blocks with different kernel sizes
        self.cnn_block3 = self._build_cnn_block(kernel_size=3, filters=32)
        self.cnn_block5 = self._build_cnn_block(kernel_size=5, filters=32)
        self.cnn_block7 = self._build_cnn_block(kernel_size=7, filters=32)
        
        # Spatial attention blocks
        self.spatial_attention3 = SpatialAttention()
        self.spatial_attention5 = SpatialAttention()
        self.spatial_attention7 = SpatialAttention()
        
        # Feature fusion
        self.fusion_conv = layers.Conv2D(32, 1, padding='same')
        
        # Max pooling
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))
        
        super().build(input_shape)
        
    def _build_cnn_block(self, kernel_size, filters):
        return layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal()
        )
        
    def call(self, inputs):
        # CNN blocks with different kernel sizes
        x3 = self.cnn_block3(inputs)
        x5 = self.cnn_block5(inputs)
        x7 = self.cnn_block7(inputs)
        
        # Spatial attention
        x3_att = self.spatial_attention3(x3)
        x5_att = self.spatial_attention5(x5)
        x7_att = self.spatial_attention7(x7)
        
        # Feature fusion (concatenate and 1x1 conv)
        concat = tf.concat([x3_att, x5_att, x7_att], axis=-1)
        fused = self.fusion_conv(concat)
        
        # Max pooling
        pooled = self.max_pool(fused)
        
        return pooled
    
    def get_config(self):
        return super().get_config()

class CrossRobotAttention(layers.Layer):
    """
    Implementation of the Cross-Robot Attention module as shown in image 3
    """
    def __init__(self, d_model=256, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        # R1 attending to R2
        self.r1_to_r2_dense = layers.Dense(self.d_model)
        self.r1_norm = LayerNormalization()
        
        # R2 attending to R1
        self.r2_to_r1_dense = layers.Dense(self.d_model)
        self.r2_norm = LayerNormalization()
        
        # Output projection
        self.output_dense = layers.Dense(self.d_model)
        
        super().build(input_shape)
        
    def call(self, inputs):
        # Robot 1 and Robot 2 features
        r1_features, r2_features = inputs
        
        # R1 attending to R2
        r1_attends_r2 = self.r1_to_r2_dense(r2_features)
        r1_attends_r2 = self.r1_norm(r1_attends_r2)
        
        # R2 attending to R1
        r2_attends_r1 = self.r2_to_r1_dense(r1_features)
        r2_attends_r1 = self.r2_norm(r2_attends_r1)
        
        # Concatenate attended features
        cross_features = tf.concat([r1_attends_r2, r2_attends_r1], axis=-1)
        
        # Final projection
        output = self.output_dense(cross_features)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model
        })
        return config

class FrontierAttentionLayer(layers.Layer):
    """
    Implementation of the Frontier Attention Layer as shown in image 3
    """
    def __init__(self, d_model=64, max_frontiers=50, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_frontiers = max_frontiers
        
    def build(self, input_shape):
        # Initial dense layer
        self.dense64 = layers.Dense(self.d_model, activation='relu')
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(self.max_frontiers, self.d_model)
        
        # Self-attention components
        self.query_dense = layers.Dense(self.d_model)
        self.key_dense = layers.Dense(self.d_model)
        self.value_dense = layers.Dense(self.d_model)
        self.attention_dense = layers.Dense(self.d_model)
        
        # Feed forward network
        self.ff_dense1 = layers.Dense(self.d_model * 4, activation='relu')
        self.ff_dense2 = layers.Dense(self.d_model)
        self.layer_norm = LayerNormalization()
        
        super().build(input_shape)
        
    def call(self, inputs):
        # Initial processing
        x = self.dense64(inputs)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Self-attention
        q = self.query_dense(x)
        k = self.key_dense(x)
        v = self.value_dense(x)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_output = tf.matmul(attention_weights, v)
        
        # Projection
        attention_output = self.attention_dense(attention_output)
        
        # Residual connection and normalization
        x = x + attention_output
        x = self.layer_norm(x)
        
        # Feed forward network
        ff_output = self.ff_dense1(x)
        ff_output = self.ff_dense2(ff_output)
        
        # Final residual connection and normalization
        output = x + ff_output
        output = self.layer_norm(output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'max_frontiers': self.max_frontiers
        })
        return config

class EnhancedMultiRobotA2CModel:
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        """Initialize the multi-robot A2C model with the architecture from the images
        
        Args:
            input_shape: Input map shape, default (84, 84, 1)
            max_frontiers: Maximum number of frontier points, default 50
        """
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.d_model = 256  # Model dimension
        self.num_heads = 8  # Attention heads
        self.dff = 512  # Feed forward network dimension
        self.dropout_rate = 0.1  # Dropout rate
        self.entropy_beta = 0.01  # Entropy regularization coefficient
        self.value_loss_weight = 0.5  # Value loss weight
        
        # Build separate actor and critic models
        self.actor = self._build_actor()
        self.critic = self._build_critic()
    
    def _build_perception_module(self, map_input):
        """Build the perception module as shown in image 3"""
        # CNN blocks with different kernel sizes
        conv3 = layers.Conv2D(32, 3, padding='same', activation='relu')(map_input)
        conv5 = layers.Conv2D(32, 5, padding='same', activation='relu')(map_input)
        conv7 = layers.Conv2D(32, 7, padding='same', activation='relu')(map_input)
        
        # Spatial attention
        # Simple implementation using average and max pooling
        def spatial_attention(x):
            avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
            max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
            concat = tf.concat([avg_pool, max_pool], axis=-1)
            attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
            return x * attention
        
        conv3_att = spatial_attention(conv3)
        conv5_att = spatial_attention(conv5)
        conv7_att = spatial_attention(conv7)
        
        # Feature fusion
        concat = tf.concat([conv3_att, conv5_att, conv7_att], axis=-1)
        fusion = layers.Conv2D(32, 1, padding='same', activation='relu')(concat)
        
        # Max pooling
        pooled = layers.MaxPooling2D(pool_size=(2, 2))(fusion)
        
        return pooled
    
    def _build_state_processing(self, robot1_pos, robot1_target, robot2_pos, robot2_target):
        """Build the state processing module as shown in image 3"""
        # Robot 1 state
        robot1_state = layers.Concatenate()([robot1_pos, robot1_target])
        robot1_features = layers.Dense(16, activation='relu')(robot1_state)
        
        # Robot 2 state
        robot2_state = layers.Concatenate()([robot2_pos, robot2_target])
        robot2_features = layers.Dense(16, activation='relu')(robot2_state)
        
        # Cross-robot attention - simplified implementation
        # Robot 1 attends to Robot 2
        r1_to_r2 = layers.Dense(self.d_model, activation='relu')(robot2_features)
        r1_to_r2 = layers.LayerNormalization()(r1_to_r2)
        
        # Robot 2 attends to Robot 1
        r2_to_r1 = layers.Dense(self.d_model, activation='relu')(robot1_features)
        r2_to_r1 = layers.LayerNormalization()(r2_to_r1)
        
        # Combine cross-robot features
        cross_robot_features = layers.Concatenate()([r1_to_r2, r2_to_r1])
        cross_robot_features = layers.Dense(self.d_model)(cross_robot_features)
        
        return robot1_features, robot2_features, cross_robot_features
    
    def _build_frontier_attention(self, frontier_input):
        """Build the frontier attention layer as shown in image 3"""
        # Initial dense layer
        x = layers.Dense(64, activation='relu')(frontier_input)
        
        # Add positional encoding - simplified
        # We'll use a learned positional embedding
        pos_embedding = layers.Dense(64)(frontier_input)
        x = x + pos_embedding
        
        # Self-attention mechanism - simplified
        query = layers.Dense(64)(x)
        key = layers.Dense(64)(x)
        value = layers.Dense(64)(x)
        
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(64, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention
        attention_output = tf.matmul(attention_weights, value)
        
        # Feed forward network
        ff_output = layers.Dense(256, activation='relu')(attention_output)
        ff_output = layers.Dense(64)(ff_output)
        
        # Final output with layer normalization
        output = layers.Add()([attention_output, ff_output])
        output = layers.LayerNormalization()(output)
        
        return output
    
    def _build_temporal_attention(self, features):
        """Build a simplified temporal attention module as shown in image 2"""
        # Since we can't maintain memory between calls easily in this implementation,
        # we'll use a simplified version that doesn't rely on persistent memory
        
        # Project features
        x = layers.Reshape((1, -1))(features)  # Add sequence dimension
        
        # Self-attention on this single timestep
        query = layers.Dense(self.d_model)(x)
        key = layers.Dense(self.d_model)(x)
        value = layers.Dense(self.d_model)(x)
        
        # Attention mechanism
        attention_output = layers.Attention()([query, value, key])
        
        # Final projection
        output = layers.Dense(self.d_model)(attention_output)
        output = layers.Reshape((-1,))(output)  # Remove sequence dimension
        
        return output
    
    def _build_adaptive_fusion(self, frontier_features, cross_robot_features, 
                              temporal_features, map_features):
        """Build a fixed version of the adaptive attention fusion module"""
        # Project all features to the same dimension
        frontier_proj = layers.Dense(self.d_model)(frontier_features)
        cross_robot_proj = layers.Dense(self.d_model)(cross_robot_features)
        temporal_proj = layers.Dense(self.d_model)(temporal_features)
        map_proj = layers.Dense(self.d_model)(map_features)
        
        # Concatenate for weight calculation
        concat = layers.Concatenate()([frontier_proj, cross_robot_proj, temporal_proj])
        
        # Calculate weights
        weights = layers.Dense(3, activation='softmax')(concat)
        
        # Extract weights
        w1 = tf.expand_dims(weights[:, 0], axis=-1)
        w2 = tf.expand_dims(weights[:, 1], axis=-1)
        w3 = tf.expand_dims(weights[:, 2], axis=-1)
        
        # Apply weights
        weighted_sum = (
            w1 * frontier_proj + 
            w2 * cross_robot_proj + 
            w3 * temporal_proj
        )
        
        # Residual connection
        output = layers.Add()([weighted_sum, map_proj])
        output = layers.LayerNormalization()(output)
        
        return output
    
    def _build_actor_network(self, fusion_features, name_prefix):
        """Build the actor network for each robot"""
        x = layers.Dense(128, activation='relu')(fusion_features)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Policy output
        logits = layers.Dense(self.max_frontiers, name=f"{name_prefix}_policy_logits")(x)
        policy = layers.Lambda(
            lambda x: tf.nn.softmax(x),
            name=f"{name_prefix}_policy"
        )(logits)
        
        return policy, logits
    
    def _build_critic_network(self, fusion_features, name_prefix):
        """Build the critic network for each robot"""
        x = layers.Dense(64, activation='relu')(fusion_features)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Value output
        value = layers.Dense(1, name=f"{name_prefix}_value")(x)
        
        return value
    
    def _build_actor(self):
        """Build the complete actor model with fixed dimensions"""
        # Input layers
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 1. Perception module
        map_features = self._build_perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features)
        
        # 2. State processing
        robot1_features, robot2_features, cross_robot_features = self._build_state_processing(
            robot1_pos, robot1_target, robot2_pos, robot2_target
        )
        
        # 3. Frontier attention
        frontier_features = self._build_frontier_attention(frontier_input)
        frontier_features_flat = layers.Flatten()(frontier_features)
        
        # 4. Temporal attention
        temporal_features = self._build_temporal_attention(frontier_features_flat)
        
        # 5. Adaptive attention fusion
        robot1_fusion = self._build_adaptive_fusion(
            frontier_features_flat, cross_robot_features, 
            temporal_features, map_features_flat
        )
        
        robot2_fusion = self._build_adaptive_fusion(
            frontier_features_flat, cross_robot_features, 
            temporal_features, map_features_flat
        )
        
        # 6. Actor network outputs
        robot1_policy, robot1_logits = self._build_actor_network(robot1_fusion, "robot1")
        robot2_policy, robot2_logits = self._build_actor_network(robot2_fusion, "robot2")
        
        # Create actor model
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
        
        # Set optimizer
        actor_model.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=True
        )
        
        return actor_model
    
    def _build_critic(self):
        """Build the complete critic model with fixed dimensions"""
        # Input layers
        map_input = layers.Input(shape=self.input_shape, name='map_input')
        frontier_input = layers.Input(shape=(self.max_frontiers, 2), name='frontier_input')
        robot1_pos = layers.Input(shape=(2,), name='robot1_pos_input')
        robot2_pos = layers.Input(shape=(2,), name='robot2_pos_input')
        robot1_target = layers.Input(shape=(2,), name='robot1_target_input')
        robot2_target = layers.Input(shape=(2,), name='robot2_target_input')
        
        # 1. Perception module
        map_features = self._build_perception_module(map_input)
        map_features_flat = layers.Flatten()(map_features)
        
        # 2. State processing
        robot1_features, robot2_features, cross_robot_features = self._build_state_processing(
            robot1_pos, robot1_target, robot2_pos, robot2_target
        )
        
        # 3. Frontier attention
        frontier_features = self._build_frontier_attention(frontier_input)
        frontier_features_flat = layers.Flatten()(frontier_features)
        
        # 4. Temporal attention
        temporal_features = self._build_temporal_attention(frontier_features_flat)
        
        # 5. Adaptive attention fusion
        robot1_fusion = self._build_adaptive_fusion(
            frontier_features_flat, cross_robot_features, 
            temporal_features, map_features_flat
        )
        
        robot2_fusion = self._build_adaptive_fusion(
            frontier_features_flat, cross_robot_features, 
            temporal_features, map_features_flat
        )
        
        # 6. Critic network outputs
        robot1_value = self._build_critic_network(robot1_fusion, "robot1")
        robot2_value = self._build_critic_network(robot2_fusion, "robot2")
        
        # Create critic model
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
        
        # Set optimizer and compile
        critic_model.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            amsgrad=True
        )
        critic_model.compile(optimizer=critic_model.optimizer, loss='mse')
        
        return critic_model
    
    def train_actor(self, states, frontiers, robot1_pos, robot2_pos, 
                   robot1_target, robot2_target, robot1_actions, robot2_actions, 
                   robot1_advantages, robot2_advantages, robot1_old_logits=None, 
                   robot2_old_logits=None, training_history=None, episode=0):
        """Train the actor network"""
        with tf.GradientTape() as tape:
            # Get policy predictions
            policy_dict = self.actor({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            # Apply epsilon smoothing to avoid extreme values
            epsilon = 1e-8
            robot1_policy_smoothed = (1 - epsilon) * policy_dict['robot1_policy'] + epsilon / self.max_frontiers
            robot2_policy_smoothed = (1 - epsilon) * policy_dict['robot2_policy'] + epsilon / self.max_frontiers
            
            # Calculate log probabilities
            actions_one_hot_1 = tf.one_hot(robot1_actions, self.max_frontiers)
            actions_one_hot_2 = tf.one_hot(robot2_actions, self.max_frontiers)
            
            log_prob_1 = tf.math.log(tf.reduce_sum(robot1_policy_smoothed * actions_one_hot_1, axis=1) + 1e-10)
            log_prob_2 = tf.math.log(tf.reduce_sum(robot2_policy_smoothed * actions_one_hot_2, axis=1) + 1e-10)
            
            # Calculate policy loss
            policy_loss_coef = 1.0
            
            robot1_loss = -tf.reduce_mean(log_prob_1 * robot1_advantages) * policy_loss_coef
            robot2_loss = -tf.reduce_mean(log_prob_2 * robot2_advantages) * policy_loss_coef
            
            # Add entropy regularization - get entropy coefficient dynamically
            entropy_coef = self.get_entropy_coefficient(training_history, episode)
            entropy_1 = -tf.reduce_mean(tf.reduce_sum(
                robot1_policy_smoothed * tf.math.log(robot1_policy_smoothed + 1e-10), 
                axis=1
            ))
            entropy_2 = -tf.reduce_mean(tf.reduce_sum(
                robot2_policy_smoothed * tf.math.log(robot2_policy_smoothed + 1e-10), 
                axis=1
            ))
            
            entropy_reward = entropy_coef * (entropy_1 + entropy_2)
            
            # PPO-style ratio limiting
            if robot1_old_logits is not None and robot2_old_logits is not None:
                # Calculate old policy log probabilities
                robot1_old_logits = tf.convert_to_tensor(robot1_old_logits, dtype=tf.float32)
                robot2_old_logits = tf.convert_to_tensor(robot2_old_logits, dtype=tf.float32)
                
                robot1_old_policy = tf.nn.softmax(robot1_old_logits, axis=-1)
                robot2_old_policy = tf.nn.softmax(robot2_old_logits, axis=-1)
                
                robot1_old_policy = (1 - epsilon) * robot1_old_policy + epsilon / self.max_frontiers
                robot2_old_policy = (1 - epsilon) * robot2_old_policy + epsilon / self.max_frontiers
                
                old_log_prob_1 = tf.math.log(tf.reduce_sum(robot1_old_policy * actions_one_hot_1, axis=1) + 1e-10)
                old_log_prob_2 = tf.math.log(tf.reduce_sum(robot2_old_policy * actions_one_hot_2, axis=1) + 1e-10)
                
                # Calculate probability ratios
                ratio_1 = tf.exp(log_prob_1 - old_log_prob_1)
                ratio_2 = tf.exp(log_prob_2 - old_log_prob_2)
                
                # Clip ratios
                clip_range = 0.2
                clipped_ratio_1 = tf.clip_by_value(ratio_1, 1 - clip_range, 1 + clip_range)
                clipped_ratio_2 = tf.clip_by_value(ratio_2, 1 - clip_range, 1 + clip_range)
                
                # Calculate clipped policy loss
                robot1_clipped_loss = -tf.reduce_mean(
                    tf.minimum(
                        ratio_1 * robot1_advantages,
                        clipped_ratio_1 * robot1_advantages
                    )
                ) * policy_loss_coef
                
                robot2_clipped_loss = -tf.reduce_mean(
                    tf.minimum(
                        ratio_2 * robot2_advantages,
                        clipped_ratio_2 * robot2_advantages
                    )
                ) * policy_loss_coef
                
                # Use clipped losses
                robot1_loss = robot1_clipped_loss
                robot2_loss = robot2_clipped_loss
            
            # Total policy loss
            policy_loss = robot1_loss + robot2_loss
            
            # Total loss = policy loss - entropy reward
            total_loss = policy_loss - entropy_reward
            
            # Add coordination loss to encourage robots to choose different actions
            # Calculate similarity between policy distributions
            similarity = tf.reduce_mean(
                tf.reduce_sum(
                    tf.sqrt(robot1_policy_smoothed * robot2_policy_smoothed), 
                    axis=1
                )
            )
            
            # Coordination coefficient
            coordination_coef = 0.1
            coordination_loss = coordination_coef * similarity
            
            # Add coordination loss to total loss
            total_loss += coordination_loss
        
        # Calculate gradients
        grads = tape.gradient(total_loss, self.actor.trainable_variables)
        
        # Gradient processing
        # Replace NaN gradients with zeros
        grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else g for g in grads]
        
        # Gradient clipping to avoid gradient explosion
        clipped_grads, grad_norm = tf.clip_by_global_norm(grads, 1.0)
        
        # Apply gradients
        self.actor.optimizer.apply_gradients(zip(clipped_grads, self.actor.trainable_variables))
        
        # Return metrics
        metrics = {
            'total_policy_loss': total_loss,
            'robot1_policy_loss': robot1_loss,
            'robot2_policy_loss': robot2_loss,
            'robot1_entropy': entropy_1,
            'robot2_entropy': entropy_2,
            'entropy_coef': entropy_coef,
            'coordination_loss': coordination_loss,
            'gradient_norm': grad_norm
        }
        
        return metrics
    
    def get_entropy_coefficient(self, training_history=None, episode=0):
        """Dynamically adjust entropy coefficient to balance exploration and exploitation"""
        # Default base entropy coefficient
        base_coef = 0.01
        
        # Use higher coefficient for early training to promote exploration
        if training_history is None or episode < 20:
            return 0.05
        
        # Calculate recent entropy values (if available)
        if ('robot1_entropy' in training_history and 
            len(training_history['robot1_entropy']) > 0):
            
            recent_entropy1 = np.mean(training_history['robot1_entropy'][-10:])
            recent_entropy2 = np.mean(training_history['robot2_entropy'][-10:])
            avg_entropy = (recent_entropy1 + recent_entropy2) / 2
            
            # Theoretical maximum entropy (uniform distribution)
            max_entropy = np.log(self.max_frontiers)
            
            # Calculate entropy ratio
            entropy_ratio = avg_entropy / max_entropy if max_entropy > 0 else 0
            
            # Adjust coefficient based on entropy ratio
            if entropy_ratio < 0.3:
                # Entropy too low, increase coefficient to promote exploration
                coef = base_coef * 2.0
            elif entropy_ratio > 0.7:
                # Entropy too high, decrease coefficient to promote exploitation
                coef = base_coef * 0.5
            else:
                # Entropy in reasonable range
                coef = base_coef
                
        else:
            # No entropy history, use default value
            coef = base_coef
        
        # Periodically increase entropy coefficient to avoid local optima
        if episode % 100 == 0 and episode > 0:
            # Increase coefficient every 100 episodes
            coef = coef * 1.5
        
        # Limit coefficient range
        coef = np.clip(coef, 0.001, 0.1)
        
        return coef
    
    def train_critic(self, states, frontiers, robot1_pos, robot2_pos, 
                    robot1_target, robot2_target, robot1_returns, robot2_returns):
        """Train the critic network with improved gradient handling"""
        with tf.GradientTape() as tape:
            # Get value predictions
            values = self.critic({
                'map_input': states,
                'frontier_input': frontiers,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos,
                'robot1_target_input': robot1_target,
                'robot2_target_input': robot2_target
            }, training=True)
            
            # Calculate value losses using Huber loss to handle outliers
            robot1_value_loss = tf.keras.losses.Huber(delta=1.0)(
                robot1_returns, values['robot1_value'])
            
            robot2_value_loss = tf.keras.losses.Huber(delta=1.0)(
                robot2_returns, values['robot2_value'])
            
            # Total value loss
            value_loss = robot1_value_loss + robot2_value_loss
        
        # Calculate gradients
        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        
        # Gradient processing
        # Replace NaN gradients with zeros
        grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else g for g in grads]
        
        # Clip abnormal gradients
        clipped_grads = []
        for g in grads:
            if g is not None:
                # Detect abnormally large gradients
                norm = tf.norm(g)
                
                # Clip if norm is too large
                if norm > 1.0:
                    g = g * (1.0 / norm)
            clipped_grads.append(g)
        
        # Global gradient clipping - stricter for critic
        clipped_grads, _ = tf.clip_by_global_norm(clipped_grads, 0.5)
        
        # Apply gradients
        self.critic.optimizer.apply_gradients(zip(clipped_grads, self.critic.trainable_variables))
        
        return {
            'total_value_loss': value_loss,
            'robot1_value_loss': robot1_value_loss,
            'robot2_value_loss': robot2_value_loss
        }
    
    def train_batch(self, states, frontiers, robot1_pos, robot2_pos, 
                  robot1_target, robot2_target, robot1_actions, robot2_actions, 
                  robot1_advantages, robot2_advantages, robot1_returns, robot2_returns,
                  robot1_old_values, robot2_old_values, robot1_old_logits, robot2_old_logits,
                  training_history=None):
        """Train both actor and critic networks on a batch of data"""
        # Get current episode for entropy calculation
        current_episode = len(training_history['episode_rewards']) if training_history else 0
        
        # Train actor
        actor_metrics = self.train_actor(
            states, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target, robot1_actions, robot2_actions, 
            robot1_advantages, robot2_advantages, robot1_old_logits, robot2_old_logits,
            training_history, current_episode
        )
        
        # Train critic
        critic_metrics = self.train_critic(
            states, frontiers, robot1_pos, robot2_pos, 
            robot1_target, robot2_target, robot1_returns, robot2_returns
        )
        
        # Calculate entropy for monitoring
        policy_dict = self.actor.predict({
            'map_input': states,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        }, verbose=0)
        
        # Calculate entropy with epsilon smoothing
        epsilon = 1e-8
        robot1_policy = policy_dict['robot1_policy']
        robot2_policy = policy_dict['robot2_policy']
        
        robot1_policy_smoothed = (1 - epsilon) * robot1_policy + epsilon / self.max_frontiers
        robot2_policy_smoothed = (1 - epsilon) * robot2_policy + epsilon / self.max_frontiers
        
        robot1_entropy = -tf.reduce_mean(tf.reduce_sum(
            robot1_policy_smoothed * tf.math.log(robot1_policy_smoothed + 1e-10), 
            axis=1
        ))
        robot2_entropy = -tf.reduce_mean(tf.reduce_sum(
            robot2_policy_smoothed * tf.math.log(robot2_policy_smoothed + 1e-10), 
            axis=1
        ))
        
        # Combine all metrics
        return {
            'total_loss': actor_metrics['total_policy_loss'] + critic_metrics['total_value_loss'],
            'robot1_policy_loss': actor_metrics['robot1_policy_loss'],
            'robot2_policy_loss': actor_metrics['robot2_policy_loss'],
            'robot1_value_loss': critic_metrics['robot1_value_loss'],
            'robot2_value_loss': critic_metrics['robot2_value_loss'],
            'robot1_entropy': robot1_entropy,
            'robot2_entropy': robot2_entropy,
            'gradient_norm': actor_metrics.get('gradient_norm', 0.0)
        }
    
    def predict(self, state, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target):
        """Predict action distributions and state values"""
        # Ensure input shapes are correct
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
        
        # Build input dictionary
        inputs = {
            'map_input': state,
            'frontier_input': frontiers,
            'robot1_pos_input': robot1_pos,
            'robot2_pos_input': robot2_pos,
            'robot1_target_input': robot1_target,
            'robot2_target_input': robot2_target
        }
        
        # Get actor outputs
        actor_outputs = self.actor.predict(inputs, verbose=0)
        
        # Get critic outputs
        critic_outputs = self.critic.predict(inputs, verbose=0)
        
        # Validate policy outputs
        for key in ['robot1_policy', 'robot2_policy']:
            policy = actor_outputs[key]
            if np.any(np.isnan(policy)):
                print(f"Warning: {key} contains NaN values, replacing with uniform distribution")
                actor_outputs[key] = np.ones_like(policy) / policy.shape[-1]
            elif np.any(policy < 0) or np.any(policy > 1):
                print(f"Warning: {key} contains out-of-range values, clipping")
                actor_outputs[key] = np.clip(policy, 0, 1)
                # Renormalize
                actor_outputs[key] = actor_outputs[key] / np.sum(actor_outputs[key], axis=-1, keepdims=True)
        
        # Return prediction results
        return {
            'robot1_policy': actor_outputs['robot1_policy'],
            'robot2_policy': actor_outputs['robot2_policy'],
            'robot1_value': critic_outputs['robot1_value'],
            'robot2_value': critic_outputs['robot2_value'],
            'robot1_logits': actor_outputs['robot1_logits'],
            'robot2_logits': actor_outputs['robot2_logits']
        }
    
    def compute_returns_and_advantages(self, rewards, values, dones, next_value):
        """Compute returns and advantages using GAE (Generalized Advantage Estimation)"""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        gae = 0
        
        # Keep rewards scaled reasonably
        reward_scale = 1.0
        scaled_rewards = rewards * reward_scale
        
        # GAE parameters
        gamma = 0.99  # Discount factor
        lambda_param = 0.95  # GAE lambda parameter
        
        # Compute backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_val = values[t+1]
            
            # Calculate TD error
            delta = scaled_rewards[t] + gamma * next_val * next_non_terminal - values[t]
            
            # Calculate GAE
            gae = delta + gamma * lambda_param * next_non_terminal * gae
            
            # Store advantage and return
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Robust normalization
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        
        # Normalize advantages
        normalized_advantages = (advantages - adv_mean) / adv_std
        
        # Limit range to avoid extreme values
        max_value = 15.0
        normalized_advantages = np.clip(normalized_advantages, -max_value, max_value)
        
        return returns, normalized_advantages
    
    def save(self, filepath):
        """Save model as .h5 format"""
        # Save actor model
        actor_path = filepath + '_actor.h5'
        print(f"Saving actor model to: {actor_path}")
        self.actor.save(actor_path, save_format='h5')
        
        # Save critic model
        critic_path = filepath + '_critic.h5'
        print(f"Saving critic model to: {critic_path}")
        self.critic.save(critic_path, save_format='h5')
        
        # Save additional configuration
        config = {
            'input_shape': self.input_shape,
            'max_frontiers': self.max_frontiers,
            'd_model': self.d_model,
            'entropy_beta': self.entropy_beta,
            'value_loss_weight': self.value_loss_weight
        }
        
        with open(filepath + '_config.json', 'w') as f:
            json.dump(config, f)
        
        print(f"Model saved in .h5 format")
    
    def load(self, filepath):
        """Load model from .h5 format"""
        # Create custom objects dictionary
        custom_objects = {
            'LayerNormalization': LayerNormalization,
            'SpatialAttention': SpatialAttention,
            'PositionalEncoding': PositionalEncoding,
            'AdaptiveAttentionFusion': AdaptiveAttentionFusion,
            'TemporalAttentionModule': TemporalAttentionModule, 
            'PerceptionModule': PerceptionModule,
            'CrossRobotAttention': CrossRobotAttention,
            'FrontierAttentionLayer': FrontierAttentionLayer
        }
        
        try:
            # Load configuration
            with open(filepath + '_config.json', 'r') as f:
                config = json.load(f)
            
            # Update model configuration
            self.input_shape = tuple(config['input_shape'])
            self.max_frontiers = config['max_frontiers']
            self.d_model = config['d_model']
            self.entropy_beta = config['entropy_beta']
            self.value_loss_weight = config['value_loss_weight']
            
            # Load actor model
            actor_path = filepath + '_actor.h5'
            print(f"Loading actor model: {actor_path}")
            self.actor = tf.keras.models.load_model(
                actor_path,
                custom_objects=custom_objects
            )
            
            # Load critic model
            critic_path = filepath + '_critic.h5'
            print(f"Loading critic model: {critic_path}")
            self.critic = tf.keras.models.load_model(
                critic_path,
                custom_objects=custom_objects
            )
            
            # Set optimizers if not present
            if not hasattr(self.actor, 'optimizer') or self.actor.optimizer is None:
                self.actor.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=0.0001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-8,
                    amsgrad=True
                )
            
            if not hasattr(self.critic, 'optimizer') or self.critic.optimizer is None:
                self.critic.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=0.0001,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-8,
                    amsgrad=True
                )
                self.critic.compile(optimizer=self.critic.optimizer, loss='mse')
            
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False