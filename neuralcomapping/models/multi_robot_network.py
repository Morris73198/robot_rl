import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers

class MLPBlock(layers.Layer):
    """Multi-layer perceptron block"""
    def __init__(self, channels, do_bn=False):
        super().__init__()
        self.layers = []
        n = len(channels)
        
        for i in range(1, n):
            self.layers.append(layers.Conv1D(
                channels[i], 
                kernel_size=1,
                padding='valid',
                use_bias=True))
                
            if i < (n-1):
                if do_bn:
                    self.layers.append(layers.BatchNormalization())
                self.layers.append(layers.ReLU())

    def call(self, x, training=False):
        for layer in self.layers:
            if isinstance(layer, layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

class Encoder(layers.Layer):
    """Joint encoding of state information using MLPs"""
    def __init__(self, feature_dim, hidden_layers):
        super().__init__()
        self.encoder = MLPBlock([4] + hidden_layers + [feature_dim])
        self.dist_encoder = MLPBlock([1, feature_dim, feature_dim])

    def call(self, inputs, dist, pos_history=None, goal_history=None, extras=None, training=False):
        # Process frontiers
        frontier_feats = self._process_frontiers(inputs)
        frontier_enc = self.encoder(frontier_feats, training=training)
        
        # Process distances
        dist_enc = self.dist_encoder(dist, training=training)
        
        # Process agents
        agent_feats = self._process_agents(inputs, extras)
        agent_enc = self.encoder(agent_feats, training=training)
        
        return frontier_enc, agent_enc, dist_enc
        
    def _process_frontiers(self, inputs):
        """Process frontier information from input tensor
        
        Args:
            inputs: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Tensor of shape [batch_size, num_frontiers, 4]
        """
        batch_size = tf.shape(inputs)[0]
        
        # Convert frontiers to features using tf.where
        frontiers = tf.where(tf.equal(inputs[:, 1], 1))  # Get frontier coordinates
        
        # Separate batch indices and coordinates
        batch_indices = tf.cast(frontiers[:, 0], tf.int64)
        y_coords = tf.cast(frontiers[:, 1], tf.float32)
        x_coords = tf.cast(frontiers[:, 2], tf.float32)
        
        # Create feature tensor
        num_frontiers = tf.shape(frontiers)[0]
        features = tf.zeros([batch_size, num_frontiers, 4], dtype=tf.float32)
        
        # Create indices for scatter update
        scatter_indices = tf.stack([
            batch_indices,
            tf.cast(tf.range(num_frontiers), tf.int64)
        ], axis=1)
        
        # Create updates tensor with position and zero features
        updates = tf.concat([
            tf.stack([x_coords, y_coords], axis=1),
            tf.zeros([num_frontiers, 2], dtype=tf.float32)
        ], axis=1)
        
        # Perform scatter update
        features = tf.tensor_scatter_nd_update(
            features,
            scatter_indices,
            updates
        )
        
        return features
        
    def _process_agents(self, inputs, extras):
        """Process agent information
        
        Args:
            inputs: Input tensor
            extras: Extra features tensor containing agent information
            
        Returns:
            Tensor of shape [batch_size, num_agents, 4]
        """
        batch_size = tf.shape(inputs)[0]
        num_agents = tf.shape(extras)[1]
        
        # Create agent feature tensor
        features = tf.zeros([batch_size, num_agents, 4], dtype=tf.float32)
        
        # Create indices for each batch and agent with consistent types
        batch_indices = tf.reshape(
            tf.cast(tf.tile(tf.range(batch_size)[:, None], [1, num_agents]), tf.int64),
            [-1]
        )
        agent_indices = tf.tile(tf.range(num_agents, dtype=tf.int64)[None, :], [batch_size, 1])
        agent_indices = tf.reshape(agent_indices, [-1])
        
        # Stack indices for scatter update
        scatter_indices = tf.stack([batch_indices, agent_indices], axis=1)
        
        # Reshape extras for updates
        updates = tf.reshape(extras, [-1, 4])
        
        # Perform scatter update
        features = tf.tensor_scatter_nd_update(
            features,
            scatter_indices,
            updates
        )
        
        return features

class AttentionModule(layers.Layer):
    """Multi-head attention module"""
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        assert feature_dim % num_heads == 0
        self.depth = feature_dim // num_heads
        
        self.wq = layers.Dense(feature_dim)
        self.wk = layers.Dense(feature_dim) 
        self.wv = layers.Dense(feature_dim)
        
        self.dense = layers.Dense(feature_dim)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, query, key, value, mask=None, training=False):
        batch_size = tf.shape(query)[0]
        
        # Linear layers
        q = self.wq(query)  
        k = self.wk(key)
        v = self.wv(value)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
            
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_output = tf.matmul(attention_weights, v)
        
        # Reshape output
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.feature_dim))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights

class GNNLayer(layers.Layer):
    """Graph neural network layer"""
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.attention = AttentionModule(feature_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(feature_dim * 2, activation='relu'),
            layers.Dense(feature_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x, training=False, mask=None):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x, mask, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class MultiRobotNetwork(tf.keras.Model):
    def __init__(self, input_shape=(84, 84, 1), num_robots=2, feature_dim=128, num_gnn_layers=3):
        super().__init__()
        self.num_robots = num_robots
        self.feature_dim = feature_dim
        
        # Encoders
        self.encoder = Encoder(feature_dim, [32, 64, 128])
        
        # GNN layers
        self.gnn_layers = [GNNLayer(feature_dim) for _ in range(num_gnn_layers)]
        
        # Policy heads (one per robot)
        self.policy_heads = [
            tf.keras.Sequential([
                layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                layers.Dense(1)  # Output logits for each frontier
            ]) for _ in range(num_robots)
        ]
        
        # Value heads (one per robot)
        self.value_heads = [
            tf.keras.Sequential([
                layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
                layers.Dense(1)  # State value
            ]) for _ in range(num_robots)
        ]

    def call(self, inputs, training=False):
        states, frontiers, robot_poses, robot_targets = inputs
        
        # Encode observations
        frontier_enc, agent_enc, dist_enc = self.encoder(
            states, 
            frontiers,
            robot_poses,
            robot_targets,
            tf.concat([robot_poses, robot_targets], axis=-1),
            training=training
        )
        
        # Process through GNN layers
        x = frontier_enc
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, training=training)
            
        # Ensure x has proper shape for concatenation
        batch_size = tf.shape(states)[0]
        if tf.shape(x)[1] == 0:  # If no frontiers found
            x = tf.zeros([batch_size, 1, self.feature_dim])
            
        # Get policies and values for each robot
        policies = []
        values = []
        
        for i in range(self.num_robots):
            # Extract features for current robot
            agent_features = tf.expand_dims(agent_enc[:, i], 1)  # Add extra dimension to match shape
            robot_features = tf.concat([x, tf.tile(agent_features, [1, tf.shape(x)[1], 1])], axis=-1)
            
            # Get policy (one logit per frontier)
            policy = self.policy_heads[i](robot_features)  # Shape: [batch_size, num_frontiers, 1]
            policy = tf.squeeze(policy, axis=-1)  # Shape: [batch_size, num_frontiers]
            
            # Get value
            pooled_features = tf.reduce_mean(robot_features, axis=1)  # Global average pooling
            value = self.value_heads[i](pooled_features)
            
            policies.append(policy)
            values.append(value)
            
        return policies, values

    def save(self, filepath):
        """Save model weights"""
        self.save_weights(filepath)

    def load(self, filepath):
        """Load model weights"""
        self.load_weights(filepath)