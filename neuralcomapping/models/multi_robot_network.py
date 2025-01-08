import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers

class MultiLayerPerceptron(layers.Layer):
    """Simple MLP block"""
    def __init__(self, layer_dims, activation='relu'):
        super().__init__()
        self.layers = []
        for dim in layer_dims[1:]:
            self.layers.append(layers.Dense(dim))
            if activation:
                self.layers.append(layers.Activation(activation))
        
    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x)
        return x

class GraphAttention(layers.Layer):
    """Graph attention for intra-graph operations"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # Adjust dimensions to match input size
        self.query = layers.Dense(feature_dim, use_bias=False)
        self.key = layers.Dense(feature_dim, use_bias=False)
        self.value = layers.Dense(feature_dim, use_bias=False)
        
    def build(self, input_shape):
        # Add explicit feature projection if needed
        input_dim = input_shape[-1]
        if input_dim != self.feature_dim:
            self.input_projection = layers.Dense(self.feature_dim)
        else:
            self.input_projection = None
        super().build(input_shape)
        
    def call(self, node_features, training=False):
        # Project input features if necessary
        if self.input_projection is not None:
            node_features = self.input_projection(node_features)
            
        # Compute Q, K, V
        q = self.query(node_features)  # [batch_size, num_nodes, feature_dim]
        k = self.key(node_features)    # [batch_size, num_nodes, feature_dim]
        v = self.value(node_features)  # [batch_size, num_nodes, feature_dim]
        
        # Compute attention weights
        attention = tf.matmul(q, k, transpose_b=True)  # [batch_size, num_nodes, num_nodes]
        attention_scale = tf.sqrt(tf.cast(self.feature_dim, tf.float32))
        attention = tf.nn.softmax(attention / attention_scale)
        
        # Apply attention to values
        output = tf.matmul(attention, v)  # [batch_size, num_nodes, feature_dim]
        
        return output, attention

class GraphNode(layers.Layer):
    """Graph node feature processor"""
    def __init__(self, feature_dim):
        super().__init__()
        self.mlp = MultiLayerPerceptron([feature_dim * 2, feature_dim])
        
    def call(self, node_feature, aggregated_message, training=False):
        combined = tf.concat([node_feature, aggregated_message], axis=-1)
        return self.mlp(combined)

class InterGraphOperation(layers.Layer):
    """Cross-graph operation between robot and frontier nodes"""
    def __init__(self, feature_dim):
        super().__init__()
        self.edge_mlp = MultiLayerPerceptron([feature_dim * 3, feature_dim, 1])
        self.node_processor = GraphNode(feature_dim)
        
    class InterGraphOperation(layers.Layer):
        """Cross-graph operation between robot and frontier nodes"""
        def __init__(self, feature_dim):
            super().__init__()
            self.edge_mlp = MultiLayerPerceptron([feature_dim * 3, feature_dim, 1])
            self.node_processor = GraphNode(feature_dim)
            
        def call(self, robot_features, frontier_features, geodesic_distances, training=False):
            batch_size = tf.shape(robot_features)[0]
            num_robots = tf.shape(robot_features)[1]
            num_frontiers = tf.shape(frontier_features)[1]
            feature_dim = robot_features.shape[-1]
            
            # Reshape features for cross attention
            robots_expanded = tf.reshape(robot_features, [batch_size, num_robots, 1, feature_dim])
            frontiers_expanded = tf.reshape(frontier_features, [batch_size, 1, num_frontiers, feature_dim])
            
            # Repeat features to match dimensions
            robots_tiled = tf.repeat(robots_expanded, repeats=num_frontiers, axis=2)
            frontiers_tiled = tf.repeat(frontiers_expanded, repeats=num_robots, axis=1)
            
            # Properly handle geodesic distances
            # Reshape and expand dimensions to match the feature dimensions
            distances_expanded = tf.expand_dims(geodesic_distances, -1)  # Add feature dimension
            distances_expanded = tf.tile(
                distances_expanded, 
                [1, 1, 1, feature_dim]
            )
            
            # Concatenate features with geodesic distance
            edge_inputs = tf.concat([robots_tiled, frontiers_tiled, distances_expanded], axis=-1)
            
            # Compute edge features (affinities)
            edge_features = self.edge_mlp(edge_inputs)
            edge_weights = tf.nn.softmax(tf.squeeze(edge_features, -1))
            
            # Update robot features
            robot_messages = tf.matmul(edge_weights, frontier_features)
            new_robot_features = self.node_processor(robot_features, robot_messages)
            
            return new_robot_features, edge_weights

class IntraGraphOperation(layers.Layer):
    """Internal graph operations for robot and frontier nodes"""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = GraphAttention(feature_dim)
        self.node_processor = GraphNode(feature_dim)
        
    def call(self, node_features, training=False):
        # Apply self-attention
        attention_output = self.attention(node_features, training=training)
        if isinstance(attention_output, tuple):
            attended_features, attention_weights = attention_output
        else:
            attended_features = attention_output
            attention_weights = None
        
        # Update node features
        new_features = self.node_processor(node_features, attended_features)
        
        return new_features, attention_weights

class MultiRobotNetwork(tf.keras.Model):
    def __init__(self, input_shape=(84, 84, 1), num_robots=2, feature_dim=128, num_gnn_layers=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_robots = num_robots
        self.num_gnn_layers = num_gnn_layers
        
        # CNN for processing occupancy maps
        self.cnn = tf.keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu', padding='same'),
            layers.GlobalAveragePooling2D()
        ])
        
        # Feature initialization
        self.robot_init = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(feature_dim // 2)
        ])
        
        # Frontier feature processing
        self.frontier_encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(feature_dim)
        ])
        
        # Occupancy feature projection
        self.occupancy_proj = layers.Dense(feature_dim // 2)
        
        # Attention layers
        self.robot_query = layers.Dense(feature_dim)
        self.frontier_query = layers.Dense(feature_dim)
        self.robot_key = layers.Dense(feature_dim)
        self.frontier_key = layers.Dense(feature_dim)
        self.robot_value = layers.Dense(feature_dim)
        self.frontier_value = layers.Dense(feature_dim)
        
        # Output projection
        self.output_proj = layers.Dense(feature_dim)
        
        # Value heads (one per robot)
        self.value_heads = []
        for _ in range(num_robots):
            head = tf.keras.Sequential([
                layers.Dense(64, activation='relu'),
                layers.Dense(1)
            ])
            self.value_heads.append(head)

    def compute_attention(self, query, key, value, scale=True):
        """Compute scaled dot-product attention"""
        # Compute attention scores
        attention = tf.matmul(query, key, transpose_b=True)
        
        if scale:
            d_k = tf.cast(tf.shape(key)[-1], tf.float32)
            attention = attention / tf.sqrt(d_k)
            
        # Apply softmax
        attention_weights = tf.nn.softmax(attention, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, value)
        
        return output, attention_weights

    def encode_frontiers(self, frontiers_batch):
        """Encode frontiers with proper reshaping"""
        batch_size = tf.shape(frontiers_batch)[0]
        num_frontiers = tf.shape(frontiers_batch)[2]
        
        # Reshape to process all frontiers at once
        frontiers_reshaped = tf.reshape(frontiers_batch, [-1, 2])  # Combine batch and frontier dims
        
        # Encode each frontier
        encoded_frontiers = self.frontier_encoder(frontiers_reshaped)
        
        # Reshape back to batch format
        return tf.reshape(encoded_frontiers, [batch_size, -1, num_frontiers, self.feature_dim])

    def call(self, inputs, training=False):
        states, frontiers, robot_poses, robot_targets = inputs
        batch_size = tf.shape(states)[0]
        
        # Process state through CNN
        state_features = self.cnn(states)
        
        # Process robot features
        robot_features = tf.map_fn(
            lambda x: self.robot_init(x),
            robot_poses,
            fn_output_signature=tf.float32
        )
        
        # Add occupancy information to robot features
        occupancy_features = self.occupancy_proj(state_features)
        occupancy_features = tf.expand_dims(occupancy_features, 1)
        occupancy_features = tf.tile(
            occupancy_features, [1, self.num_robots, 1]
        )
        
        # Combine robot and occupancy features
        robot_features = tf.concat([robot_features, occupancy_features], axis=-1)
        
        # Process frontiers
        frontier_features = self.encode_frontiers(frontiers)
        
        # Initialize GNN output
        final_affinity = None
        
        # Apply GNN layers
        for _ in range(self.num_gnn_layers):
            # Robot self-attention
            robot_q = self.robot_query(robot_features)
            robot_k = self.robot_key(robot_features)
            robot_v = self.robot_value(robot_features)
            robot_output, _ = self.compute_attention(robot_q, robot_k, robot_v)
            robot_features = robot_features + robot_output
            
            # Frontier self-attention (for each robot's frontiers)
            frontier_features_list = tf.unstack(frontier_features, axis=1)
            updated_frontier_features = []
            
            for robot_frontiers in frontier_features_list:
                frontier_q = self.frontier_query(robot_frontiers)
                frontier_k = self.frontier_key(robot_frontiers)
                frontier_v = self.frontier_value(robot_frontiers)
                frontier_output, _ = self.compute_attention(frontier_q, frontier_k, frontier_v)
                updated_frontier_features.append(robot_frontiers + frontier_output)
            
            frontier_features = tf.stack(updated_frontier_features, axis=1)
            
            # Cross attention between robots and their frontiers
            layer_affinity = []
            for i in range(self.num_robots):
                robot_feat = tf.expand_dims(robot_features[:, i], 1)
                robot_proj = self.output_proj(robot_feat)
                frontier_proj = self.output_proj(frontier_features[:, i])
                
                affinity = tf.matmul(robot_proj, frontier_proj, transpose_b=True)
                layer_affinity.append(affinity[:, 0])  # Remove extra dimension
            
            layer_affinity = tf.stack(layer_affinity, axis=1)
            
            if final_affinity is None:
                final_affinity = layer_affinity
            else:
                final_affinity = final_affinity + layer_affinity
        
        # Average across layers and compute probabilities
        final_affinity = final_affinity / tf.cast(self.num_gnn_layers, tf.float32)
        assignment_probs = tf.nn.softmax(final_affinity, axis=-1)
        
        # Compute values
        values = []
        value_features = tf.concat([
            state_features,
            tf.reduce_mean(robot_features, axis=1)
        ], axis=-1)
        
        for head in self.value_heads:
            values.append(head(value_features))
        
        return tf.unstack(assignment_probs, axis=1), values
        
    def save(self, filepath):
        self.save_weights(filepath)
        
    def load(self, filepath):
        self.load_weights(filepath)