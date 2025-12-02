import os
import tensorflow as tf

# 建議加上這兩段，避免 OOM
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 如要用 CPU 可設 "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from two_robot_dueling_dqn_attention.models.multi_robot_network import MultiRobotNetworkModel

model = MultiRobotNetworkModel(
    input_shape=(84, 84, 1),
    max_frontiers=50
)
model.model.load_weights('dueling.h5')
model.model.save('dueling.keras')