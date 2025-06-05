import socket
import json
import numpy as np
import os
import sys

# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# === 修改這裡的 import 路徑以符合你的專案結構 ===
from two_robot_dueling_dqn_attention.models.multi_robot_network import MultiRobotNetworkModel
from two_robot_dueling_dqn_attention.config import MODEL_CONFIG, MODEL_DIR

# 載入模型
MODEL_PATH = os.path.join(MODEL_DIR, 'dueling.h5')
model = MultiRobotNetworkModel(
    input_shape=MODEL_CONFIG['input_shape'],
    max_frontiers=MODEL_CONFIG['max_frontiers']
)
model.load(MODEL_PATH)
print(f"Loaded DRL model from {MODEL_PATH}")

def preprocess_state(state):
    # 1. 檢查 map
    if ("map" not in state) or (state["map"] is None):
        raise ValueError("state['map'] is None")
    map_img = np.array(state["map"], dtype=np.float32)
    # 如果 map 維度不是 (84,84,1)，嘗試 reshape
    if map_img.shape == (84, 84):
        map_img = map_img[..., None]
    if map_img.shape != (84, 84, 1):
        raise ValueError(f"state['map'] shape is {map_img.shape}, expected (84, 84, 1)")
    map_img = np.expand_dims(map_img, 0)  # (1, 84, 84, 1)

    # 2. 處理 frontiers
    frontiers = np.array(state.get("frontiers", []), dtype=np.float32)
    if frontiers.ndim == 1:
        frontiers = frontiers.reshape(-1, 2)
    max_frontiers = MODEL_CONFIG['max_frontiers']
    padded_frontiers = np.zeros((max_frontiers, 2), dtype=np.float32)
    n = min(len(frontiers), max_frontiers)
    if n > 0:
        padded_frontiers[:n] = frontiers[:n]
    frontiers = np.expand_dims(padded_frontiers, 0)  # (1, max_frontiers, 2)

    # 3. 處理 robot 位置
    robot1_pos = np.array(state.get("robot1_pose", [0.0, 0.0]), dtype=np.float32)
    robot2_pos = np.array(state.get("robot2_pose", [0.0, 0.0]), dtype=np.float32)
    robot1_pos = np.expand_dims(robot1_pos, 0)
    robot2_pos = np.expand_dims(robot2_pos, 0)

    # 4. 處理目標（可選）
    robot1_target = np.array(state.get("robot1_target", [0.0, 0.0]), dtype=np.float32)
    robot2_target = np.array(state.get("robot2_target", [0.0, 0.0]), dtype=np.float32)
    robot1_target = np.expand_dims(robot1_target, 0)
    robot2_target = np.expand_dims(robot2_target, 0)

    return map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target

def get_best_frontier(state, robot_name="robot1"):
    map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target = preprocess_state(state)
    preds = model.predict(map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target)
    valid_frontiers = np.count_nonzero(np.any(frontiers[0] != 0, axis=1))
    if valid_frontiers == 0:
        return None
    qvals = preds[robot_name][0, :valid_frontiers]
    best_idx = int(np.argmax(qvals))
    best_point = frontiers[0][best_idx].tolist()
    return best_point

def main():
    host = "0.0.0.0"
    port = 9000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(5)
    print(f"robot_rl socket server listening on {host}:{port} ...")
    while True:
        conn, addr = s.accept()
        all_data = b''
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            all_data += chunk
        try:
            state = json.loads(all_data.decode())
            print("收到 state:", {k: type(v).__name__ for k, v in state.items()})
            # 檢查 map
            if ("map" not in state) or (state["map"] is None):
                print("收到 state['map'] is None，回傳預設點")
                conn.sendall(json.dumps({"target_point": [0.0, 0.0], "msg": "map is None"}).encode())
                conn.close()
                continue
            # 推理 robot1
            best_point = get_best_frontier(state, robot_name="robot1")
            # 回傳
            resp = {"target_point": best_point if best_point else [0.0, 0.0]}
            conn.sendall(json.dumps(resp).encode())
            print(f"回傳 target: {resp['target_point']}")
        except Exception as e:
            print("JSON parse or inference error:", e)
            conn.sendall(json.dumps({"target_point": [0.0, 0.0], "msg": str(e)}).encode())
        conn.close()

if __name__ == "__main__":
    main()