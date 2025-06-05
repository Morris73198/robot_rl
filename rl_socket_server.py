import socket
import json
import numpy as np
import os
import sys
import traceback

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
print(f"Model config: {MODEL_CONFIG}")

def preprocess_state(state):
    """預處理狀態，增強錯誤檢查"""
    try:
        # 1. 檢查並處理地圖
        if ("map" not in state) or (state["map"] is None):
            raise ValueError("state['map'] is None or missing")
        
        map_img = np.array(state["map"], dtype=np.float32)
        print(f"Original map shape: {map_img.shape}")
        
        # 處理地圖維度
        if map_img.shape == (84, 84):
            map_img = map_img[..., None]  # 添加通道維度
        elif len(map_img.shape) == 3 and map_img.shape[2] == 1:
            pass  # 已經是正確維度
        else:
            print(f"Warning: Unexpected map shape {map_img.shape}, trying to reshape...")
            map_img = map_img.reshape(84, 84, 1)
            
        if map_img.shape != (84, 84, 1):
            raise ValueError(f"Map shape is {map_img.shape}, expected (84, 84, 1)")
            
        map_img = np.expand_dims(map_img, 0)  # 添加batch維度: (1, 84, 84, 1)
        print(f"Processed map shape: {map_img.shape}")

        # 2. 處理 frontiers
        frontiers = state.get("frontiers", [])
        if not isinstance(frontiers, list):
            frontiers = []
        
        frontiers = np.array(frontiers, dtype=np.float32)
        print(f"Original frontiers shape: {frontiers.shape}, count: {len(frontiers) if len(frontiers.shape) > 0 else 0}")
        
        if len(frontiers.shape) == 1 or (len(frontiers.shape) == 2 and frontiers.shape[0] == 0):
            frontiers = np.zeros((0, 2), dtype=np.float32)
        elif frontiers.ndim == 1:
            frontiers = frontiers.reshape(-1, 2)
            
        # 確保frontiers是2D且第二維度是2
        if len(frontiers.shape) != 2 or frontiers.shape[1] != 2:
            print(f"Warning: Frontiers shape {frontiers.shape} invalid, using empty array")
            frontiers = np.zeros((0, 2), dtype=np.float32)
            
        max_frontiers = MODEL_CONFIG['max_frontiers']
        padded_frontiers = np.zeros((max_frontiers, 2), dtype=np.float32)
        n = min(len(frontiers), max_frontiers)
        if n > 0:
            padded_frontiers[:n] = frontiers[:n]
        frontiers = np.expand_dims(padded_frontiers, 0)  # (1, max_frontiers, 2)
        print(f"Processed frontiers shape: {frontiers.shape}, valid count: {n}")

        # 3. 處理 robot 位置
        robot1_pos = state.get("robot1_pose", [0.0, 0.0])
        robot2_pos = state.get("robot2_pose", [0.0, 0.0])
        
        if not isinstance(robot1_pos, (list, tuple, np.ndarray)) or len(robot1_pos) < 2:
            print(f"Warning: Invalid robot1_pos {robot1_pos}, using default [0.0, 0.0]")
            robot1_pos = [0.0, 0.0]
        if not isinstance(robot2_pos, (list, tuple, np.ndarray)) or len(robot2_pos) < 2:
            print(f"Warning: Invalid robot2_pos {robot2_pos}, using default [0.0, 0.0]") 
            robot2_pos = [0.0, 0.0]
            
        robot1_pos = np.array(robot1_pos[:2], dtype=np.float32)
        robot2_pos = np.array(robot2_pos[:2], dtype=np.float32)
        robot1_pos = np.expand_dims(robot1_pos, 0)  # (1, 2)
        robot2_pos = np.expand_dims(robot2_pos, 0)  # (1, 2)
        print(f"Robot positions - robot1: {robot1_pos[0]}, robot2: {robot2_pos[0]}")

        # 4. 處理目標（可選，設為預設值）
        robot1_target = np.array(state.get("robot1_target", [0.0, 0.0])[:2], dtype=np.float32)
        robot2_target = np.array(state.get("robot2_target", [0.0, 0.0])[:2], dtype=np.float32)
        robot1_target = np.expand_dims(robot1_target, 0)
        robot2_target = np.expand_dims(robot2_target, 0)

        return map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target, n
        
    except Exception as e:
        print(f"預處理狀態時發生錯誤: {e}")
        traceback.print_exc()
        raise

def get_best_frontier(state, robot_name="robot1"):
    """獲取最佳frontier點"""
    try:
        print(f"\n=== 為 {robot_name} 尋找最佳frontier ===")
        
        # 預處理狀態
        map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target, valid_frontiers = preprocess_state(state)
        
        if valid_frontiers == 0:
            print("沒有有效的frontier點")
            return [0.0, 0.0]  # 回傳預設點而不是None
        
        print(f"開始模型預測，有效frontier數量: {valid_frontiers}")
        
        # 模型預測
        preds = model.predict(map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target)
        print(f"模型預測完成，輸出結構: {type(preds)}")
        
        if isinstance(preds, dict):
            if robot_name not in preds:
                print(f"警告: 模型輸出中沒有 {robot_name} 的結果")
                robot_name = list(preds.keys())[0]  # 使用第一個可用的
                print(f"使用 {robot_name} 的結果")
            
            qvals = preds[robot_name][0, :valid_frontiers]
        else:
            # 如果preds不是字典，假設它是numpy array
            print(f"模型輸出不是字典，形狀: {preds.shape}")
            qvals = preds[0, :valid_frontiers]
        
        print(f"Q值: {qvals}")
        
        # 選擇最佳行動
        best_idx = int(np.argmax(qvals))
        best_point = frontiers[0][best_idx].tolist()
        
        print(f"選擇的frontier索引: {best_idx}, 座標: {best_point}")
        return best_point
        
    except Exception as e:
        print(f"獲取最佳frontier時發生錯誤: {e}")
        traceback.print_exc()
        return [0.0, 0.0]  # 回傳預設點

def receive_complete_data(conn):
    """接收完整的資料"""
    all_data = b''
    while True:
        try:
            chunk = conn.recv(4096)
            if not chunk:
                break
            all_data += chunk
            
            # 嘗試解析JSON，看是否收到完整資料
            try:
                data = json.loads(all_data.decode('utf-8'))
                return data  # 成功解析，回傳資料
            except json.JSONDecodeError:
                # 資料不完整，繼續接收
                continue
                
        except socket.timeout:
            print("接收資料超時")
            break
        except Exception as e:
            print(f"接收資料時發生錯誤: {e}")
            break
    
    # 如果到這裡，表示接收失敗
    try:
        return json.loads(all_data.decode('utf-8'))
    except:
        return None

def main():
    host = "0.0.0.0"
    port = 9000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(5)
    print(f"robot_rl socket server listening on {host}:{port} ...")
    
    try:
        while True:
            conn, addr = s.accept()
            print(f"\n新連接來自 {addr}")
            
            try:
                conn.settimeout(10.0)  # 設定超時
                
                # 接收完整資料
                state = receive_complete_data(conn)
                
                if state is None:
                    print("接收資料失敗")
                    conn.sendall(json.dumps({"target_point": [0.0, 0.0], "msg": "接收資料失敗"}).encode())
                    conn.close()
                    continue
                
                print("收到 state 摘要:")
                print(f"  - map: {'存在' if state.get('map') else '缺失'}")
                print(f"  - frontiers: {len(state.get('frontiers', []))} 個")
                print(f"  - robot1_pose: {state.get('robot1_pose', '缺失')}")
                print(f"  - robot2_pose: {state.get('robot2_pose', '缺失')}")
                
                # 檢查關鍵資料
                if ("map" not in state) or (state["map"] is None):
                    print("收到 state['map'] is None，回傳預設點")
                    conn.sendall(json.dumps({"target_point": [0.0, 0.0], "msg": "map is None"}).encode())
                    conn.close()
                    continue
                
                if not state.get("frontiers"):
                    print("沒有frontier點，回傳預設點")
                    conn.sendall(json.dumps({"target_point": [0.0, 0.0], "msg": "no frontiers"}).encode())
                    conn.close()
                    continue
                
                # 推理 robot1
                best_point = get_best_frontier(state, robot_name="robot1")
                
                # 回傳結果
                resp = {"target_point": best_point}
                response_data = json.dumps(resp, ensure_ascii=False).encode('utf-8')
                conn.sendall(response_data)
                print(f"回傳 target: {resp['target_point']}")
                
            except Exception as e:
                print(f"處理請求時發生錯誤: {e}")
                traceback.print_exc()
                try:
                    error_resp = {"target_point": [0.0, 0.0], "msg": f"伺服器錯誤: {str(e)}"}
                    conn.sendall(json.dumps(error_resp).encode())
                except:
                    pass
            finally:
                try:
                    conn.close()
                except:
                    pass
                    
    except KeyboardInterrupt:
        print("\n收到中斷信號，正在關閉伺服器...")
    except Exception as e:
        print(f"伺服器發生嚴重錯誤: {e}")
        traceback.print_exc()
    finally:
        try:
            s.close()
        except:
            pass
        print("伺服器已關閉")

if __name__ == "__main__":
    main()