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

def get_best_frontiers_for_both_robots(state):
    """為兩個機器人獲取不同的最佳frontier點"""
    try:
        print(f"\n=== 為兩個機器人尋找不同的最佳frontier ===")
        
        # 預處理狀態
        map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target, valid_frontiers = preprocess_state(state)
        
        if valid_frontiers == 0:
            print("沒有有效的frontier點")
            return {
                "robot1_target": [0.0, 0.0],
                "robot2_target": [0.0, 0.0]
            }
        
        print(f"開始模型預測，有效frontier數量: {valid_frontiers}")
        
        # 模型預測
        preds = model.predict(map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target)
        print(f"模型預測完成，輸出結構: {type(preds)}")
        
        results = {}
        
        if isinstance(preds, dict) and 'robot1' in preds and 'robot2' in preds:
            # 獲取兩個機器人的Q值
            robot1_qvals = preds['robot1'][0, :valid_frontiers]
            robot2_qvals = preds['robot2'][0, :valid_frontiers]
            
            print(f"robot1 Q值: {robot1_qvals}")
            print(f"robot2 Q值: {robot2_qvals}")
            
            # 智能分配策略：確保不同目標
            if valid_frontiers == 1:
                # 只有一個frontier，兩個機器人分配相同點
                best_point = frontiers[0][0].tolist()
                results["robot1_target"] = best_point
                results["robot2_target"] = best_point
                print(f"只有一個frontier，兩個機器人都分配: {best_point}")
                
            elif valid_frontiers >= 2:
                # 多個frontier，使用智能分配
                robot1_best_idx = int(np.argmax(robot1_qvals))
                robot2_best_idx = int(np.argmax(robot2_qvals))
                
                if robot1_best_idx != robot2_best_idx:
                    # 第一選擇不同，直接分配
                    results["robot1_target"] = frontiers[0][robot1_best_idx].tolist()
                    results["robot2_target"] = frontiers[0][robot2_best_idx].tolist()
                    print(f"不同第一選擇 - robot1:{robot1_best_idx}, robot2:{robot2_best_idx}")
                else:
                    # 第一選擇相同，需要智能分配
                    print(f"相同第一選擇: {robot1_best_idx}")
                    
                    # 比較Q值，高的拿第一選擇
                    if robot1_qvals[robot1_best_idx] >= robot2_qvals[robot2_best_idx]:
                        # robot1 獲得第一選擇
                        results["robot1_target"] = frontiers[0][robot1_best_idx].tolist()
                        # robot2 獲得第二選擇
                        robot2_sorted = np.argsort(robot2_qvals)[::-1]
                        robot2_second_idx = robot2_sorted[1] if len(robot2_sorted) > 1 else robot2_sorted[0]
                        results["robot2_target"] = frontiers[0][robot2_second_idx].tolist()
                        print(f"robot1獲得第一選擇:{robot1_best_idx}, robot2獲得第二選擇:{robot2_second_idx}")
                    else:
                        # robot2 獲得第一選擇
                        results["robot2_target"] = frontiers[0][robot2_best_idx].tolist()
                        # robot1 獲得第二選擇
                        robot1_sorted = np.argsort(robot1_qvals)[::-1]
                        robot1_second_idx = robot1_sorted[1] if len(robot1_sorted) > 1 else robot1_sorted[0]
                        results["robot1_target"] = frontiers[0][robot1_second_idx].tolist()
                        print(f"robot2獲得第一選擇:{robot2_best_idx}, robot1獲得第二選擇:{robot1_second_idx}")
                
        else:
            # 處理非標準格式輸出
            print(f"使用備用分配策略")
            if isinstance(preds, np.ndarray):
                if len(preds.shape) >= 2:
                    qvals = preds[0, :valid_frontiers]
                else:
                    qvals = preds[:valid_frontiers]
            else:
                qvals = np.random.rand(valid_frontiers)
            
            # 按Q值排序分配
            sorted_indices = np.argsort(qvals)[::-1]
            
            if valid_frontiers >= 2:
                results["robot1_target"] = frontiers[0][sorted_indices[0]].tolist()
                results["robot2_target"] = frontiers[0][sorted_indices[1]].tolist()
                print(f"備用策略 - robot1:{sorted_indices[0]}, robot2:{sorted_indices[1]}")
            else:
                best_point = frontiers[0][sorted_indices[0]].tolist()
                results["robot1_target"] = best_point
                results["robot2_target"] = best_point
                print(f"備用策略 - 相同點: {best_point}")
        
        return results
        
    except Exception as e:
        print(f"獲取最佳frontier時發生錯誤: {e}")
        traceback.print_exc()
        return {
            "robot1_target": [0.0, 0.0],
            "robot2_target": [0.0, 0.0]
        }

def get_best_frontier_single(state, robot_name="robot1"):
    """為單個機器人獲取最佳frontier點（保持向後兼容）"""
    try:
        print(f"\n=== 為 {robot_name} 尋找最佳frontier（單機器人模式）===")
        
        # 預處理狀態
        map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target, valid_frontiers = preprocess_state(state)
        
        if valid_frontiers == 0:
            print("沒有有效的frontier點")
            return [0.0, 0.0]
        
        print(f"開始模型預測，有效frontier數量: {valid_frontiers}")
        
        # 模型預測
        preds = model.predict(map_img, frontiers, robot1_pos, robot2_pos, robot1_target, robot2_target)
        print(f"模型預測完成，輸出結構: {type(preds)}")
        
        if isinstance(preds, dict):
            if robot_name in preds:
                qvals = preds[robot_name][0, :valid_frontiers]
            else:
                print(f"警告: 模型輸出中沒有 {robot_name} 的結果，使用第一個可用結果")
                available_keys = list(preds.keys())
                if available_keys:
                    qvals = preds[available_keys[0]][0, :valid_frontiers]
                else:
                    return [0.0, 0.0]
        else:
            # 假設是numpy array
            print(f"模型輸出不是字典，形狀: {preds.shape}")
            if len(preds.shape) >= 2:
                qvals = preds[0, :valid_frontiers]
            else:
                qvals = preds[:valid_frontiers]
        
        print(f"Q值: {qvals}")
        
        # 選擇最佳行動
        best_idx = int(np.argmax(qvals))
        best_point = frontiers[0][best_idx].tolist()
        
        print(f"選擇的frontier索引: {best_idx}, 座標: {best_point}")
        return best_point
        
    except Exception as e:
        print(f"獲取最佳frontier時發生錯誤: {e}")
        traceback.print_exc()
        return [0.0, 0.0]

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
    print(f"智能多機器人RL服務器啟動 - {host}:{port}")
    print("支援功能：避免分配相同目標、智能衝突解決")
    
    try:
        while True:
            conn, addr = s.accept()
            print(f"\n新連接來自 {addr}")
            
            try:
                conn.settimeout(10.0)
                
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
                print(f"  - request_robot: {state.get('request_robot', '未指定')}")
                
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
                
                # 判斷請求類型
                if "request_robot" in state:
                    # 單機器人請求
                    robot_name = state["request_robot"]
                    best_point = get_best_frontier_single(state, robot_name)
                    resp = {"target_point": best_point}
                    print(f"單機器人模式 - {robot_name}: {best_point}")
                else:
                    # 多機器人請求（新格式）
                    results = get_best_frontiers_for_both_robots(state)
                    resp = results
                    print(f"多機器人模式: {results}")
                
                # 回傳結果
                response_data = json.dumps(resp, ensure_ascii=False).encode('utf-8')
                conn.sendall(response_data)
                print(f"回傳結果: {resp}")
                
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