import os
# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import numpy as np
import matplotlib
# 設置 matplotlib 後端為 Agg (無需顯示)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import spatial
import csv
import random
import time
import heapq
from two_robot_cnndqn_attention.environment.multi_robot_no_unknown import Robot
from two_robot_cnndqn_attention.config import MODEL_CONFIG, ROBOT_CONFIG
from two_robot_cnndqn_attention.environment.robot_local_map_tracker import RobotIndividualMapTracker

class MarketTask:
    """表示市場中的任務 (探索目標點)"""
    def __init__(self, position, id, owner_id, expected_info_gain=None):
        """
        初始化任務
        
        參數:
            position: 目標點位置 [x, y]
            id: 任務ID
            owner_id: 當前擁有該任務的機器人ID
            expected_info_gain: 預期信息收益，如果為None則需計算
        """
        self.position = np.array(position)
        self.id = id
        self.owner_id = owner_id
        self.expected_info_gain = expected_info_gain
        self.reservation_price = 0  # 保留價格 (賣家最低接受的價格)
        self.already_seen = False  # 標記該點是否已經被探索過

class MarketTask:
    """表示市場中的任務 (探索目標點)"""
    def __init__(self, position, id, owner_id, expected_info_gain=None):
        """
        初始化任務
        
        參數:
            position: 目標點位置 [x, y]
            id: 任務ID
            owner_id: 當前擁有該任務的機器人ID
            expected_info_gain: 預期信息收益，如果為None則需計算
        """
        self.position = np.array(position)
        self.id = id
        self.owner_id = owner_id
        self.expected_info_gain = expected_info_gain
        self.reservation_price = 0  # 保留價格 (賣家最低接受的價格)
        self.already_seen = False  # 標記該點是否已經被探索過

class MarketRobot:
    """市場架構中的機器人代理"""
    
    def __init__(self, robot, robot_id, op_exec=None, info_to_dist_weight=0.1):
        """
        初始化市場機器人
        
        參數:
            robot: 基礎機器人實例
            robot_id: 機器人ID
            op_exec: 操作員執行器 (如果需要)
            info_to_dist_weight: 將信息轉換為距離單位的權重
        """
        self.robot = robot
        self.id = robot_id
        self.op_exec = op_exec
        self.info_to_dist_weight = info_to_dist_weight
        
        # 任務相關
        self.tasks = []  # 當前任務列表 (計劃路徑)
        self.task_id_counter = 0  # 任務ID計數器
        self.current_task = None  # 當前執行的任務
        self.completed_tasks = []  # 已完成的任務
        
        # 通信與拍賣相關
        self.peers = []  # 其他機器人列表
        self.bid_alpha = 0.9  # 出價策略參數
        self.bid_timeout = 2.0  # 拍賣超時時間(秒)
        self.auction_markup = 0.1  # 保留價格的加價比例
        
        # 統計信息
        self.total_distance = 0  # 總行駛距離
        self.total_info_gain = 0  # 總信息收益
        self.exchanges = 0  # 任務交換次數
        self.auctions_won = 0  # 贏得的拍賣數
        self.auctions_lost = 0  # 失去的拍賣數
        
        # 探索終止條件
        self.exploration_complete = False
        
        # 創建初始任務
        self.generate_initial_tasks()
    
    def register_peer(self, peer):
        """註冊一個對等機器人"""
        if peer not in self.peers and peer.id != self.id:
            self.peers.append(peer)
    
    def generate_initial_tasks(self, num_tasks=10):
        """生成初始任務列表，只使用frontier點"""
        # 使用frontier策略生成目標點
        frontiers = self.robot.get_frontiers()
        
        if len(frontiers) == 0:
            print(f"警告: 機器人 {self.id} 沒有找到任何frontier點作為初始任務")
            return
        
        # 如果frontier不足，選擇可用的數量
        num_to_use = min(num_tasks, len(frontiers))
        selected_frontiers = frontiers[:num_to_use]
        
        for pos in selected_frontiers:
            task = MarketTask(pos, self.get_next_task_id(), self.id)
            # 計算預期信息收益
            task.expected_info_gain = self.estimate_info_gain(task.position)
            self.tasks.append(task)
        
        print(f"機器人 {self.id} 生成了 {len(self.tasks)} 個初始frontier任務")
        
        # 初始規劃最佳路徑
        self.plan_optimal_tour()
    
    def get_next_task_id(self):
        """獲取下一個任務ID"""
        task_id = f"{self.id}_{self.task_id_counter}"
        self.task_id_counter += 1
        return task_id
    
    def is_valid_goal(self, position):
        """檢查目標點是否有效 (在未知區域附近且不是障礙物)"""
        x, y = int(position[0]), int(position[1])
        
        # 確保位置在地圖範圍內
        map_shape = self.robot.op_map.shape
        if x < 0 or x >= map_shape[1] or y < 0 or y >= map_shape[0]:
            return False
        
        # 檢查是否是障礙物
        if self.robot.op_map[y, x] == 1:
            return False
        
        # 對於frontier點，我們假設它們已經通過frontier檢測算法驗證
        return True
    
    def generate_frontier_goals(self, num_goals=5):
        """生成基於frontier的目標點"""
        # 獲取當前frontier點
        frontiers = self.robot.get_frontiers()
        
        if len(frontiers) == 0:
            print(f"機器人 {self.id} 沒有找到任何frontier點")
            return []
        
        # 選擇最多num_goals個frontier點
        selected_frontiers = frontiers[:min(num_goals, len(frontiers))]
        
        new_tasks = []
        for pos in selected_frontiers:
            task = MarketTask(pos, self.get_next_task_id(), self.id)
            task.expected_info_gain = self.estimate_info_gain(task.position)
            new_tasks.append(task)
        
        print(f"機器人 {self.id} 生成了 {len(new_tasks)} 個新frontier任務")
        return new_tasks
    
    def plan_optimal_tour(self):
        """規劃最佳路徑 (使用貪婪插入啟發式)"""
        if not self.tasks:
            return
        
        # 獲取當前位置
        current_pos = self.robot.robot_position
        
        # 使用貪婪插入啟發式規劃路徑
        # 首先選擇最近的點作為起點
        best_task = None
        best_dist = float('inf')
        
        for task in self.tasks:
            dist = np.linalg.norm(task.position - current_pos)
            if dist < best_dist:
                best_dist = dist
                best_task = task
        
        # 如果沒有任務，直接返回
        if best_task is None:
            return
        
        # 開始構建路徑
        planned_tasks = [best_task]
        remaining_tasks = [t for t in self.tasks if t.id != best_task.id]
        
        # 貪婪插入剩餘任務
        while remaining_tasks:
            best_task = None
            best_position = -1
            best_extra_dist = float('inf')
            
            for task in remaining_tasks:
                # 嘗試將任務插入每個可能的位置
                for i in range(len(planned_tasks) + 1):
                    # 計算插入後的額外距離
                    if i == 0:
                        prev_pos = current_pos
                    else:
                        prev_pos = planned_tasks[i-1].position
                    
                    if i == len(planned_tasks):
                        next_pos = current_pos
                    else:
                        next_pos = planned_tasks[i].position
                    
                    # 計算不插入時的距離
                    orig_dist = np.linalg.norm(next_pos - prev_pos)
                    
                    # 計算插入後的距離
                    new_dist = (np.linalg.norm(task.position - prev_pos) + 
                               np.linalg.norm(next_pos - task.position))
                    
                    # 計算額外距離
                    extra_dist = new_dist - orig_dist
                    
                    if extra_dist < best_extra_dist:
                        best_extra_dist = extra_dist
                        best_task = task
                        best_position = i
            
            if best_task:
                planned_tasks.insert(best_position, best_task)
                remaining_tasks.remove(best_task)
        
        # 更新任務列表
        self.tasks = planned_tasks
    
    def estimate_info_gain(self, position):
        """估計在給定位置的信息收益"""
        # 獲取地圖形狀
        map_shape = self.robot.op_map.shape
        x, y = int(position[0]), int(position[1])
        
        # 計算感知範圍內的未知區域數量
        sensor_range = ROBOT_CONFIG['sensor_range']
        min_x = max(0, x - sensor_range)
        max_x = min(map_shape[1] - 1, x + sensor_range)
        min_y = max(0, y - sensor_range)
        max_y = min(map_shape[0] - 1, y + sensor_range)
        
        region = self.robot.op_map[min_y:max_y+1, min_x:max_x+1]
        unknown_count = np.sum(region == 127)
        
        return unknown_count
    
    def estimate_travel_cost(self, task_position):
        """估計到達目標點的旅行成本 (使用A*路徑規劃)"""
        try:
            # 使用機器人的A*路徑規劃
            path = self.robot.astar_path(
                self.robot.op_map,
                self.robot.robot_position.astype(np.int32),
                task_position.astype(np.int32),
                safety_distance=ROBOT_CONFIG['safety_distance']
            )
            
            if path is None:
                # 如果無法找到路徑，返回一個較大的成本
                return 1000.0
            
            # 計算路徑長度
            path_length = 0.0
            for i in range(1, path.shape[1]):
                path_length += np.linalg.norm(path[:, i] - path[:, i-1])
            
            return path_length
        except:
            # 如果路徑規劃失敗，返回一個較大的成本
            return 1000.0
    
    def calculate_task_profit(self, task, from_position=None):
        """計算執行任務的預期利潤"""
        try:
            if from_position is None:
                from_position = self.robot.robot_position
            
            # 如果任務已經被探索過，利潤為負
            if task.already_seen:
                return -100.0
            
            # 計算從當前位置到任務位置的成本
            direct_dist = np.linalg.norm(task.position - from_position)
            
            # 使用A*路徑規劃計算更準確的成本
            # 但如果距離很短，可以直接使用歐氏距離
            if direct_dist < 20:
                cost = direct_dist
            else:
                cost = self.estimate_travel_cost(task.position)
            
            # 如果沒有預先計算的信息收益，則計算
            if task.expected_info_gain is None:
                task.expected_info_gain = self.estimate_info_gain(task.position)
            
            # 計算利潤 = 收益 - 成本
            # 將收益轉換為與成本相同的單位
            revenue = task.expected_info_gain * self.info_to_dist_weight
            profit = revenue - cost
            
            return profit
        except Exception as e:
            print(f"計算任務 {task.id} 利潤時發生錯誤: {str(e)}")
            return -1.0  # 返回負數表示計算失敗
    
    def calculate_task_insertion_profit(self, task, task_list):
        """計算將任務插入現有計劃的利潤"""
        if not task_list:
            # 如果任務列表為空，直接計算從當前位置到任務位置的利潤
            profit = self.calculate_task_profit(task)
            return profit, 0  # 返回利潤和插入位置(0)
        
        # 找到最佳插入位置
        best_position = -1
        best_profit = float('-inf')
        
        # 嘗試所有可能的插入位置
        for i in range(len(task_list) + 1):
            # 計算插入前的路徑成本
            if i == 0:
                prev_pos = self.robot.robot_position
            else:
                prev_pos = task_list[i-1].position
            
            if i == len(task_list):
                next_pos = self.robot.robot_position  # 回到起點
            else:
                next_pos = task_list[i].position
            
            # 計算不插入時的成本
            orig_cost = np.linalg.norm(next_pos - prev_pos)
            
            # 計算插入後的成本
            new_cost = (np.linalg.norm(task.position - prev_pos) + 
                        np.linalg.norm(next_pos - task.position))
            
            # 計算插入的額外成本
            extra_cost = new_cost - orig_cost
            
            # 計算收益和利潤
            revenue = task.expected_info_gain * self.info_to_dist_weight
            insertion_profit = revenue - extra_cost
            
            if insertion_profit > best_profit:
                best_profit = insertion_profit
                best_position = i
        
        return best_profit, best_position
    
    def auction_task(self, task):
        """拍賣一個任務"""
        try:
            print(f"機器人 {self.id} 開始拍賣frontier任務 {task.id} 在位置 {task.position}")
            
            # 計算賣家的保留價格 (賣家對任務的估值加上一定比例)
            seller_valuation = self.calculate_task_profit(task)
            reservation_price = seller_valuation * (1.0 + self.auction_markup)
            task.reservation_price = reservation_price
            
            # 收集所有買家的出價
            bids = []
            
            for peer in self.peers:
                # 發送拍賣請求
                try:
                    bid = peer.bid_for_task(task, self, reservation_price)
                    if bid > reservation_price:
                        bids.append((bid, peer))
                except Exception as e:
                    print(f"與機器人 {peer.id} 通信出錯: {str(e)}")
            
            # 如果沒有買家出價高於保留價格，賣家保留任務
            if not bids:
                print(f"frontier任務 {task.id} 沒有收到有效出價，機器人 {self.id} 保留任務")
                return False
            
            # 找到最高出價
            highest_bid, highest_bidder = max(bids, key=lambda x: x[0])
            
            # 將任務轉讓給最高出價者
            print(f"機器人 {highest_bidder.id} 以 {highest_bid:.2f} 的價格贏得frontier任務 {task.id}")
            
            # 賣家給買家轉讓任務
            highest_bidder.accept_task(task, self)
            
            # 從賣家的任務列表中移除任務
            self.tasks = [t for t in self.tasks if t.id != task.id]
            
            # 記錄統計信息
            self.exchanges += 1
            
            return True
            
        except Exception as e:
            print(f"拍賣frontier任務 {task.id} 時發生錯誤: {str(e)}")
            return False
    
    def bid_for_task(self, task, auctioneer, reservation_price):
        """為任務出價"""
        try:
            # 確認這是一個frontier任務
            is_frontier = False
            frontiers = self.robot.get_frontiers()
            for frontier in frontiers:
                if np.linalg.norm(task.position - frontier) < 5:  # 允許一些誤差
                    is_frontier = True
                    break
            
            if not is_frontier:
                print(f"機器人 {self.id} 拒絕為非frontier任務 {task.id} 出價")
                return 0.0
            
            # 計算買家對任務的估值
            buyer_valuation_result = self.calculate_task_insertion_profit(task, self.tasks)
            
            # 確保結果是一個元組 (profit, position)
            if isinstance(buyer_valuation_result, tuple) and len(buyer_valuation_result) == 2:
                buyer_valuation, insert_position = buyer_valuation_result
            else:
                # 如果不是元組，可能只返回了利潤值，設置默認插入位置
                buyer_valuation = buyer_valuation_result
                insert_position = 0
            
            # 檢查該區域是否已經被探索過
            if task.already_seen or buyer_valuation <= 0:
                # 通知賣家該任務不值得拍賣
                auctioneer.task_already_explored(task)
                return 0.0
            
            # 如果買家估值高於保留價格，計算出價
            if buyer_valuation > reservation_price:
                # 使用論文中的出價策略公式
                bid = reservation_price + self.bid_alpha * (buyer_valuation - reservation_price)
                print(f"機器人 {self.id} 對frontier任務 {task.id} 出價 {bid:.2f}，估值為 {buyer_valuation:.2f}")
                
                # 記錄統計信息
                self.auctions_won += 1
                
                return bid
            else:
                # 估值低於保留價格，不出價
                return 0.0
        except Exception as e:
            print(f"機器人 {self.id} 為frontier任務 {task.id} 出價時發生錯誤: {str(e)}")
            return 0.0
    
    def accept_task(self, task, seller):
        """接受從拍賣中獲得的任務"""
        # 確認這是一個frontier任務
        is_frontier = False
        frontiers = self.robot.get_frontiers()
        for frontier in frontiers:
            if np.linalg.norm(task.position - frontier) < 5:  # 允許一些誤差
                is_frontier = True
                break
        
        if not is_frontier:
            print(f"機器人 {self.id} 拒絕接受非frontier任務 {task.id}")
            return
            
        # 更新任務擁有者
        task.owner_id = self.id
        
        # 將任務添加到任務列表
        self.tasks.append(task)
        
        # 重新規劃最佳路徑
        self.plan_optimal_tour()
        
        print(f"機器人 {self.id} 接受了frontier任務 {task.id} 從機器人 {seller.id}")
    
    def task_already_explored(self, task):
        """處理當被告知任務區域已經被探索"""
        print(f"機器人 {self.id} 被告知frontier任務 {task.id} 區域已經被探索過")
        task.already_seen = True
    
    def move_to_next_task(self):
        """移動到下一個任務點，確保只使用frontier任務"""
        try:
            if not self.tasks:
                # 如果沒有任務，生成新的frontier任務
                new_tasks = self.generate_frontier_goals(5)
                for task in new_tasks:
                    self.tasks.append(task)
                self.plan_optimal_tour()
                
                if not self.tasks:
                    return None, 0, False  # 仍然沒有任務
            
            # 獲取下一個任務
            next_task = self.tasks[0]
            self.current_task = next_task
            
            print(f"機器人 {self.id} 開始執行frontier任務 {next_task.id} 在 {next_task.position}")
            
            # 確保任務位置是整數陣列
            task_position = np.array(next_task.position, dtype=np.int32)
            
            # 使用機器人的移動到frontier功能
            next_state, reward, done = self.robot.move_to_frontier(task_position)
            
            # 記錄已行駛的距離
            moved_distance = np.linalg.norm(self.robot.robot_position - task_position)
            self.total_distance += moved_distance
            
            if done:
                # 任務完成或無法到達
                # 從任務列表中移除任務
                if self.tasks:  # 確保任務列表不為空
                    self.tasks.pop(0)
                    self.completed_tasks.append(next_task)
                
                # 計算信息收益
                info_gain = self.estimate_info_gain(task_position)
                self.total_info_gain += info_gain
                
                # 在任務點生成新的frontier目標點
                new_tasks = self.generate_frontier_goals(3)
                
                # 將新任務添加到任務列表
                for task in new_tasks:
                    self.tasks.append(task)
                
                # 重新規劃最佳路徑
                self.plan_optimal_tour()
                
                # 拍賣所有任務
                tasks_to_auction = list(self.tasks)  # 複製列表以避免在迭代中修改
                for task in tasks_to_auction:
                    if random.random() < 0.8:  # 80%的概率進行拍賣，避免過多拍賣
                        self.auction_task(task)
            
            return next_state, reward, done
            
        except Exception as e:
            print(f"執行frontier任務時發生錯誤: {str(e)}")
            # 嘗試恢復：從任務列表中移除當前任務
            if self.tasks:
                self.tasks.pop(0)
            return None, -1, True  # 返回負獎勵並結束當前執行
    
    def check_exploration_complete(self):
        """檢查探索是否完成"""
        # 檢查已探索區域比例
        exploration_ratio = np.sum(self.robot.op_map == 255) / np.sum(self.robot.global_map == 255)
        
        if exploration_ratio > ROBOT_CONFIG['finish_percent']:
            self.exploration_complete = True
            return True
        
        # 檢查是否還有frontier
        frontiers = self.robot.get_frontiers()
        if len(frontiers) == 0 and not self.tasks:
            self.exploration_complete = True
            return True
        
        return False
    
    def update_status(self):
        """更新機器人狀態和統計信息"""
        # 更新其他機器人的位置
        for peer in self.peers:
            self.robot.update_other_robot_pos(peer.robot.robot_position)
            peer.robot.update_other_robot_pos(self.robot.robot_position)
        
        # 更新地圖和探索統計
        self.check_exploration_complete()
        
        # 更新地圖共享
        self.share_map_info()
    
    def share_map_info(self, share_interval=100):
        """與其他機器人共享地圖信息"""
        # 每隔一段時間共享一次
        if random.randint(0, share_interval) != 0:
            return
        
        for peer in self.peers:
            try:
                # 只發送小部分地圖 (機器人周圍的區域)
                robot_pos = self.robot.robot_position
                map_shape = self.robot.op_map.shape
                
                # 定義要共享的區域範圍
                share_range = ROBOT_CONFIG['sensor_range'] * 2
                min_x = max(0, int(robot_pos[0]) - share_range)
                max_x = min(map_shape[1] - 1, int(robot_pos[0]) + share_range)
                min_y = max(0, int(robot_pos[1]) - share_range)
                max_y = min(map_shape[0] - 1, int(robot_pos[1]) + share_range)
                
                # 提取地圖區域
                map_section = self.robot.op_map[min_y:max_y+1, min_x:max_x+1].copy()
                
                # 發送地圖區域和位置信息
                peer.receive_map_info(map_section, (min_x, min_y), self.id)
            except Exception as e:
                print(f"與機器人 {peer.id} 共享地圖時出錯: {str(e)}")
    
    def receive_map_info(self, map_section, position, sender_id):
        """接收來自其他機器人的地圖信息"""
        min_x, min_y = position
        max_y, max_x = map_section.shape
        
        # 更新自己的地圖
        # 檢查每個單元格，只接受已知區域的信息 (值為1或255)
        for y in range(max_y):
            for x in range(max_x):
                map_y = min_y + y
                map_x = min_x + x
                
                # 確保坐標在地圖範圍內
                if (0 <= map_y < self.robot.op_map.shape[0] and 
                    0 <= map_x < self.robot.op_map.shape[1]):
                    
                    # 只接受已知區域的信息
                    if map_section[y, x] in [1, 255]:
                        # 如果自己的地圖該位置是未知的，則更新
                        if self.robot.op_map[map_y, map_x] == 127:
                            self.robot.op_map[map_y, map_x] = map_section[y, x]
                        # 如果信息衝突，優先相信自己的地圖
        
        # 更新任務列表中的預期信息收益
        for task in self.tasks:
            task.expected_info_gain = self.estimate_info_gain(task.position)
            
            # 檢查該任務是否在已探索區域
            x, y = int(task.position[0]), int(task.position[1])
            if (0 <= y < self.robot.op_map.shape[0] and 
                0 <= x < self.robot.op_map.shape[1] and
                self.robot.op_map[y, x] == 255):
                
                task.already_seen = True
    
    def get_statistics(self):
        """獲取統計信息"""
        return {
            "id": self.id,
            "total_distance": self.total_distance,
            "total_info_gain": self.total_info_gain,
            "exchanges": self.exchanges,
            "auctions_won": self.auctions_won,
            "auctions_lost": self.auctions_lost,
            "exploration_ratio": self.get_exploration_ratio(),
            "tasks_completed": len(self.completed_tasks),
            "tasks_remaining": len(self.tasks)
        }
    
    def get_exploration_ratio(self):
        """獲取探索比例"""
        return np.sum(self.robot.op_map == 255) / np.sum(self.robot.global_map == 255)

class MarketOperatorExecutive:
    """操作員執行器 (OpExec) - 代表用戶的代理"""
    
    def __init__(self):
        """初始化操作員執行器"""
        self.robots = []  # 註冊的機器人
        self.combined_map = None  # 合併的地圖
        self.map_size = None  # 地圖大小
    
    def register_robot(self, robot):
        """註冊一個機器人"""
        self.robots.append(robot)
        robot.op_exec = self
        
        # 更新地圖大小
        if self.map_size is None and robot.robot.op_map is not None:
            self.map_size = robot.robot.op_map.shape
            self.combined_map = np.ones(self.map_size) * 127
    
    def request_map(self):
        """請求所有機器人的地圖"""
        if not self.robots:
            return None
        
        # 初始化合併地圖
        if self.map_size is None:
            self.map_size = self.robots[0].robot.op_map.shape
        
        self.combined_map = np.ones(self.map_size) * 127
        
        # 合併所有機器人的地圖
        for robot in self.robots:
            # 檢查機器人是否可達 (模擬可能的通信中斷)
            if random.random() > 0.1:  # 90%的機率機器人可達
                robot_map = robot.robot.op_map
                
                # 將機器人地圖合併到總地圖中
                # 使用加權機制處理衝突 (+1 表示障礙物, -1 表示自由空間)
                for y in range(self.map_size[0]):
                    for x in range(self.map_size[1]):
                        if robot_map[y, x] == 1:  # 障礙物
                            if self.combined_map[y, x] == 127:  # 未知
                                self.combined_map[y, x] = 1
                            elif self.combined_map[y, x] == 255:  # 自由空間
                                # 衝突: 一個機器人認為是障礙物，另一個認為是自由空間
                                # 保守處理，標記為未知
                                self.combined_map[y, x] = 127
                        elif robot_map[y, x] == 255:  # 自由空間
                            if self.combined_map[y, x] == 127:  # 未知
                                self.combined_map[y, x] = 255
        
        return self.combined_map
    
    def get_statistics(self):
        """獲取所有機器人的統計信息"""
        if not self.robots:
            return {}
        
        combined_stats = {
            "total_distance": 0,
            "total_info_gain": 0,
            "total_exchanges": 0,
            "total_tasks_completed": 0,
            "exploration_ratio": 0,
            "robot_stats": []
        }
        
        for robot in self.robots:
            robot_stats = robot.get_statistics()
            combined_stats["total_distance"] += robot_stats["total_distance"]
            combined_stats["total_info_gain"] += robot_stats["total_info_gain"]
            combined_stats["total_exchanges"] += robot_stats["exchanges"]
            combined_stats["total_tasks_completed"] += robot_stats["tasks_completed"]
            combined_stats["robot_stats"].append(robot_stats)
        
        # 計算平均探索率
        if self.robots:
            combined_stats["exploration_ratio"] = sum(r.get_exploration_ratio() for r in self.robots) / len(self.robots)
        
        return combined_stats

def create_robots_with_custom_positions(map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
    """創建使用特定地圖檔案和自定義起始位置的機器人
    
    參數:
        map_file_path: 指定地圖檔案的完整路徑
        robot1_pos: 機器人1的起始位置 [x, y]，如果為None則使用預設位置
        robot2_pos: 機器人2的起始位置 [x, y]，如果為None則使用預設位置
        train: 是否處於訓練模式
        plot: 是否繪製可視化
    
    返回:
        robot1, robot2: 兩個共享環境的機器人實例
    """
    # 創建一個自定義的 Robot 類繼承原始的 Robot 類
    class CustomRobot(Robot):
        @classmethod
        def create_shared_robots_with_custom_setup(cls, map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
            """創建共享環境的機器人實例，使用指定的地圖檔案和起始位置"""
            print(f"使用指定地圖創建共享環境的機器人: {map_file_path}")
            if robot1_pos is not None:
                print(f"機器人1自定義起始位置: {robot1_pos}")
            if robot2_pos is not None:
                print(f"機器人2自定義起始位置: {robot2_pos}")
            
            # 創建第一個機器人，它會載入和初始化地圖
            robot1 = cls(0, train, plot, is_primary=True)
            
            # 載入地圖但不使用預設起始位置
            global_map, initial_positions = robot1.map_setup(map_file_path)
            robot1.global_map = global_map
            
            # 建立自由空間KD樹 (用於確保起始位置在可行區域)
            robot1.t = robot1.map_points(global_map)
            robot1.free_tree = spatial.KDTree(robot1.free_points(global_map).tolist())
            
            # 設置機器人位置
            if robot1_pos is not None and robot2_pos is not None:
                # 驗證位置是否有效 (在地圖範圍內且不是障礙物)
                robot1_pos = np.array(robot1_pos, dtype=np.int64)
                robot2_pos = np.array(robot2_pos, dtype=np.int64)
                
                # 確保位置在地圖範圍內
                map_height, map_width = global_map.shape
                robot1_pos[0] = np.clip(robot1_pos[0], 0, map_width-1)
                robot1_pos[1] = np.clip(robot1_pos[1], 0, map_height-1)
                robot2_pos[0] = np.clip(robot2_pos[0], 0, map_width-1)
                robot2_pos[1] = np.clip(robot2_pos[1], 0, map_height-1)
                
                # 確保位置不在障礙物上
                if global_map[robot1_pos[1], robot1_pos[0]] == 1:
                    print("警告: 機器人1的指定位置是障礙物，將移至最近的自由空間")
                    robot1_pos = robot1.nearest_free(robot1.free_tree, robot1_pos)
                    
                if global_map[robot2_pos[1], robot2_pos[0]] == 1:
                    print("警告: 機器人2的指定位置是障礙物，將移至最近的自由空間")
                    robot2_pos = robot1.nearest_free(robot1.free_tree, robot2_pos)
                
                # 使用自定義起始位置
                robot1.robot_position = robot1_pos
                robot1.other_robot_position = robot2_pos
            else:
                # 使用地圖中的預設位置或隨機選擇位置
                robot1.robot_position = initial_positions[0].astype(np.int64)
                robot1.other_robot_position = initial_positions[1].astype(np.int64)
            
            # 初始化其他屬性
            robot1.op_map = np.ones(global_map.shape) * 127
            robot1.map_size = np.shape(global_map)
            
            # 創建第二個機器人，共享第一個機器人的地圖和相關資源
            robot2 = cls(0, train, plot, is_primary=False, shared_env=robot1)
            
            robot1.other_robot = robot2
            
            # 重新初始化可視化
            if plot:
                # 清理舊的可視化
                if hasattr(robot1, 'fig'):
                    plt.close(robot1.fig)
                if hasattr(robot2, 'fig'):
                    plt.close(robot2.fig)
                
                # 初始化可視化
                robot1.initialize_visualization()
                robot2.initialize_visualization()
            
            return robot1, robot2
    
    # 使用自定義機器人類創建機器人
    return CustomRobot.create_shared_robots_with_custom_setup(
        map_file_path, 
        robot1_pos=robot1_pos, 
        robot2_pos=robot2_pos, 
        train=train, 
        plot=plot
    )

def save_plot(robot, step, output_path):
    """儲存單個機器人的繪圖
    
    參數:
        robot: 機器人實例
        step: 當前步驟數
        output_path: 儲存繪圖的路徑
    """
    plt.figure(figsize=(10, 10))
    robot.plot_env()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')  # 關閉所有圖形以釋放記憶體

def market_exploration(map_file_path, start_points_list, output_dir='results_market_based'):
    """基於市場架構的多機器人探索，只使用frontier點
    
    參數:
        map_file_path: 要使用的特定地圖檔案的路徑
        start_points_list: 包含多個起始點的列表，每個元素是 [robot1_pos, robot2_pos]
        output_dir: 輸出目錄
    """
    # 設定更詳細的異常捕獲
    import traceback
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 創建CSV檔案來記錄每個步驟的覆蓋率數據
    csv_path = os.path.join(output_dir, 'coverage_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['StartPoint', 'Step', 'Robot1Coverage', 'Robot2Coverage', 'IntersectionCoverage', 'UnionCoverage', 'TotalDistance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # 為每個起始點運行測試
    for start_idx, (robot1_pos, robot2_pos) in enumerate(start_points_list):
        print(f"\n===== 測試起始點 {start_idx+1}/{len(start_points_list)} =====")
        print(f"機器人1起始位置: {robot1_pos}")
        print(f"機器人2起始位置: {robot2_pos}")
        
        # 創建當前起始點的輸出目錄
        current_output_dir = os.path.join(output_dir, f'start_point_{start_idx+1}')
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        
        # 創建個人地圖目錄
        individual_maps_dir = os.path.join(current_output_dir, 'individual_maps')
        if not os.path.exists(individual_maps_dir):
            os.makedirs(individual_maps_dir)
        
        # 創建共享環境的機器人
        robot1, robot2 = create_robots_with_custom_positions(
            map_file_path,
            robot1_pos=robot1_pos,
            robot2_pos=robot2_pos,
            train=False,
            plot=True
        )
        
        # 創建地圖追蹤器
        tracker = RobotIndividualMapTracker(
            robot1, 
            robot2, 
            save_dir=individual_maps_dir
        )
        
        # 創建市場機器人代理
        market_robot1 = MarketRobot(robot1, 'R1', info_to_dist_weight=0.1)
        market_robot2 = MarketRobot(robot2, 'R2', info_to_dist_weight=0.1)
        
        # 創建操作員執行器
        op_exec = MarketOperatorExecutive()
        op_exec.register_robot(market_robot1)
        op_exec.register_robot(market_robot2)
        
        # 註冊對等機器人
        market_robot1.register_peer(market_robot2)
        market_robot2.register_peer(market_robot1)
        
        try:
            # 初始化環境
            state1 = robot1.begin()
            state2 = robot2.begin()
            
            # 開始追蹤個人地圖
            tracker.start_tracking()
            
            # 儲存初始狀態
            save_plot(robot1, 0, os.path.join(current_output_dir, 'robot1_step_0000.png'))
            save_plot(robot2, 0, os.path.join(current_output_dir, 'robot2_step_0000.png'))
            
            # 更新並儲存初始個人地圖
            tracker.update()
            tracker.save_current_maps(0)
            
            steps = 0
            intersection_data = []
            
            # 主探索循環
            while not (market_robot1.exploration_complete or market_robot2.exploration_complete):
                # 檢查是否還有frontier
                if not market_robot1.tasks and not market_robot2.tasks:
                    frontiers1 = market_robot1.robot.get_frontiers()
                    frontiers2 = market_robot2.robot.get_frontiers()
                    
                    if len(frontiers1) == 0 and len(frontiers2) == 0:
                        print("沒有剩餘frontier，探索完成")
                        break
                
                # 更新機器人狀態
                market_robot1.update_status()
                market_robot2.update_status()
                
                # 移動機器人1
                if market_robot1.tasks:
                    try:
                        next_state1, r1, d1 = market_robot1.move_to_next_task()
                    except Exception as e:
                        print(f"移動機器人1時發生錯誤: {str(e)}")
                        traceback.print_exc()
                        if market_robot1.tasks:
                            market_robot1.tasks.pop(0)  # 移除第一個任務
                else:
                    # 如果沒有任務，生成新的frontier任務
                    new_tasks1 = market_robot1.generate_frontier_goals(5)
                    for task in new_tasks1:
                        market_robot1.tasks.append(task)
                    market_robot1.plan_optimal_tour()
                
                # 移動機器人2
                if market_robot2.tasks:
                    try:
                        next_state2, r2, d2 = market_robot2.move_to_next_task()
                    except Exception as e:
                        print(f"移動機器人2時發生錯誤: {str(e)}")
                        traceback.print_exc()
                        if market_robot2.tasks:
                            market_robot2.tasks.pop(0)  # 移除第一個任務
                else:
                    # 如果沒有任務，生成新的frontier任務
                    new_tasks2 = market_robot2.generate_frontier_goals(5)
                    for task in new_tasks2:
                        market_robot2.tasks.append(task)
                    market_robot2.plan_optimal_tour()
                
                # 拍賣任務
                for robot in [market_robot1, market_robot2]:
                    if robot.tasks and steps % 5 == 0:  # 每5步進行一次拍賣
                        tasks_to_auction = list(robot.tasks)  # 複製列表以避免在迭代中修改
                        for task in tasks_to_auction:
                            try:
                                # 確認這是一個frontier任務
                                is_frontier = False
                                frontiers = robot.robot.get_frontiers()
                                for frontier in frontiers:
                                    if np.linalg.norm(task.position - frontier) < 5:  # 允許一些誤差
                                        is_frontier = True
                                        break
                                
                                if is_frontier:
                                    robot.auction_task(task)
                                else:
                                    print(f"跳過非frontier任務 {task.id} 的拍賣")
                                    # 從任務列表中移除非frontier任務
                                    robot.tasks = [t for t in robot.tasks if t.id != task.id]
                            except Exception as e:
                                print(f"機器人 {robot.id} 拍賣任務 {task.id} 時發生錯誤: {str(e)}")
                                traceback.print_exc()
                
                # 更新個人地圖追蹤器
                tracker.update()
                
                # 計算並記錄覆蓋率數據
                robot1_map = tracker.robot1_individual_map
                robot2_map = tracker.robot2_individual_map
                
                if robot1_map is not None and robot2_map is not None:
                    # 計算全局地圖的可探索區域總數
                    total_explorable = np.sum(robot1.global_map == 255)
                    
                    # 計算機器人1的覆蓋區域
                    robot1_explored = np.sum(robot1_map == 255)
                    robot1_coverage = robot1_explored / total_explorable if total_explorable > 0 else 0
                    
                    # 計算機器人2的覆蓋區域
                    robot2_explored = np.sum(robot2_map == 255)
                    robot2_coverage = robot2_explored / total_explorable if total_explorable > 0 else 0
                    
                    # 計算兩個機器人都探索過的區域（交集）
                    intersection = np.sum((robot1_map == 255) & (robot2_map == 255))
                    intersection_coverage = intersection / total_explorable if total_explorable > 0 else 0
                    
                    # 計算任一機器人探索過的區域（聯集）
                    union = np.sum((robot1_map == 255) | (robot2_map == 255))
                    union_coverage = union / total_explorable if total_explorable > 0 else 0
                    
                    # 計算總行駛距離
                    total_distance = market_robot1.total_distance + market_robot2.total_distance
                    
                    # 記錄數據
                    intersection_data.append({
                        'step': steps,
                        'robot1_coverage': robot1_coverage,
                        'robot2_coverage': robot2_coverage,
                        'intersection_coverage': intersection_coverage,
                        'union_coverage': union_coverage,
                        'total_distance': total_distance
                    })
                    
                    # 寫入CSV
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow({
                            'StartPoint': start_idx+1,
                            'Step': steps,
                            'Robot1Coverage': robot1_coverage,
                            'Robot2Coverage': robot2_coverage,
                            'IntersectionCoverage': intersection_coverage,
                            'UnionCoverage': union_coverage,
                            'TotalDistance': total_distance
                        })
                
                steps += 1
                
                # 每 10 步儲存繪圖和個人地圖
                if steps % 10 == 0:
                    save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_step_{steps:04d}.png'))
                    save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_step_{steps:04d}.png'))
                    
                    # 儲存當前個人地圖
                    tracker.save_current_maps(steps)
                    
                    # 獲取並顯示統計數據
                    stats = op_exec.get_statistics()
                    print(f"步數: {steps}, 聯合覆蓋率: {stats['exploration_ratio']:.1%}, 總距離: {stats['total_distance']:.1f}")
                
                # 檢查是否達到最大步數限制
                if steps >= 50000:
                    print("達到最大步數限制，終止探索")
                    break
            
            # 儲存最終狀態
            save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_final_step_{steps:04d}.png'))
            save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_final_step_{steps:04d}.png'))
            
            # 儲存最終個人地圖
            tracker.save_current_maps(steps)
            
            # 生成每個起始點的覆蓋率時間變化圖表
            plt.figure(figsize=(12, 8))
            steps_data = [data['step'] for data in intersection_data]
            robot1_coverage = [data['robot1_coverage'] for data in intersection_data]
            robot2_coverage = [data['robot2_coverage'] for data in intersection_data]
            intersection_coverage = [data['intersection_coverage'] for data in intersection_data]
            union_coverage = [data['union_coverage'] for data in intersection_data]
            
            # 繪製四條曲線
            plt.plot(steps_data, robot1_coverage, 'b-', linewidth=2, label='Robot 1')
            plt.plot(steps_data, robot2_coverage, 'r-', linewidth=2, label='Robot 2')
            plt.plot(steps_data, intersection_coverage, 'g-', linewidth=2, label='Intersection')
            plt.plot(steps_data, union_coverage, 'k-', linewidth=2, label='Union')
            
            # 添加標籤和標題
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Coverage', fontsize=14)
            plt.title(f'Time-Coverage Analysis - Market Based (Start Point {start_idx+1})', fontsize=16)
            
            # 添加網格和圖例
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            # 設置y軸範圍
            plt.ylim(0, 1.05)
            
            plt.savefig(os.path.join(current_output_dir, 'time_coverage_analysis.png'), dpi=300)
            plt.close()
            
            # 生成效率圖表
            plt.figure(figsize=(12, 8))
            efficiency = [data['union_coverage'] / data['total_distance'] if data['total_distance'] > 0 else 0 
                         for data in intersection_data]
            
            plt.plot(steps_data, efficiency, 'b-', linewidth=2)
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Efficiency (Coverage/Distance)', fontsize=14)
            plt.title(f'Exploration Efficiency - Market Based (Start Point {start_idx+1})', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.savefig(os.path.join(current_output_dir, 'efficiency_analysis.png'), dpi=300)
            plt.close()
            
            # 停止追蹤
            tracker.stop_tracking()
            
            # 清理資源
            tracker.cleanup()
            
            # 清理機器人資源
            for robot in [robot1, robot2]:
                if robot is not None and hasattr(robot, 'cleanup_visualization'):
                    robot.cleanup_visualization()
            
            print(f"完成起始點 {start_idx+1} 的測試，總步數: {steps}")
            
        except Exception as e:
            print(f"測試過程中出錯: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 確保清理資源
            try:
                if tracker is not None:
                    tracker.cleanup()
                
                for robot in [robot1, robot2]:
                    if robot is not None and hasattr(robot, 'cleanup_visualization'):
                        robot.cleanup_visualization()
            except:
                pass
    
    # 生成所有起始點的比較圖表
    try:
        # 從CSV檔案讀取所有數據
        all_data = {}
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                start_point = int(row['StartPoint'])
                step = int(row['Step'])
                
                if start_point not in all_data:
                    all_data[start_point] = {}
                    all_data[start_point]['steps'] = []
                    all_data[start_point]['robot1'] = []
                    all_data[start_point]['robot2'] = []
                    all_data[start_point]['intersection'] = []
                    all_data[start_point]['union'] = []
                    all_data[start_point]['total_distance'] = []
                
                all_data[start_point]['steps'].append(step)
                all_data[start_point]['robot1'].append(float(row['Robot1Coverage']))
                all_data[start_point]['robot2'].append(float(row['Robot2Coverage']))
                all_data[start_point]['intersection'].append(float(row['IntersectionCoverage']))
                all_data[start_point]['union'].append(float(row['UnionCoverage']))
                all_data[start_point]['total_distance'].append(float(row['TotalDistance']))
        
        # 為每個起始點創建單獨的全面覆蓋率圖表
        for start_point in sorted(all_data.keys()):
            plt.figure(figsize=(12, 8))
            
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['robot1'], 
                    'b-', linewidth=2, label='Robot 1')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['robot2'], 
                    'r-', linewidth=2, label='Robot 2')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['intersection'], 
                    'g-', linewidth=2, label='Intersection')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['union'], 
                    'k-', linewidth=2, label='Union')
            
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Coverage', fontsize=14)
            plt.title(f'Time-Coverage Analysis - Market Based (Start Point {start_point})', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.ylim(0, 1.05)
            
            plt.savefig(os.path.join(output_dir, f'time_coverage_startpoint_{start_point}.png'), dpi=300)
            plt.close()
        
        # 創建聯集覆蓋率比較圖表
        plt.figure(figsize=(12, 8))
        
        for start_point in sorted(all_data.keys()):
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['union'], 
                    linewidth=2, label=f'Start Point {start_point}')
        
        plt.xlabel('Time (steps)', fontsize=14)
        plt.ylabel('Total Coverage (Union)', fontsize=14)
        plt.title('Total Coverage Comparison - Market Based', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.ylim(0, 1.05)
        
        plt.savefig(os.path.join(output_dir, 'all_total_coverage_comparison.png'), dpi=300)
        plt.close()
        
        # 創建效率比較圖表
        plt.figure(figsize=(12, 8))
        
        for start_point in sorted(all_data.keys()):
            efficiency = [all_data[start_point]['union'][i] / all_data[start_point]['total_distance'][i] 
                         if all_data[start_point]['total_distance'][i] > 0 else 0 
                         for i in range(len(all_data[start_point]['steps']))]
            
            plt.plot(all_data[start_point]['steps'], efficiency, 
                    linewidth=2, label=f'Start Point {start_point}')
        
        plt.xlabel('Time (steps)', fontsize=14)
        plt.ylabel('Efficiency (Coverage/Distance)', fontsize=14)
        plt.title('Exploration Efficiency Comparison - Market Based', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        plt.savefig(os.path.join(output_dir, 'all_efficiency_comparison.png'), dpi=300)
        plt.close()
        
        print(f"已生成所有起始點的覆蓋率分析圖表")
        
    except Exception as e:
        print(f"生成比較圖表時出錯: {str(e)}")
    
    print(f"\n===== 完成所有起始點的測試 =====")
    print(f"結果儲存在: {output_dir}")
    print(f"覆蓋率數據儲存在: {csv_path}")
    print(f"測試完成！共生成了 {len(start_points_list)} 組不同起始點的時間-覆蓋率分析圖表")

def compare_with_random_strategy(map_file_path, start_points_list, output_dir='comparison_results'):
    """比較基於市場架構與隨機策略的探索效率
    
    參數:
        map_file_path: 要使用的特定地圖檔案的路徑
        start_points_list: 包含多個起始點的列表，每個元素是 [robot1_pos, robot2_pos]
        output_dir: 輸出目錄
    """
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 建立子目錄
    market_dir = os.path.join(output_dir, 'market_based')
    random_dir = os.path.join(output_dir, 'random_targets')
    
    if not os.path.exists(market_dir):
        os.makedirs(market_dir)
    if not os.path.exists(random_dir):
        os.makedirs(random_dir)
    
    # 執行市場策略實驗
    print("開始執行基於市場架構的探索實驗...")
    market_exploration(map_file_path, start_points_list, market_dir)
    
    # 執行隨機策略實驗 (使用test.py中的原始函數)
    print("開始執行隨機策略的探索實驗...")
    from test import test_multiple_start_points
    test_multiple_start_points(map_file_path, start_points_list, random_dir)
    
    # 比較兩種策略
    try:
        # 讀取兩個策略的CSV數據
        market_csv = os.path.join(market_dir, 'coverage_data.csv')
        random_csv = os.path.join(random_dir, 'coverage_data.csv')
        
        market_data = {}
        random_data = {}
        
        # 讀取市場策略數據
        with open(market_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                start_point = int(row['StartPoint'])
                step = int(row['Step'])
                
                if start_point not in market_data:
                    market_data[start_point] = {}
                    market_data[start_point]['steps'] = []
                    market_data[start_point]['union'] = []
                    market_data[start_point]['total_distance'] = []
                
                market_data[start_point]['steps'].append(step)
                market_data[start_point]['union'].append(float(row['UnionCoverage']))
                market_data[start_point]['total_distance'].append(float(row['TotalDistance']))
        
        # 讀取隨機策略數據
        with open(random_csv, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                start_point = int(row['StartPoint'])
                step = int(row['Step'])
                
                if start_point not in random_data:
                    random_data[start_point] = {}
                    random_data[start_point]['steps'] = []
                    random_data[start_point]['union'] = []
                    random_data[start_point]['total_distance'] = []
                
                random_data[start_point]['steps'].append(step)
                random_data[start_point]['union'].append(float(row['UnionCoverage']))
                random_data[start_point]['total_distance'].append(float(row.get('TotalDistance', '0')))
        
        # 生成覆蓋率比較圖表
        for start_point in sorted(set(market_data.keys()) & set(random_data.keys())):
            plt.figure(figsize=(12, 8))
            
            plt.plot(market_data[start_point]['steps'], market_data[start_point]['union'], 
                    'b-', linewidth=2, label='Market Based')
            plt.plot(random_data[start_point]['steps'], random_data[start_point]['union'], 
                    'r-', linewidth=2, label='Random Strategy')
            
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Total Coverage (Union)', fontsize=14)
            plt.title(f'Coverage Comparison - Start Point {start_point}', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.ylim(0, 1.05)
            
            plt.savefig(os.path.join(output_dir, f'coverage_comparison_startpoint_{start_point}.png'), dpi=300)
            plt.close()
        
        # 生成效率比較圖表
        for start_point in sorted(set(market_data.keys()) & set(random_data.keys())):
            plt.figure(figsize=(12, 8))
            
            # 計算效率 (覆蓋率/距離)
            market_efficiency = [market_data[start_point]['union'][i] / market_data[start_point]['total_distance'][i] 
                               if market_data[start_point]['total_distance'][i] > 0 else 0 
                               for i in range(len(market_data[start_point]['steps']))]
            
            random_efficiency = [random_data[start_point]['union'][i] / random_data[start_point]['total_distance'][i] 
                               if random_data[start_point]['total_distance'][i] > 0 else 0 
                               for i in range(len(random_data[start_point]['steps']))]
            
            plt.plot(market_data[start_point]['steps'], market_efficiency, 
                    'b-', linewidth=2, label='Market Based')
            plt.plot(random_data[start_point]['steps'], random_efficiency, 
                    'r-', linewidth=2, label='Random Strategy')
            
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Efficiency (Coverage/Distance)', fontsize=14)
            plt.title(f'Exploration Efficiency Comparison - Start Point {start_point}', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            plt.savefig(os.path.join(output_dir, f'efficiency_comparison_startpoint_{start_point}.png'), dpi=300)
            plt.close()
        
        # 計算平均覆蓋率和效率
        market_avg_coverage = []
        random_avg_coverage = []
        market_avg_efficiency = []
        random_avg_efficiency = []
        
        for start_point in sorted(set(market_data.keys()) & set(random_data.keys())):
            # 使用最後一個步驟的值作為最終覆蓋率
            market_avg_coverage.append(market_data[start_point]['union'][-1])
            random_avg_coverage.append(random_data[start_point]['union'][-1])
            
            # 計算平均效率
            m_eff = market_data[start_point]['union'][-1] / market_data[start_point]['total_distance'][-1] if market_data[start_point]['total_distance'][-1] > 0 else 0
            r_eff = random_data[start_point]['union'][-1] / random_data[start_point]['total_distance'][-1] if random_data[start_point]['total_distance'][-1] > 0 else 0
            
            market_avg_efficiency.append(m_eff)
            random_avg_efficiency.append(r_eff)
        
        # 計算總平均值
        market_coverage_avg = sum(market_avg_coverage) / len(market_avg_coverage) if market_avg_coverage else 0
        random_coverage_avg = sum(random_avg_coverage) / len(random_avg_coverage) if random_avg_coverage else 0
        
        market_efficiency_avg = sum(market_avg_efficiency) / len(market_avg_efficiency) if market_avg_efficiency else 0
        random_efficiency_avg = sum(random_avg_efficiency) / len(random_avg_efficiency) if random_avg_efficiency else 0
        
        # 繪製總體比較圖
        plt.figure(figsize=(10, 6))
        
        # 覆蓋率比較
        plt.subplot(1, 2, 1)
        plt.bar(['Market Based', 'Random Strategy'], [market_coverage_avg, random_coverage_avg], color=['blue', 'red'])
        plt.ylabel('Average Coverage')
        plt.title('Coverage Comparison')
        plt.ylim(0, 1.0)
        
        # 效率比較
        plt.subplot(1, 2, 2)
        plt.bar(['Market Based', 'Random Strategy'], [market_efficiency_avg, random_efficiency_avg], color=['blue', 'red'])
        plt.ylabel('Average Efficiency (Coverage/Distance)')
        plt.title('Efficiency Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=300)
        plt.close()
        
        # 創建比較結果CSV
        comparison_csv = os.path.join(output_dir, 'comparison_results.csv')
        with open(comparison_csv, 'w', newline='') as csvfile:
            fieldnames = ['StartPoint', 'MarketCoverage', 'RandomCoverage', 'MarketEfficiency', 'RandomEfficiency', 'CoverageImprovement', 'EfficiencyImprovement']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, start_point in enumerate(sorted(set(market_data.keys()) & set(random_data.keys()))):
                coverage_improvement = (market_avg_coverage[i] / random_avg_coverage[i] - 1) * 100 if random_avg_coverage[i] > 0 else 0
                efficiency_improvement = (market_avg_efficiency[i] / random_avg_efficiency[i] - 1) * 100 if random_avg_efficiency[i] > 0 else 0
                
                writer.writerow({
                    'StartPoint': start_point,
                    'MarketCoverage': market_avg_coverage[i],
                    'RandomCoverage': random_avg_coverage[i],
                    'MarketEfficiency': market_avg_efficiency[i],
                    'RandomEfficiency': random_avg_efficiency[i],
                    'CoverageImprovement': coverage_improvement,
                    'EfficiencyImprovement': efficiency_improvement
                })
        
        print(f"比較結果已保存至 {comparison_csv}")
        print(f"平均覆蓋率提升: {(market_coverage_avg / random_coverage_avg - 1) * 100:.2f}%")
        print(f"平均效率提升: {(market_efficiency_avg / random_efficiency_avg - 1) * 100:.2f}%")
        
    except Exception as e:
        print(f"生成比較結果時出錯: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # 指定地圖檔案路徑
    map_file_path = os.path.join(os.getcwd(), 'data', 'DungeonMaps', 'test', 'img_6032b.png')
    
    # 檢查地圖檔案是否存在
    if not os.path.exists(map_file_path):
        print(f"警告: 在 {map_file_path} 找不到指定的地圖檔案")
        print("請提供正確的地圖檔案路徑。")
        exit(1)
    
    # 定義起始點位置 [robot1_pos, robot2_pos]
    start_points = [
        [[100, 100], [100, 100]],  # 起始點 1
        [[520, 120], [520, 120]],  # 起始點 2
        [[630, 150], [630, 150]],   # 起始點 3
        [[250, 130], [250, 130]],   # 起始點 4
        [[250, 100], [250, 100]],  # 起始點 5
        [[400, 120], [400, 120]],  # 起始點 6
        [[140, 410], [140, 410]],   # 起始點 7
        [[110, 590], [110, 590]],   # 起始點 8
        [[900, 300], [90, 300]],   # 起始點 9
        [[260, 200], [260, 200]],  # 起始點 10
    ]
    
    # 選擇要執行的模式
    mode = input("請選擇執行模式: \n1 - 僅執行市場架構策略\n2 - 僅執行隨機策略\n3 - 執行兩種策略並比較\n請輸入 (1/2/3): ")
    
    if mode == '1':
        # 設置輸出目錄
        output_dir = 'results_market_based'
        print(f"執行基於市場架構的多機器人探索...")
        market_exploration(map_file_path, start_points, output_dir)
    elif mode == '2':
        # 設置輸出目錄
        output_dir = 'results_random_targets'
        print(f"執行隨機策略的多機器人探索...")
        from test import test_multiple_start_points
        test_multiple_start_points(map_file_path, start_points, output_dir)
    elif mode == '3':
        # 設置輸出目錄
        output_dir = 'comparison_results'
        print(f"執行兩種策略的比較實驗...")
        compare_with_random_strategy(map_file_path, start_points, output_dir)
    else:
        print("無效的選擇，請重新執行程序並選擇有效的模式。")

if __name__ == '__main__':
    main()