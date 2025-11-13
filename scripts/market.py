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
        self.position = np.array(position)
        self.id = id
        self.owner_id = owner_id
        self.expected_info_gain = expected_info_gain
        self.reservation_price = 0
        self.already_seen = False

class MarketRobot:
    """市場架構中的機器人代理 - 優化版本"""
    
    def __init__(self, robot, robot_id, op_exec=None, info_to_dist_weight=0.1):
        self.robot = robot
        self.id = robot_id
        self.op_exec = op_exec
        self.info_to_dist_weight = info_to_dist_weight
        
        # 任務相關
        self.tasks = []
        self.task_id_counter = 0
        self.current_task = None
        self.completed_tasks = []
        
        # 通信與拍賣相關
        self.peers = []
        self.bid_alpha = 0.9
        self.bid_timeout = 2.0
        self.auction_markup = 0.1
        
        # 統計信息
        self.total_distance = 0
        self.total_info_gain = 0
        self.exchanges = 0
        self.auctions_won = 0
        self.auctions_lost = 0
        
        # 探索終止條件
        self.exploration_complete = False
        
        # 優化：緩存計算結果
        self.info_gain_cache = {}  # 緩存信息收益計算
        self.cache_valid_steps = 0  # 緩存有效的步數
        self.current_step = 0
        
        # 創建初始任務
        self.generate_initial_tasks()
    
    def register_peer(self, peer):
        """註冊一個對等機器人"""
        if peer not in self.peers and peer.id != self.id:
            self.peers.append(peer)
    
    def generate_initial_tasks(self, num_tasks=10):  # 從10減少到5
        """生成初始任務列表，只使用frontier點"""
        frontiers = self.robot.get_frontiers()
        
        if len(frontiers) == 0:
            print(f"警告: 機器人 {self.id} 沒有找到任何frontier點作為初始任務")
            return
        
        num_to_use = min(num_tasks, len(frontiers))
        selected_frontiers = frontiers[:num_to_use]
        
        for pos in selected_frontiers:
            task = MarketTask(pos, self.get_next_task_id(), self.id)
            task.expected_info_gain = self.estimate_info_gain(task.position)
            self.tasks.append(task)
        
        print(f"機器人 {self.id} 生成了 {len(self.tasks)} 個初始frontier任務")
        self.plan_optimal_tour()
    
    def get_next_task_id(self):
        """獲取下一個任務ID"""
        task_id = f"{self.id}_{self.task_id_counter}"
        self.task_id_counter += 1
        return task_id
    
    def generate_frontier_goals(self, num_goals=10):  # 從5減少到3
        """生成基於frontier的目標點"""
        frontiers = self.robot.get_frontiers()
        
        if len(frontiers) == 0:
            return []
        
        selected_frontiers = frontiers[:min(num_goals, len(frontiers))]
        
        new_tasks = []
        for pos in selected_frontiers:
            task = MarketTask(pos, self.get_next_task_id(), self.id)
            task.expected_info_gain = self.estimate_info_gain(task.position)
            new_tasks.append(task)
        
        return new_tasks
    
    def clean_invalid_tasks(self):
        """清理無效的任務（已探索的區域）- 優化版本"""
        if not self.tasks:
            return
        
        valid_tasks = []
        removed_count = 0
        
        # 批量處理，減少重複計算
        for task in self.tasks:
            if task.already_seen:
                removed_count += 1
                continue
            
            x, y = int(task.position[0]), int(task.position[1])
            
            if not (0 <= y < self.robot.op_map.shape[0] and 
                    0 <= x < self.robot.op_map.shape[1]):
                removed_count += 1
                continue
            
            # 使用較小的檢查範圍以加速
            sensor_range = 8  # 從10減少到8
            min_x = max(0, x - sensor_range)
            max_x = min(self.robot.op_map.shape[1] - 1, x + sensor_range)
            min_y = max(0, y - sensor_range)
            max_y = min(self.robot.op_map.shape[0] - 1, y + sensor_range)
            
            region = self.robot.op_map[min_y:max_y+1, min_x:max_x+1]
            unknown_count = np.sum(region == 127)
            
            if unknown_count > 8:  # 從10減少到8
                valid_tasks.append(task)
            else:
                removed_count += 1
                task.already_seen = True
        
        if removed_count > 0:
            print(f"機器人 {self.id} 清理了 {removed_count} 個無效任務")
        
        self.tasks = valid_tasks
    
    def plan_optimal_tour(self):
        """規劃最佳路徑 - 簡化版本"""
        self.clean_invalid_tasks()
        
        if not self.tasks:
            return
        
        # 簡化：只選擇最近的任務作為下一個目標
        current_pos = self.robot.robot_position
        
        # 計算所有任務的距離
        distances = []
        for task in self.tasks:
            dist = np.linalg.norm(task.position - current_pos)
            distances.append((dist, task))
        
        # 按距離排序
        distances.sort(key=lambda x: x[0])
        self.tasks = [task for _, task in distances]
    
    def estimate_info_gain(self, position):
        """估計在給定位置的信息收益 - 帶緩存"""
        # 使用緩存減少重複計算
        pos_key = (int(position[0]), int(position[1]))
        
        # 每10步清空一次緩存
        if self.current_step - self.cache_valid_steps > 10:
            self.info_gain_cache.clear()
            self.cache_valid_steps = self.current_step
        
        if pos_key in self.info_gain_cache:
            return self.info_gain_cache[pos_key]
        
        map_shape = self.robot.op_map.shape
        x, y = int(position[0]), int(position[1])
        
        sensor_range = ROBOT_CONFIG['sensor_range']
        min_x = max(0, x - sensor_range)
        max_x = min(map_shape[1] - 1, x + sensor_range)
        min_y = max(0, y - sensor_range)
        max_y = min(map_shape[0] - 1, y + sensor_range)
        
        region = self.robot.op_map[min_y:max_y+1, min_x:max_x+1]
        unknown_count = np.sum(region == 127)
        
        self.info_gain_cache[pos_key] = unknown_count
        return unknown_count
    
    def estimate_travel_cost(self, task_position):
        """估計到達目標點的旅行成本 - 簡化版本"""
        # 使用歐氏距離代替A*以加速
        return np.linalg.norm(task_position - self.robot.robot_position)
    
    def calculate_task_profit(self, task, from_position=None):
        """計算執行任務的預期利潤 - 簡化版本"""
        if from_position is None:
            from_position = self.robot.robot_position
        
        if task.already_seen:
            return -100.0
        
        # 直接使用歐氏距離，不使用A*
        cost = np.linalg.norm(task.position - from_position)
        
        if task.expected_info_gain is None:
            task.expected_info_gain = self.estimate_info_gain(task.position)
        
        revenue = task.expected_info_gain * self.info_to_dist_weight
        profit = revenue - cost
        
        return profit
    
    def calculate_task_insertion_profit(self, task, task_list):
        """計算將任務插入現有計劃的利潤 - 簡化版本"""
        if not task_list:
            profit = self.calculate_task_profit(task)
            return profit, 0
        
        # 簡化：只檢查插入到列表開頭
        current_pos = self.robot.robot_position
        first_task_pos = task_list[0].position
        
        # 計算不插入時的成本
        orig_cost = np.linalg.norm(first_task_pos - current_pos)
        
        # 計算插入後的成本
        new_cost = (np.linalg.norm(task.position - current_pos) + 
                    np.linalg.norm(first_task_pos - task.position))
        
        extra_cost = new_cost - orig_cost
        revenue = task.expected_info_gain * self.info_to_dist_weight
        insertion_profit = revenue - extra_cost
        
        return insertion_profit, 0
    
    def auction_task(self, task):
        """拍賣一個任務"""
        try:
            seller_valuation = self.calculate_task_profit(task)
            reservation_price = seller_valuation * (1.0 + self.auction_markup)
            task.reservation_price = reservation_price
            
            bids = []
            
            for peer in self.peers:
                try:
                    bid = peer.bid_for_task(task, self, reservation_price)
                    if bid > reservation_price:
                        bids.append((bid, peer))
                except Exception as e:
                    print(f"與機器人 {peer.id} 通信出錯: {str(e)}")
            
            if not bids:
                return False
            
            highest_bid, highest_bidder = max(bids, key=lambda x: x[0])
            
            highest_bidder.accept_task(task, self)
            self.tasks = [t for t in self.tasks if t.id != task.id]
            self.exchanges += 1
            
            return True
            
        except Exception as e:
            print(f"拍賣frontier任務 {task.id} 時發生錯誤: {str(e)}")
            return False
    
    def bid_for_task(self, task, auctioneer, reservation_price):
        """為任務出價 - 簡化檢查"""
        try:
            x, y = int(task.position[0]), int(task.position[1])
            if (0 <= y < self.robot.op_map.shape[0] and 
                0 <= x < self.robot.op_map.shape[1]):
                
                sensor_range = 8
                min_x = max(0, x - sensor_range)
                max_x = min(self.robot.op_map.shape[1] - 1, x + sensor_range)
                min_y = max(0, y - sensor_range)
                max_y = min(self.robot.op_map.shape[0] - 1, y + sensor_range)
                
                region = self.robot.op_map[min_y:max_y+1, min_x:max_x+1]
                unknown_count = np.sum(region == 127)
                
                if unknown_count < 8:
                    auctioneer.task_already_explored(task)
                    return 0.0
            
            # 簡化frontier檢查
            is_frontier = True  # 假設已經是frontier任務
            
            if not is_frontier:
                return 0.0
            
            buyer_valuation_result = self.calculate_task_insertion_profit(task, self.tasks)
            
            if isinstance(buyer_valuation_result, tuple) and len(buyer_valuation_result) == 2:
                buyer_valuation, insert_position = buyer_valuation_result
            else:
                buyer_valuation = buyer_valuation_result
                insert_position = 0
            
            if task.already_seen or buyer_valuation <= 0:
                auctioneer.task_already_explored(task)
                return 0.0
            
            if buyer_valuation > reservation_price:
                bid = reservation_price + self.bid_alpha * (buyer_valuation - reservation_price)
                self.auctions_won += 1
                return bid
            else:
                return 0.0
        except Exception as e:
            return 0.0
    
    def accept_task(self, task, seller):
        """接受從拍賣中獲得的任務"""
        x, y = int(task.position[0]), int(task.position[1])
        if (0 <= y < self.robot.op_map.shape[0] and 
            0 <= x < self.robot.op_map.shape[1]):
            
            sensor_range = 8
            min_x = max(0, x - sensor_range)
            max_x = min(self.robot.op_map.shape[1] - 1, x + sensor_range)
            min_y = max(0, y - sensor_range)
            max_y = min(self.robot.op_map.shape[0] - 1, y + sensor_range)
            
            region = self.robot.op_map[min_y:max_y+1, min_x:max_x+1]
            unknown_count = np.sum(region == 127)
            
            if unknown_count < 8:
                return
        
        task.owner_id = self.id
        self.tasks.append(task)
        self.plan_optimal_tour()
    
    def task_already_explored(self, task):
        """處理當被告知任務區域已經被探索"""
        task.already_seen = True
        self.tasks = [t for t in self.tasks if t.id != task.id]
    
    def move_to_next_task(self):
        """移動到下一個任務點"""
        try:
            self.current_step += 1
            self.clean_invalid_tasks()
            
            if not self.tasks:
                new_tasks = self.generate_frontier_goals(3)
                for task in new_tasks:
                    self.tasks.append(task)
                self.plan_optimal_tour()
                
                if not self.tasks:
                    return None, 0, False
            
            next_task = self.tasks[0]
            self.current_task = next_task
            
            task_position = np.array(next_task.position, dtype=np.int32)
            next_state, reward, done = self.robot.move_to_frontier(task_position)
            
            moved_distance = np.linalg.norm(self.robot.robot_position - task_position)
            self.total_distance += moved_distance
            
            if done:
                if self.tasks:
                    self.tasks.pop(0)
                    self.completed_tasks.append(next_task)
                
                info_gain = self.estimate_info_gain(task_position)
                self.total_info_gain += info_gain
                
                new_tasks = self.generate_frontier_goals(5)  # 從3減少到2
                for task in new_tasks:
                    self.tasks.append(task)
                
                self.plan_optimal_tour()
                
                # 降低拍賣頻率
                if random.random() < 0.5:  # 從0.5降到0.3
                    tasks_to_auction = list(self.tasks[:2])  # 只拍賣前2個任務
                    for task in tasks_to_auction:
                        self.auction_task(task)
            
            return next_state, reward, done
            
        except Exception as e:
            print(f"執行frontier任務時發生錯誤: {str(e)}")
            if self.tasks:
                self.tasks.pop(0)
            return None, -1, True
    
    def check_exploration_complete(self):
        """檢查探索是否完成"""
        exploration_ratio = np.sum(self.robot.op_map == 255) / np.sum(self.robot.global_map == 255)
        
        if exploration_ratio > ROBOT_CONFIG['finish_percent']:
            self.exploration_complete = True
            return True
        
        frontiers = self.robot.get_frontiers()
        if len(frontiers) == 0 and not self.tasks:
            self.exploration_complete = True
            return True
        
        return False
    
    def update_status(self):
        """更新機器人狀態和統計信息"""
        for peer in self.peers:
            self.robot.update_other_robot_pos(peer.robot.robot_position)
            peer.robot.update_other_robot_pos(self.robot.robot_position)
        
        self.check_exploration_complete()
        
        # 每3步才共享一次地圖（降低頻率）
        if self.current_step % 3 == 0:
            self.share_map_info()
    
    def share_map_info(self):
        """與其他機器人共享地圖信息 - 優化版本"""
        for peer in self.peers:
            try:
                robot_pos = self.robot.robot_position
                map_shape = self.robot.op_map.shape
                
                # 減小共享範圍以加速
                share_range = ROBOT_CONFIG['sensor_range']  # 從 *2 改為 *1
                min_x = max(0, int(robot_pos[0]) - share_range)
                max_x = min(map_shape[1] - 1, int(robot_pos[0]) + share_range)
                min_y = max(0, int(robot_pos[1]) - share_range)
                max_y = min(map_shape[0] - 1, int(robot_pos[1]) + share_range)
                
                map_section = self.robot.op_map[min_y:max_y+1, min_x:max_x+1].copy()
                peer.receive_map_info(map_section, (min_x, min_y), self.id)
            except Exception as e:
                pass  # 忽略錯誤以加速
    
    def receive_map_info(self, map_section, position, sender_id):
        """接收來自其他機器人的地圖信息 - 優化版本"""
        min_x, min_y = position
        max_y, max_x = map_section.shape
        
        # 批量更新地圖
        for y in range(max_y):
            for x in range(max_x):
                map_y = min_y + y
                map_x = min_x + x
                
                if (0 <= map_y < self.robot.op_map.shape[0] and 
                    0 <= map_x < self.robot.op_map.shape[1]):
                    
                    if map_section[y, x] in [1, 255]:
                        if self.robot.op_map[map_y, map_x] == 127:
                            self.robot.op_map[map_y, map_x] = map_section[y, x]
        
        # 每5步才清理一次任務
        if self.current_step % 5 == 0:
            self.clean_invalid_tasks()
    
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
        self.robots = []
        self.combined_map = None
        self.map_size = None
    
    def register_robot(self, robot):
        """註冊一個機器人"""
        self.robots.append(robot)
        robot.op_exec = self
        
        if self.map_size is None and robot.robot.op_map is not None:
            self.map_size = robot.robot.op_map.shape
            self.combined_map = np.ones(self.map_size) * 127
    
    def request_map(self):
        """請求所有機器人的地圖"""
        if not self.robots:
            return None
        
        if self.map_size is None:
            self.map_size = self.robots[0].robot.op_map.shape
        
        self.combined_map = np.ones(self.map_size) * 127
        
        for robot in self.robots:
            if random.random() > 0.1:
                robot_map = robot.robot.op_map
                
                for y in range(self.map_size[0]):
                    for x in range(self.map_size[1]):
                        if robot_map[y, x] == 1:
                            if self.combined_map[y, x] == 127:
                                self.combined_map[y, x] = 1
                            elif self.combined_map[y, x] == 255:
                                self.combined_map[y, x] = 127
                        elif robot_map[y, x] == 255:
                            if self.combined_map[y, x] == 127:
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
        
        if self.robots:
            combined_stats["exploration_ratio"] = sum(r.get_exploration_ratio() for r in self.robots) / len(self.robots)
        
        return combined_stats

def create_robots_with_custom_positions(map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
    """創建使用特定地圖檔案和自定義起始位置的機器人"""
    class CustomRobot(Robot):
        @classmethod
        def create_shared_robots_with_custom_setup(cls, map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
            print(f"使用指定地圖創建共享環境的機器人: {map_file_path}")
            
            robot1 = cls(0, train, plot, is_primary=True)
            global_map, initial_positions = robot1.map_setup(map_file_path)
            robot1.global_map = global_map
            robot1.t = robot1.map_points(global_map)
            robot1.free_tree = spatial.KDTree(robot1.free_points(global_map).tolist())
            
            if robot1_pos is not None and robot2_pos is not None:
                robot1_pos = np.array(robot1_pos, dtype=np.int64)
                robot2_pos = np.array(robot2_pos, dtype=np.int64)
                
                map_height, map_width = global_map.shape
                robot1_pos[0] = np.clip(robot1_pos[0], 0, map_width-1)
                robot1_pos[1] = np.clip(robot1_pos[1], 0, map_height-1)
                robot2_pos[0] = np.clip(robot2_pos[0], 0, map_width-1)
                robot2_pos[1] = np.clip(robot2_pos[1], 0, map_height-1)
                
                if global_map[robot1_pos[1], robot1_pos[0]] == 1:
                    robot1_pos = robot1.nearest_free(robot1.free_tree, robot1_pos)
                    
                if global_map[robot2_pos[1], robot2_pos[0]] == 1:
                    robot2_pos = robot1.nearest_free(robot1.free_tree, robot2_pos)
                
                robot1.robot_position = robot1_pos
                robot1.other_robot_position = robot2_pos
            else:
                robot1.robot_position = initial_positions[0].astype(np.int64)
                robot1.other_robot_position = initial_positions[1].astype(np.int64)
            
            robot1.op_map = np.ones(global_map.shape) * 127
            robot1.map_size = np.shape(global_map)
            
            robot2 = cls(0, train, plot, is_primary=False, shared_env=robot1)
            robot1.other_robot = robot2
            
            if plot:
                if hasattr(robot1, 'fig'):
                    plt.close(robot1.fig)
                if hasattr(robot2, 'fig'):
                    plt.close(robot2.fig)
                
                robot1.initialize_visualization()
                robot2.initialize_visualization()
            
            return robot1, robot2
    
    return CustomRobot.create_shared_robots_with_custom_setup(
        map_file_path, 
        robot1_pos=robot1_pos, 
        robot2_pos=robot2_pos, 
        train=train, 
        plot=plot
    )

def save_plot(robot, step, output_path):
    """儲存單個機器人的繪圖"""
    plt.figure(figsize=(10, 10))
    robot.plot_env()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')

def market_exploration(map_file_path, start_points_list, output_dir='results_market_based'):
    """基於市場架構的多機器人探索 - 優化版本"""
    import traceback
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_path = os.path.join(output_dir, 'coverage_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['StartPoint', 'Step', 'Robot1Coverage', 'Robot2Coverage', 'IntersectionCoverage', 'UnionCoverage', 'TotalDistance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    for start_idx, (robot1_pos, robot2_pos) in enumerate(start_points_list):
        print(f"\n===== 測試起始點 {start_idx+1}/{len(start_points_list)} =====")
        
        current_output_dir = os.path.join(output_dir, f'start_point_{start_idx+1}')
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        
        individual_maps_dir = os.path.join(current_output_dir, 'individual_maps')
        if not os.path.exists(individual_maps_dir):
            os.makedirs(individual_maps_dir)
        
        robot1, robot2 = create_robots_with_custom_positions(
            map_file_path,
            robot1_pos=robot1_pos,
            robot2_pos=robot2_pos,
            train=False,
            plot=True
        )
        
        tracker = RobotIndividualMapTracker(robot1, robot2, save_dir=individual_maps_dir)
        
        market_robot1 = MarketRobot(robot1, 'R1', info_to_dist_weight=0.1)
        market_robot2 = MarketRobot(robot2, 'R2', info_to_dist_weight=0.1)
        
        op_exec = MarketOperatorExecutive()
        op_exec.register_robot(market_robot1)
        op_exec.register_robot(market_robot2)
        
        market_robot1.register_peer(market_robot2)
        market_robot2.register_peer(market_robot1)
        
        try:
            state1 = robot1.begin()
            state2 = robot2.begin()
            
            tracker.start_tracking()
            
            # 只在開始和結束時保存圖片
            save_plot(robot1, 0, os.path.join(current_output_dir, 'robot1_step_0000.png'))
            save_plot(robot2, 0, os.path.join(current_output_dir, 'robot2_step_0000.png'))
            
            tracker.update()
            tracker.save_current_maps(0)
            
            steps = 0
            intersection_data = []
            
            while not (market_robot1.exploration_complete or market_robot2.exploration_complete):
                if not market_robot1.tasks and not market_robot2.tasks:
                    frontiers1 = market_robot1.robot.get_frontiers()
                    frontiers2 = market_robot2.robot.get_frontiers()
                    
                    if len(frontiers1) == 0 and len(frontiers2) == 0:
                        print("沒有剩餘frontier，探索完成")
                        break
                
                market_robot1.update_status()
                market_robot2.update_status()
                
                if market_robot1.tasks:
                    try:
                        next_state1, r1, d1 = market_robot1.move_to_next_task()
                    except Exception as e:
                        if market_robot1.tasks:
                            market_robot1.tasks.pop(0)
                else:
                    new_tasks1 = market_robot1.generate_frontier_goals(3)
                    for task in new_tasks1:
                        market_robot1.tasks.append(task)
                    market_robot1.plan_optimal_tour()
                
                if market_robot2.tasks:
                    try:
                        next_state2, r2, d2 = market_robot2.move_to_next_task()
                    except Exception as e:
                        if market_robot2.tasks:
                            market_robot2.tasks.pop(0)
                else:
                    new_tasks2 = market_robot2.generate_frontier_goals(3)
                    for task in new_tasks2:
                        market_robot2.tasks.append(task)
                    market_robot2.plan_optimal_tour()
                
                # 大幅降低拍賣頻率
                if steps % 10 == 0:  # 從5改為10
                    for robot in [market_robot1, market_robot2]:
                        if robot.tasks and random.random() < 0.5:  # 只有30%機率
                            task = robot.tasks[0]  # 只拍賣第一個任務
                            try:
                                robot.auction_task(task)
                            except:
                                pass
                
                tracker.update()
                
                robot1_map = tracker.robot1_individual_map
                robot2_map = tracker.robot2_individual_map
                
                if robot1_map is not None and robot2_map is not None:
                    total_explorable = np.sum(robot1.global_map == 255)
                    
                    robot1_explored = np.sum(robot1_map == 255)
                    robot1_coverage = robot1_explored / total_explorable if total_explorable > 0 else 0
                    
                    robot2_explored = np.sum(robot2_map == 255)
                    robot2_coverage = robot2_explored / total_explorable if total_explorable > 0 else 0
                    
                    intersection = np.sum((robot1_map == 255) & (robot2_map == 255))
                    intersection_coverage = intersection / total_explorable if total_explorable > 0 else 0
                    
                    union = np.sum((robot1_map == 255) | (robot2_map == 255))
                    union_coverage = union / total_explorable if total_explorable > 0 else 0
                    
                    total_distance = market_robot1.total_distance + market_robot2.total_distance
                    
                    intersection_data.append({
                        'step': steps,
                        'robot1_coverage': robot1_coverage,
                        'robot2_coverage': robot2_coverage,
                        'intersection_coverage': intersection_coverage,
                        'union_coverage': union_coverage,
                        'total_distance': total_distance
                    })
                    
                    # 每5步記錄一次數據
                    if steps % 5 == 0:
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
                
                # 每20步儲存一次繪圖（從10改為20）
                if steps % 20 == 0:
                    save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_step_{steps:04d}.png'))
                    save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_step_{steps:04d}.png'))
                    tracker.save_current_maps(steps)
                    
                    stats = op_exec.get_statistics()
                    print(f"步數: {steps}, 聯合覆蓋率: {stats['exploration_ratio']:.1%}, 總距離: {stats['total_distance']:.1f}")
                
                if steps >= 50000:
                    print("達到最大步數限制，終止探索")
                    break
            
            save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_final_step_{steps:04d}.png'))
            save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_final_step_{steps:04d}.png'))
            tracker.save_current_maps(steps)
            
            # 生成圖表（保持不變）
            plt.figure(figsize=(12, 8))
            steps_data = [data['step'] for data in intersection_data]
            robot1_coverage = [data['robot1_coverage'] for data in intersection_data]
            robot2_coverage = [data['robot2_coverage'] for data in intersection_data]
            intersection_coverage = [data['intersection_coverage'] for data in intersection_data]
            union_coverage = [data['union_coverage'] for data in intersection_data]
            
            plt.plot(steps_data, robot1_coverage, 'b-', linewidth=2, label='Robot 1')
            plt.plot(steps_data, robot2_coverage, 'r-', linewidth=2, label='Robot 2')
            plt.plot(steps_data, intersection_coverage, 'g-', linewidth=2, label='Intersection')
            plt.plot(steps_data, union_coverage, 'k-', linewidth=2, label='Union')
            
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Coverage', fontsize=14)
            plt.title(f'Time-Coverage Analysis - Market Based OPTIMIZED (Start Point {start_idx+1})', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.ylim(0, 1.05)
            
            plt.savefig(os.path.join(current_output_dir, 'time_coverage_analysis.png'), dpi=300)
            plt.close()
            
            plt.figure(figsize=(12, 8))
            efficiency = [data['union_coverage'] / data['total_distance'] if data['total_distance'] > 0 else 0 
                         for data in intersection_data]
            
            plt.plot(steps_data, efficiency, 'b-', linewidth=2)
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Efficiency (Coverage/Distance)', fontsize=14)
            plt.title(f'Exploration Efficiency - Market Based OPTIMIZED (Start Point {start_idx+1})', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.savefig(os.path.join(current_output_dir, 'efficiency_analysis.png'), dpi=300)
            plt.close()
            
            tracker.stop_tracking()
            tracker.cleanup()
            
            for robot in [robot1, robot2]:
                if robot is not None and hasattr(robot, 'cleanup_visualization'):
                    robot.cleanup_visualization()
            
            print(f"完成起始點 {start_idx+1} 的測試，總步數: {steps}")
            
        except Exception as e:
            print(f"測試過程中出錯: {str(e)}")
            traceback.print_exc()
            
            try:
                if tracker is not None:
                    tracker.cleanup()
                
                for robot in [robot1, robot2]:
                    if robot is not None and hasattr(robot, 'cleanup_visualization'):
                        robot.cleanup_visualization()
            except:
                pass
    
    print(f"\n===== 完成所有起始點的測試 =====")
    print(f"結果儲存在: {output_dir}")

def main():
    map_file_path = os.path.join(os.getcwd(), 'data', 'DungeonMaps', 'test', 'ttttttt.png')
    
    if not os.path.exists(map_file_path):
        print(f"警告: 在 {map_file_path} 找不到指定的地圖檔案")
        exit(1)
    
    start_points = [
        [[100, 100], [100, 100]],  # 起始點 1
        [[520, 120], [520, 120]],  # 起始點 2
        [[630, 150], [630, 150]],   # 起始點 3
        [[250, 130], [250, 130]],   # 起始點 4
        [[250, 100], [250, 100]],  # 起始點 5
        [[400, 120], [400, 120]],  # 起始點 6
        [[140, 410], [140, 410]],   # 起始點 7
        [[110, 590], [110, 590]],   # 起始點 8
        [[90, 300], [90, 300]],   # 起始點 9
        [[260, 200], [260, 200]],  # 起始點 10
    ]
    
    output_dir = 'results_market_based_optimized3'
    print(f"執行優化版本的基於市場架構的多機器人探索...")
    market_exploration(map_file_path, start_points, output_dir)

if __name__ == '__main__':
    main()