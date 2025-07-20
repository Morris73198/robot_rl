import os
import numpy as np
import numpy.ma as ma
from scipy import spatial
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from ..utils.inverse_sensor_model import inverse_sensor_model
from scipy.ndimage import distance_transform_edt
import random
from heapq import heappush, heappop
from ..config import ROBOT_CONFIG, REWARD_CONFIG

import heapq


class Robot:
    @classmethod
    def create_shared_robots(cls, index_map, train=True, plot=True):
        """創建共享環境的機器人實例"""
        print("Creating robots with shared environment...")  # 調試信息

        # 創建第一個機器人，它會加載和初始化地圖
        robot1 = cls(index_map, train, plot, is_primary=True)
        
        # 創建第二個機器人，共享第一個機器人的地圖和相關資源
        robot2 = cls(index_map, train, plot, is_primary=False, shared_env=robot1)
        
        robot1.other_robot = robot2
        print(f"Robot1 other_robot set: {robot1.other_robot is not None}")  # 調試信息
        print(f"Robot2 shared_env set: {robot2.shared_env is not None}")    # 調試信息
        
        return robot1, robot2
    
    # ================ 新增：多機器人創建方法 ================
    @classmethod
    def create_multi_robots(cls, num_robots, index_map, train=True, plot=True):
        """創建多個共享環境的機器人實例
        
        Args:
            num_robots: 機器人數量
            index_map: 地圖索引
            train: 是否處於訓練模式
            plot: 是否繪製可視化
            
        Returns:
            list: 機器人實例列表
        """
        print(f"Creating {num_robots} robots with shared environment...")
        
        if num_robots < 2:
            raise ValueError("Number of robots must be at least 2")
            
        # 創建第一個機器人，它會加載和初始化地圖
        primary_robot = cls(index_map, train, plot, is_primary=True, robot_id=0, num_robots=num_robots)
        
        robots = [primary_robot]
        
        # 創建其他機器人，共享第一個機器人的地圖和相關資源
        for i in range(1, num_robots):
            robot = cls(index_map, train, plot, is_primary=False, shared_env=primary_robot, robot_id=i, num_robots=num_robots)
            robots.append(robot)
            
        # 設置機器人之間的相互引用
        for i, robot in enumerate(robots):
            robot.all_robots = robots
            robot.other_robots = [r for j, r in enumerate(robots) if j != i]
            
        print(f"Created {num_robots} robots successfully")
        return robots
    
    def __init__(self, index_map, train, plot, is_primary=True, shared_env=None, robot_id=None, num_robots=None):
        """初始化機器人環境
        
        Args:
            index_map: 地圖索引
            train: 是否處於訓練模式
            plot: 是否繪製可視化
            is_primary: 是否為主要機器人(負責加載地圖)
            shared_env: 共享環境的機器人實例
            robot_id: 機器人ID (新增，向後兼容)
            num_robots: 總機器人數量 (新增，向後兼容)
        """
        if not shared_env and not is_primary:
            raise ValueError("Non-primary robot must have a shared environment")
        if shared_env and is_primary:
            raise ValueError("Primary robot cannot have a shared environment")
        self.mode = train
        self.plot = plot
        self.is_primary = is_primary
        
        self.shared_env = shared_env 
        
        # ================ 新增：多機器人支援 ================
        self.robot_id = robot_id if robot_id is not None else (0 if is_primary else 1)
        self.num_robots = num_robots if num_robots is not None else 2
        self.all_robots = []  # 將在後面設置
        self.other_robots = []  # 將在後面設置
        
        self.lethal_cost = 100  # 致命障礙物代價
        self.decay_factor = 3  # 代價衰減因子
        self.inflation_radius = ROBOT_CONFIG['robot_size'] * 1.5  # 膨脹半徑為機器人尺寸的1.5倍
        
        if is_primary:
            # 主要機器人負責加載地圖和初始化環境
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            
            if self.mode:
                self.map_dir = os.path.join(base_dir, 'robot_rl/data', 'DungeonMaps', 'train')
            else:
                self.map_dir = os.path.join(base_dir, 'robot_rl/data', 'DungeonMaps', 'test')
                
            os.makedirs(self.map_dir, exist_ok=True)
            
            self.map_list = os.listdir(self.map_dir)
            if not self.map_list:
                raise FileNotFoundError(f"No map files found in {self.map_dir}")
                
            self.map_number = np.size(self.map_list)
            if self.mode:
                random.shuffle(self.map_list)
                
            self.li_map = index_map
            
            # 初始化地圖
            self.global_map, self.initial_positions = self.map_setup(
                os.path.join(self.map_dir, self.map_list[self.li_map])
            )
            
            # ================ 修改：支援多機器人起始位置 ================
            # 確保有足夠的起始位置
            if len(self.initial_positions) < self.num_robots:
                self._generate_additional_positions()
                
            # 為當前機器人選擇起始位置
            self.robot_position = self.initial_positions[self.robot_id].astype(np.int64)
            # 保持向後兼容：為兩機器人情況設置 other_robot_position
            if self.num_robots >= 2:
                self.other_robot_position = self.initial_positions[1 if self.robot_id == 0 else 0].astype(np.int64)
            
            self.op_map = np.ones(self.global_map.shape) * 127
            self.map_size = np.shape(self.global_map)
            
            # 初始化其他屬性
            self.movement_step = ROBOT_CONFIG['movement_step']
            self.finish_percent = ROBOT_CONFIG['finish_percent']
            self.sensor_range = ROBOT_CONFIG['sensor_range']
            self.robot_size = ROBOT_CONFIG['robot_size']
            self.local_size = ROBOT_CONFIG['local_size']
            
            self.old_position = np.zeros([2])
            self.old_op_map = np.empty([0])
            self.current_target_frontier = None
            self.is_moving_to_target = False
            self.steps = 0
            
            self.t = self.map_points(self.global_map)
            self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())
            
            # 在設置完 robot_position 後初始化路徑記錄屬性
            self.xPoint = np.array([self.robot_position[0]])
            self.yPoint = np.array([self.robot_position[1]])
            
            if self.plot:
                self.initialize_visualization()
        else:
            # 次要機器人共享主要機器人的環境
            self.map_dir = shared_env.map_dir
            self.map_list = shared_env.map_list
            self.map_number = shared_env.map_number
            self.li_map = shared_env.li_map
            
            self.global_map = shared_env.global_map
            self.op_map = shared_env.op_map
            self.map_size = shared_env.map_size
            
            # ================ 修改：支援多機器人起始位置 ================
            # 使用對應ID的起始位置
            if hasattr(shared_env, 'initial_positions') and len(shared_env.initial_positions) > self.robot_id:
                self.robot_position = shared_env.initial_positions[self.robot_id].astype(np.int64)
            else:
                # 向後兼容：交換位置
                self.robot_position = shared_env.other_robot_position.copy()
                
            # 保持向後兼容：設置 other_robot_position
            if hasattr(shared_env, 'robot_position'):
                self.other_robot_position = shared_env.robot_position.copy()
            
            # 共享其他屬性
            self.movement_step = shared_env.movement_step
            self.finish_percent = shared_env.finish_percent
            self.sensor_range = shared_env.sensor_range
            self.robot_size = shared_env.robot_size 
            self.local_size = shared_env.local_size
            
            self.old_position = np.zeros([2])
            self.old_op_map = np.empty([0])
            self.current_target_frontier = None
            self.is_moving_to_target = False
            self.steps = 0
            
            self.t = shared_env.t
            self.free_tree = shared_env.free_tree
            
            # 在設置完 robot_position 後初始化路徑記錄屬性
            self.xPoint = np.array([self.robot_position[0]])
            self.yPoint = np.array([self.robot_position[1]])
            
            if self.plot:
                self.initialize_visualization()

    # ================ 新增：生成額外起始位置 ================
    def _generate_additional_positions(self):
        """為不足的機器人生成額外的起始位置"""
        needed = self.num_robots - len(self.initial_positions)
        if needed <= 0:
            return
            
        # 獲取自由空間
        free_space = np.where(self.global_map > 150)
        free_positions = np.array([free_space[1], free_space[0]]).T
        
        if len(free_positions) < needed:
            raise ValueError(f"Map does not have enough free space for {self.num_robots} robots")
        
        # 智能分配機器人起始位置
        min_distance = max(20, self.global_map.shape[0] // 10)
        additional_positions = []
        
        # 基於現有位置選擇新位置
        existing_positions = list(self.initial_positions)
        
        for _ in range(needed):
            best_pos = None
            best_min_distance = 0
            
            # 嘗試多次尋找最佳位置
            for _ in range(min(1000, len(free_positions))):
                candidate = free_positions[np.random.randint(len(free_positions))]
                
                # 計算到所有已選位置的最小距離
                distances = [np.linalg.norm(candidate - pos) for pos in existing_positions]
                min_dist = min(distances) if distances else float('inf')
                
                # 選擇距離最遠的位置
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_pos = candidate
                    
                # 如果找到滿足最小距離要求的位置，直接使用
                if min_dist >= min_distance:
                    break
                    
            if best_pos is not None:
                additional_positions.append(best_pos)
                existing_positions.append(best_pos)
            else:
                # 如果無法找到理想位置，隨機選擇
                additional_positions.append(free_positions[np.random.randint(len(free_positions))])
                
        # 添加到初始位置列表
        self.initial_positions = np.vstack([self.initial_positions, np.array(additional_positions)])

    # ================ 新增：多機器人相關方法 ================
    def get_all_other_robots_positions(self):
        """獲取所有其他機器人的位置"""
        if not hasattr(self, 'other_robots') or not self.other_robots:
            # 向後兼容：返回單個其他機器人位置
            if hasattr(self, 'other_robot_position'):
                return [self.other_robot_position.copy()]
            return []
        
        return [robot.robot_position.copy() for robot in self.other_robots]
    
    def get_nearby_robots(self, radius=None):
        """獲取附近的機器人"""
        if radius is None:
            radius = self.sensor_range * 2
            
        other_positions = self.get_all_other_robots_positions()
        nearby_robots = []
        
        for i, pos in enumerate(other_positions):
            distance = np.linalg.norm(self.robot_position - pos)
            if distance <= radius:
                nearby_robots.append({
                    'robot_id': i,
                    'position': pos,
                    'distance': distance
                })
                
        return nearby_robots
    
    def avoid_collision_with_robots(self, target_position):
        """避免與其他機器人碰撞"""
        other_positions = self.get_all_other_robots_positions()
        min_distance = self.robot_size * 2
        
        for other_pos in other_positions:
            distance = np.linalg.norm(target_position - other_pos)
            if distance < min_distance:
                # 計算避開的方向
                direction = target_position - other_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    target_position = other_pos + direction * min_distance
                    
        return target_position

    # ================ 保持所有原有方法不變 ================
    def begin(self):
        """初始化並返回初始狀態"""
        self.op_map = self.inverse_sensor(
            self.robot_position, self.sensor_range, self.op_map, self.global_map)
            
        step_map = self.robot_model(
            self.robot_position, self.robot_size, self.t, self.op_map)
            
        # map_local = self.local_map(
        #     self.robot_position, step_map, self.map_size, self.sensor_range + self.local_size)
        
        resized_map = resize(step_map, (84, 84))
        state = np.expand_dims(resized_map, axis=-1)
        
        if self.plot:
            self.plot_env()
            
        return state

    def move_to_frontier(self, target_frontier):
        """移動到frontier，一次移動一步，直到到達目標或確定無法到達"""
        # 保存當前目標
        self.current_target_frontier = target_frontier
        
        # 如果沒有當前路徑或需要重新規劃路徑
        if not hasattr(self, 'current_path') or self.current_path is None:
            # 檢查目標是否可達（是否已知區域可達路徑）
            known_path = self.astar_path(
                self.op_map,
                self.robot_position.astype(np.int32),
                target_frontier.astype(np.int32),
                safety_distance=ROBOT_CONFIG['safety_distance']
            )
            
            if known_path is None:
                # 無法找到只經過已知區域的路徑
                self.current_path = None
                return self.get_observation(), -1, True
                
            # 保存完整路徑
            self.current_path = self.simplify_path(known_path)
            self.current_path_index = 0
            
        # 執行一步移動
        if self.current_path_index < len(self.current_path) - 1:
            self.current_path_index += 1
            next_position = np.array(self.current_path[self.current_path_index])
            
            # ================ 新增：避免與其他機器人碰撞 ================
            next_position = self.avoid_collision_with_robots(next_position)
            
            self.robot_position = next_position
            
            # 更新傳感器觀測
            self.op_map = self.inverse_sensor(
                self.robot_position, self.sensor_range, self.op_map, self.global_map)
                
        # 檢查是否到達目標
        distance_to_target = np.linalg.norm(self.robot_position - target_frontier)
        if distance_to_target < self.movement_step * 2:
            self.current_path = None
            self.current_target_frontier = None
            self.is_moving_to_target = False
            
        reward = self.calculate_reward()
        return self.get_observation(), reward, False

    def check_completion(self):
        """检查探索是否完成"""
        exploration_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        
        if exploration_ratio > self.finish_percent:
            self.li_map += 1
            if self.li_map >= self.map_number:
                self.li_map = 0
                return True, True  # 完成所有地图
                
            self.__init__(self.li_map, self.mode, self.plot)
            return True, False  # 完成当前地图
            
        return False, False
    
    def reset(self):
        """重置環境到新地圖，並確保所有機器人使用相同的地圖"""
        # 在重置之前清理舊的可視化
        if self.plot:
            self.cleanup_visualization()

        if self.is_primary:
            self.li_map += 1
            if self.li_map >= self.map_number:
                self.li_map = 0
                
            # 主要機器人負責初始化新地圖
            self.global_map, self.initial_positions = self.map_setup(
                os.path.join(self.map_dir, self.map_list[self.li_map])
            )
            
            # ================ 修改：支援多機器人重置 ================
            # 確保有足夠的起始位置
            if len(self.initial_positions) < self.num_robots:
                self._generate_additional_positions()
            
            # 設置新的起始位置
            self.robot_position = self.initial_positions[self.robot_id].astype(np.int64)
            # 保持向後兼容
            if self.num_robots >= 2:
                self.other_robot_position = self.initial_positions[1 if self.robot_id == 0 else 0].astype(np.int64)
            
            # 重置地圖狀態
            self.op_map = np.ones(self.global_map.shape) * 127
            
            # 重新初始化KD樹和空閒空間點
            self.t = self.map_points(self.global_map)
            self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())
        else:
            # 使用共享環境的地圖和相關資源
            self.li_map = self.shared_env.li_map
            self.global_map = self.shared_env.global_map
            self.op_map = self.shared_env.op_map
            self.map_size = self.shared_env.map_size
            
            # ================ 修改：支援多機器人重置 ================
            # 使用對應ID的起始位置
            if hasattr(self.shared_env, 'initial_positions') and len(self.shared_env.initial_positions) > self.robot_id:
                self.robot_position = self.shared_env.initial_positions[self.robot_id].astype(np.int64)
            else:
                # 向後兼容：交換位置
                self.robot_position = self.shared_env.other_robot_position.copy()
                
            # 保持向後兼容：設置 other_robot_position
            if hasattr(self.shared_env, 'robot_position'):
                self.other_robot_position = self.shared_env.robot_position.copy()
            
            # 共享KD樹和空閒空間點
            self.t = self.shared_env.t
            self.free_tree = self.shared_env.free_tree
        
        # 重置路徑規劃和frontier相關的狀態
        self.old_position = np.zeros([2])
        self.old_op_map = np.empty([0])
        self.current_target_frontier = None
        self.is_moving_to_target = False
        self.current_path = None
        self.current_path_index = 0
        self.steps = 0
        
        # 重置可視化相關的狀態
        if self.plot:
            self.initialize_visualization()
        
        # 執行初始觀測
        return self.begin()

    def check_done(self):
        """检查是否需要结束当前回合"""
        # 检查探索进度
        exploration_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        if exploration_ratio > self.finish_percent:
            return True
            
        # 检查是否还有可探索的frontiers
        frontiers = self.get_frontiers()
        if len(frontiers) == 0:
            return True
            
        return False

    def get_observation(self):
        """获取当前观察状态"""
        step_map = self.robot_model(
            self.robot_position, self.robot_size, self.t, self.op_map)
            
        # 神經網路輸入只以機器人週邊為範圍    
        # map_local = self.local_map(
        #     self.robot_position, step_map, self.map_size, 
        #     self.sensor_range + self.local_size)
        
        # 3. 調整大小為神經網絡輸入大小
        resized_map = resize(step_map, (84, 84))
        # resized_map = resize(map_local, (84, 84))
        return np.expand_dims(resized_map, axis=-1)

    def get_exploration_progress(self):
        """获取探索进度"""
        return np.sum(self.op_map == 255) / np.sum(self.global_map == 255)

    def get_state_info(self):
        """获取当前状态信息"""
        return {
            'position': self.robot_position.copy(),
            'map': self.op_map.copy(),
            'frontiers': self.get_frontiers(),
            'target_frontier': self.current_target_frontier,
            'exploration_progress': self.get_exploration_progress()
        }

    def set_state(self, state_info):
        """设置状态"""
        self.robot_position = state_info['position'].copy()
        self.op_map = state_info['map'].copy()
        self.current_target_frontier = state_info['target_frontier']
        
        if self.plot:
            self.plot_env()
            
    def get_normalized_position(self):
        """獲取正規化後的機器人位置"""
        return np.array([
            self.robot_position[0] / float(self.map_size[1]),  # x座標正規化
            self.robot_position[1] / float(self.map_size[0])   # y座標正規化
        ])
        
    def initialize_visualization(self):
        """初始化可視化相關的屬性"""
        self.xPoint = np.array([self.robot_position[0]])
        self.yPoint = np.array([self.robot_position[1]])
        self.x2frontier = np.empty([0])
        self.y2frontier = np.empty([0])
        
        # 為每個機器人創建獨立的 figure
        self.fig = plt.figure(figsize=(10, 10))
        # ================ 修改：支援多機器人視窗標題 ================
        robot_name = f'Robot{self.robot_id}' if hasattr(self, 'robot_id') else ('Robot1' if self.is_primary else 'Robot2')
        self.fig.canvas.manager.set_window_title(f'{robot_name} Exploration')

    # ================ 保持向後兼容性屬性 ================
    @property
    def other_robot(self):
        """向後兼容：獲取另一個機器人（只適用於兩機器人情況）"""
        if hasattr(self, 'other_robots') and self.other_robots:
            return self.other_robots[0]
        elif hasattr(self, '_other_robot'):
            return self._other_robot
        return None
    
    @other_robot.setter
    def other_robot(self, value):
        """向後兼容：設置另一個機器人"""
        self._other_robot = value

    def get_other_robot_pos(self):
        """獲取另一個機器人的位置"""
        if hasattr(self, 'other_robot_position'):
            return self.other_robot_position.copy()
        elif hasattr(self, 'other_robots') and self.other_robots:
            return self.other_robots[0].robot_position.copy()
        return None

    def update_other_robot_pos(self, pos):
        """更新另一個機器人的位置"""
        if hasattr(self, 'other_robot_position'):
            self.other_robot_position = np.array(pos)
        
    def cleanup_visualization(self):
        """清理可視化資源"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)  # 關閉特定的figure
            plt.clf()  # 清除當前figure
            self.fig = None  # 重置figure引用

    # ================ 以下是所有原有方法，完全保持不變 ================
    
    def robot_model(self, position, robot_size, points, map_glo):
        """機器人模型"""
        map_copy = map_glo.copy()
        # robot_points = self.range_search(position, robot_size, points)
        # for point in robot_points:
        #     y, x = point[::-1].astype(int) #(x,y)轉（y,x）
        #     if 0 <= y < map_copy.shape[0] and 0 <= x < map_copy.shape[1]:
        #         map_copy[y, x] = 76 # 機器人位置標記為 76
        x, y = int(position[0]), int(position[1])
        if 0 <= x < map_copy.shape[1] and 0 <= y < map_copy.shape[0]:
            map_copy[y, x] = 76 # 機器人位置標記為 76
        return map_copy

    def range_search(self, position, r, points):
        """範圍搜索"""
        diff = points - position
        dist_sq = np.sum(diff * diff, axis=1)
        return points[dist_sq <= r * r]

    def fast_collision_check(self, start_point, end_point, map_size, map_glo):
        """快速碰撞檢查"""
        start = np.round(start_point).astype(int)
        end = np.round(end_point).astype(int)
        
        # 邊界檢查
        if not (0 <= end[0] < map_size[1] and 0 <= end[1] < map_size[0]):
            return np.array([end]).reshape(1, 2), True
            
        if map_glo[end[1], end[0]] == 1:
            return np.array([end]).reshape(1, 2), True
        
        # 簡化的線性插值檢查
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return np.array([[-1, -1]]).reshape(1, 2), False
            
        x_step = dx / steps
        y_step = dy / steps
        
        # 檢查幾個關鍵點
        check_points = np.linspace(0, steps, min(5, steps + 1))
        for t in check_points:
            x = int(start[0] + x_step * t)
            y = int(start[1] + y_step * t)
            
            if not (0 <= x < map_size[1] and 0 <= y < map_size[0]):
                return np.array([[x, y]]).reshape(1, 2), True
                
            if map_glo[y, x] == 1:
                return np.array([[x, y]]).reshape(1, 2), True
        
        return np.array([[-1, -1]]).reshape(1, 2), False

    def inverse_sensor(self, robot_position, sensor_range, op_map, map_glo):
        """反向傳感器模型"""
        return inverse_sensor_model(
            int(robot_position[0]), int(robot_position[1]), 
            sensor_range, op_map, map_glo)

    def frontier(self, op_map, map_size, points):
        """Frontier檢測"""
        y_len, x_len = map_size
        mapping = (op_map == 127).astype(int)
        
        mapping = np.pad(mapping, ((1,1), (1,1)), 'constant')
        
        fro_map = (
            mapping[2:, 1:x_len+1] +    # 下
            mapping[:y_len, 1:x_len+1] + # 上
            mapping[1:y_len+1, 2:] +     # 右
            mapping[1:y_len+1, :x_len] + # 左
            mapping[:y_len, 2:] +        # 右上
            mapping[2:, :x_len] +        # 左下
            mapping[2:, 2:] +            # 右下
            mapping[:y_len, :x_len]      # 左上
        )
        
        free_space = op_map.ravel(order='F') == 255
        frontier_condition = (1 < fro_map.ravel(order='F')) & (fro_map.ravel(order='F') < 8)
        valid_points = points[np.where(free_space & frontier_condition)[0]]
        
        if len(valid_points) > 0:
            selected_points = [valid_points[0]]
            min_dist = ROBOT_CONFIG['min_frontier_dist']
            
            for point in valid_points[1:]:
                distances = [np.linalg.norm(point - p) for p in selected_points]
                if min(distances) > min_dist:
                    selected_points.append(point)
            
            return np.array(selected_points).astype(int)
        
        return valid_points.astype(int)

    def get_frontiers(self):
        """取得當前可用的frontier點，考慮其他機器人的位置"""
        if self.is_moving_to_target and self.current_target_frontier is not None:
            return np.array([self.current_target_frontier])
            
        frontiers = self.frontier(self.op_map, self.map_size, self.t)
        if len(frontiers) == 0:
            return np.zeros((0, 2))
            
        # 計算到自己的距離
        distances = np.linalg.norm(frontiers - self.robot_position, axis=1)
        
        # ================ 修改：考慮多機器人的距離 ================
        # 獲取所有其他機器人的位置
        other_positions = self.get_all_other_robots_positions()
        
        if other_positions:
            # 計算到最近其他機器人的距離
            other_distances = []
            for frontier in frontiers:
                min_other_dist = min([np.linalg.norm(frontier - pos) for pos in other_positions])
                other_distances.append(min_other_dist)
            other_distances = np.array(other_distances)
            
            # 根據距離和其他機器人的位置對frontier進行排序
            # 優先選擇離自己近但離其他機器人遠的點
            scores = distances - 0.5 * other_distances  # 可以調整權重
        else:
            # 向後兼容：只考慮到自己的距離
            scores = distances
            
        sorted_indices = np.argsort(scores)
        return frontiers[sorted_indices]

    def plot_env(self):
        """繪製環境和機器人"""
        # 使用各自的 figure
        plt.figure(self.fig.number)
        plt.clf()  # 清除當前 figure
        
        # 1. 繪製基礎地圖
        plt.imshow(self.op_map, cmap='gray')
        plt.axis((0, self.map_size[1], self.map_size[0], 0))
        
        # ================ 修改：支援多機器人顏色 ================
        # 為不同機器人使用不同顏色
        colors = ['#800080', '#FFA500', '#00FF00', '#FF0000', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        robot_color = colors[self.robot_id % len(colors)] if hasattr(self, 'robot_id') else ('#800080' if self.is_primary else '#FFA500')
        robot_name = f'Robot{self.robot_id}' if hasattr(self, 'robot_id') else ('Robot1' if self.is_primary else 'Robot2')
        
        # 2. 繪製路徑軌跡
        if len(self.xPoint) > 1:  # 確保有超過一個點才畫線
            plt.plot(self.xPoint, self.yPoint, color=robot_color, 
                    linewidth=2, label=f'{robot_name} Path')
        
        # 3. 繪製 frontier 點
        frontiers = self.get_frontiers()
        if len(frontiers) > 0:
            plt.scatter(frontiers[:, 0], frontiers[:, 1], 
                    c='red', marker='*', s=100, label='Frontiers')
        
        # 4. 繪製當前機器人位置
        plt.scatter(self.robot_position[0], self.robot_position[1], 
                   c=robot_color, marker='o', s=200, label=robot_name)
        
        # ================ 新增：繪製其他機器人位置 ================
        other_positions = self.get_all_other_robots_positions()
        for i, pos in enumerate(other_positions):
            other_robot_id = i if i < self.robot_id else i + 1
            other_color = colors[other_robot_id % len(colors)]
            other_name = f'Robot{other_robot_id}'
            plt.scatter(pos[0], pos[1], c=other_color, marker='s', s=150, 
                       label=other_name, alpha=0.7)
        
        # 5. 繪製目標frontier
        if self.current_target_frontier is not None:
            plt.scatter(self.current_target_frontier[0], self.current_target_frontier[1], 
                       c='yellow', marker='x', s=200, label='Target')
        
        plt.legend()
        plt.title(f'{robot_name} - Exploration Progress: {self.get_exploration_progress():.2%}')
        plt.pause(0.01)

    # ================ 以下為路徑規劃相關方法，保持不變 ================
    
    def inflate_map(self, binary_map):
        """膨脹地圖以創建代價地圖"""
        obstacle_map = (binary_map == 1)
        distances = distance_transform_edt(~obstacle_map)
        cost_map = np.zeros_like(distances)
        cost_map[obstacle_map] = self.lethal_cost
        inflation_mask = (distances > 0) & (distances <= self.inflation_radius)
        cost_map[inflation_mask] = self.lethal_cost * np.exp(
            -self.decay_factor * distances[inflation_mask] / self.inflation_radius
        )
        return cost_map

    def astar_with_inflation(self, start, goal, op_map):
        """考慮膨脹的A*路徑規劃，只走已知區域"""
        binary_map = np.zeros_like(op_map, dtype=int)
        binary_map[op_map == 1] = 1
        binary_map[op_map == 127] = 1
        cost_map = self.inflate_map(binary_map)
        
        return self.astar_with_cost(start, goal, cost_map)

    def get_neighbors_with_cost(self, x, y, cost_map):
        """獲取考慮代價的鄰居節點"""
        neighbors = []
        directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (-1, 1), (1, -1), (-1, -1)
        ]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            if not (0 <= new_x < self.map_size[1] and 0 <= new_y < self.map_size[0]):
                continue
                
            if cost_map[new_y, new_x] < self.lethal_cost and self.op_map[new_y, new_x] == 255:
                neighbors.append((new_x, new_y))
                
        return neighbors

    def astar_with_cost(self, start, goal, cost_map):
        """帶代價的A*算法"""
        start = tuple(start)
        goal = tuple(goal)
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            neighbors = self.get_neighbors_with_cost(current[0], current[1], cost_map)
            
            for neighbor in neighbors:
                tentative_g_score = g_score[current] + cost_map[neighbor[1], neighbor[0]]
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None

    def astar_path(self, op_map, start, goal, safety_distance=1):
        """A*路徑規劃主方法"""
        if safety_distance > 1:
            return self.astar_with_inflation(start, goal, op_map)
        else:
            return self.astar(op_map, start, goal)

    def astar(self, op_map, start, goal):
        """A*算法實現"""
        start = tuple(start)
        goal = tuple(goal)
        
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < self.map_size[1] and 
                    0 <= neighbor[1] < self.map_size[0]):
                    continue
                    
                if op_map[neighbor[1]][neighbor[0]] == 1:
                    continue
                    
                if op_map[neighbor[1]][neighbor[0]] != 255:
                    continue
                    
                if neighbor in close_set:
                    continue
                    
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                
                if neighbor not in gscore or tentative_g_score < gscore[neighbor]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = gscore[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        return None

    def heuristic(self, a, b):
        """啟發式函數"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def simplify_path(self, path, threshold=1.5):
        """簡化路徑"""
        if not path or len(path) < 3:
            return path
        
        path = np.array(path).T
        
        def point_line_distance(point, line_start, line_end):
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = np.linalg.norm(line_vec)
            
            if line_len == 0:
                return np.linalg.norm(point_vec)
            
            line_unit_vec = line_vec / line_len
            projection_length = np.dot(point_vec, line_unit_vec)
            projection_length = np.clip(projection_length, 0, line_len)
            
            return np.linalg.norm(point_vec - projection_length * line_unit_vec)
        
        def simplify_recursive(points, epsilon, mask):
            dmax = 0
            index = 0
            end = len(points) - 1
            
            for i in range(1, end):
                d = point_line_distance(points[i], points[0], points[end])
                if d > dmax:
                    index = i
                    dmax = d
            
            if dmax > epsilon:
                mask1 = mask.copy()
                mask2 = mask.copy()
                simplify_recursive(points[:index + 1], epsilon, mask1)
                simplify_recursive(points[index:], epsilon, mask2)
                
                for i in range(len(mask)):
                    mask[i] = mask1[i] if i <= index else mask2[i - index]
            else:
                for i in range(1, end):
                    mask[i] = False
        
        points = path.T
        mask = np.ones(len(points), dtype=bool)
        simplify_recursive(points, threshold, mask)
        
        return path[:, mask]

    # ================ 以下為獎勵計算相關方法，保持不變 ================
    
    def calculate_reward(self):
        """計算獎勵"""
        # 1. 探索獎勵
        new_exploration = np.sum((self.op_map == 255) & (self.old_op_map != 255))
        exploration_reward = new_exploration * REWARD_CONFIG.get('exploration_reward', 1.0)
        
        # 2. 效率獎勵（基於移動距離）
        movement_distance = np.linalg.norm(self.robot_position - self.old_position)
        efficiency_reward = -movement_distance * REWARD_CONFIG.get('movement_penalty', 0.1)
        
        # 3. 其他機器人路徑重疊懲罰
        other_path_penalty = 0
        current_pos = np.array([self.robot_position[0], self.robot_position[1]])
        
        # ================ 修改：支援多機器人路徑重疊懲罰 ================
        if hasattr(self, 'other_robots') and self.other_robots:
            for other_robot in self.other_robots:
                if hasattr(other_robot, 'xPoint') and hasattr(other_robot, 'yPoint') and len(other_robot.xPoint) > 0:
                    recent_history = 500
                    start_idx = max(0, len(other_robot.xPoint) - recent_history)
                    
                    other_path = np.column_stack((
                        other_robot.xPoint[start_idx:],
                        other_robot.yPoint[start_idx:]
                    ))
                    
                    distances = np.linalg.norm(other_path - current_pos, axis=1)
                    min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                    
                    safe_distance = ROBOT_CONFIG['sensor_range'] * 1.5
                    if min_distance < safe_distance:
                        other_path_penalty += -4 * np.exp(-min_distance/safe_distance)
        elif hasattr(self, 'other_robot') and self.other_robot is not None:
            # 向後兼容：處理單個其他機器人
            other_robot = self.other_robot if self.is_primary else self.shared_env
            
            if (other_robot is not None and 
                hasattr(other_robot, 'xPoint') and 
                hasattr(other_robot, 'yPoint') and 
                len(other_robot.xPoint) > 0):
                
                recent_history = 500
                start_idx = max(0, len(other_robot.xPoint) - recent_history)
                
                other_path = np.column_stack((
                    other_robot.xPoint[start_idx:],
                    other_robot.yPoint[start_idx:]
                ))
                
                distances = np.linalg.norm(other_path - current_pos, axis=1)
                min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                
                safe_distance = ROBOT_CONFIG['sensor_range'] * 1.5
                if min_distance < safe_distance:
                    other_path_penalty = -4 * np.exp(-min_distance/safe_distance)
                else:
                    other_path_penalty = 1
        
        # 4. 協同探索獎勵
        distance_reward = 0
        other_positions = self.get_all_other_robots_positions()
        if other_positions:
            for other_pos in other_positions:
                distance_to_other = np.linalg.norm(self.robot_position - other_pos)
                optimal_distance = self.sensor_range * 2
                distance_reward += -0.5 * abs(distance_to_other - optimal_distance) / optimal_distance
        
        # 組合獎勵
        total_reward = (
            exploration_reward +
            efficiency_reward +
            distance_reward +
            other_path_penalty
        )
        
        return np.clip(total_reward, -10, 10)

    # ================ 以下為地圖處理相關方法，保持不變 ================
    
    def map_setup(self, location):
        """設置地圖和機器人初始位置"""
        global_map = (io.imread(location, 1) * 255).astype(int)
        
        # 尋找所有可能的起始位置(值為208的點)
        start_positions = np.where(global_map == 208)
        start_positions = np.array([start_positions[1], start_positions[0]]).T
        
        # ================ 修改：支援多機器人起始位置 ================
        required_robots = getattr(self, 'num_robots', 2)
        
        if len(start_positions) < required_robots:
            # 如果沒有足夠的預定起始點，在自由空間中選擇起始點
            free_space = np.where(global_map > 150)
            free_positions = np.array([free_space[1], free_space[0]]).T
            
            if len(free_positions) < required_robots:
                raise ValueError(f"Map does not have enough free space for {required_robots} robots")
            
            # 智能分配起始位置
            min_distance = 20
            valid_positions = []
            
            # 如果有預定起始點，先使用它們
            if len(start_positions) > 0:
                valid_positions.extend(start_positions.tolist())
            
            # 生成剩餘的起始位置
            while len(valid_positions) < required_robots:
                best_pos = None
                best_min_distance = 0
                
                # 嘗試多次尋找最佳位置
                for _ in range(min(1000, len(free_positions))):
                    candidate = free_positions[np.random.randint(len(free_positions))]
                    
                    # 計算到所有已選位置的最小距離
                    if valid_positions:
                        distances = [np.linalg.norm(candidate - pos) for pos in valid_positions]
                        min_dist = min(distances)
                    else:
                        min_dist = float('inf')
                    
                    # 選擇距離最遠的位置
                    if min_dist > best_min_distance:
                        best_min_distance = min_dist
                        best_pos = candidate
                        
                    # 如果找到滿足最小距離要求的位置，直接使用
                    if min_dist >= min_distance:
                        break
                        
                if best_pos is not None:
                    valid_positions.append(best_pos)
                else:
                    # 如果無法找到理想位置，隨機選擇
                    valid_positions.append(free_positions[np.random.randint(len(free_positions))])
                    
            initial_positions = np.array(valid_positions)
        else:
            # 如果有足夠的預定起始點，使用它們
            initial_positions = start_positions[:required_robots]
        
        # 處理地圖
        global_map = (global_map > 150)
        global_map = global_map * 254 + 1
        
        return global_map, initial_positions

    def map_points(self, map_glo):
        """生成地圖點"""
        map_x = map_glo.shape[1]
        map_y = map_glo.shape[0]
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def local_map(self, robot_location, map_glo, map_size, local_size):
        """獲取局部地圖"""
        minX = int(robot_location[0] - local_size)
        maxX = int(robot_location[0] + local_size)
        minY = int(robot_location[1] - local_size)
        maxY = int(robot_location[1] + local_size)

        minX = max(0, minX)
        maxX = min(map_size[1], maxX)
        minY = max(0, minY)
        maxY = min(map_size[0], maxY)

        return map_glo[minY:maxY, minX:maxX]

    def free_points(self, op_map):
        """提取自由空間點"""
        index = np.where(op_map == 255)
        return np.asarray([index[1], index[0]]).T

    def nearest_free(self, tree, point):
        """尋找最近的自由點"""
        pts = np.atleast_2d(point)
        index = tuple(tree.query(pts)[1])
        return tree.data[index]

    # ================ 以下為執行動作相關方法，保持不變 ================
    
    def execute_movement(self, move_vector):
        """執行移動"""
        old_position = self.robot_position.copy()
        old_op_map = self.op_map.copy()
        
        # 更新位置
        new_position = self.robot_position + move_vector
        self.robot_position = np.round(new_position).astype(np.int64)
        
        # 邊界檢查
        self.robot_position[0] = np.clip(self.robot_position[0], 0, self.map_size[1]-1)
        self.robot_position[1] = np.clip(self.robot_position[1], 0, self.map_size[0]-1)
        
        # 碰撞檢查
        collision_points, collision_index = self.fast_collision_check(
            old_position, self.robot_position, self.map_size, self.global_map)
        
        if collision_index:
            self.robot_position = self.nearest_free(self.free_tree, collision_points)
            reward = REWARD_CONFIG['collision_penalty']
            done = True
        else:
            # 更新共享地圖
            self.op_map = self.inverse_sensor(
                self.robot_position, self.sensor_range, 
                self.op_map, self.global_map
            )
            
            # ================ 修改：支援多機器人距離檢查 ================
            # 避免路徑重疊懲罰
            other_positions = self.get_all_other_robots_positions()
            path_overlap_penalty = 0.0
            for other_pos in other_positions:
                distance_to_other = np.linalg.norm(self.robot_position - other_pos)
                if distance_to_other < ROBOT_CONFIG['sensor_range'] * 2:
                    path_overlap_penalty -= 0.1
            
            reward = self.calculate_fast_reward(old_op_map, self.op_map, move_vector) + path_overlap_penalty
            done = False
        
        self.steps += 1
        if self.plot and self.steps % ROBOT_CONFIG.get('plot_interval', 10) == 0:
            self.xPoint = np.append(self.xPoint, self.robot_position[0])
            self.yPoint = np.append(self.yPoint, self.robot_position[1])
            self.plot_env()
        
        return self.get_observation(), reward, done

    def calculate_fast_reward(self, old_op_map, new_op_map, move_vector):
        """快速獎勵計算"""
        # 1. 探索獎勵
        new_explored = np.sum(new_op_map == 255) - np.sum(old_op_map == 255)
        exploration_reward = new_explored / 14000.0 * REWARD_CONFIG.get('exploration_weight', 1.0)
        
        # 2. 移動效率獎勵
        movement_length = np.linalg.norm(move_vector)
        efficiency_reward = (0 if new_explored > 0 
                            else REWARD_CONFIG.get('movement_penalty', 0.1) * movement_length)

        # 3. 其他機器人路徑懲罰（簡化版）
        other_path_penalty = 0
        current_pos = np.array([self.robot_position[0], self.robot_position[1]])
        
        # ================ 修改：支援多機器人路徑懲罰 ================
        if hasattr(self, 'other_robots') and self.other_robots:
            for other_robot in self.other_robots:
                if hasattr(other_robot, 'robot_position'):
                    distance = np.linalg.norm(current_pos - other_robot.robot_position)
                    safe_distance = ROBOT_CONFIG['sensor_range'] * 1.5
                    if distance < safe_distance:
                        other_path_penalty += -2 * np.exp(-distance/safe_distance)
        elif hasattr(self, 'other_robot_position'):
            # 向後兼容
            distance = np.linalg.norm(current_pos - self.other_robot_position)
            safe_distance = ROBOT_CONFIG['sensor_range'] * 1.5
            if distance < safe_distance:
                other_path_penalty = -2 * np.exp(-distance/safe_distance)
        
        total_reward = exploration_reward + efficiency_reward + other_path_penalty
        return np.clip(total_reward, -5, 5)

    def step(self, action):
        """執行一步動作"""
        self.old_position = self.robot_position.copy()
        self.old_op_map = self.op_map.copy()
        
        # 獲取frontier點
        frontiers = self.get_frontiers()
        
        if len(frontiers) == 0:
            return self.get_observation(), 0, True
            
        # 選擇目標frontier
        target_frontier = frontiers[action % len(frontiers)]
        
        # 移動到frontier
        obs, reward, done = self.move_to_frontier(target_frontier)
        
        self.steps += 1
        
        # 更新路徑記錄
        self.xPoint = np.append(self.xPoint, self.robot_position[0])
        self.yPoint = np.append(self.yPoint, self.robot_position[1])
        
        if self.plot:
            self.plot_env()
            
        return obs, reward, done or self.check_done()