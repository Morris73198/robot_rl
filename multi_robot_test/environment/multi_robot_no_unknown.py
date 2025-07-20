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
        print("Creating robots with shared environment...")

        # 創建第一個機器人，它會加載和初始化地圖
        robot1 = cls(index_map, train, plot, is_primary=True)
        
        # 創建第二個機器人，共享第一個機器人的地圖和相關資源
        robot2 = cls(index_map, train, plot, is_primary=False, shared_env=robot1)
        
        robot1.other_robot = robot2
        print(f"Robot1 other_robot set: {robot1.other_robot is not None}")
        print(f"Robot2 shared_env set: {robot2.shared_env is not None}")
        
        return robot1, robot2
    
    @classmethod
    def create_multi_robots(cls, num_robots, index_map, train=True, plot=True):
        """創建多個共享環境的機器人實例"""
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
        """初始化機器人環境"""
        # 確保li_map屬性存在
        self.li_map = index_map
        self.mode = train
        self.plot = plot
        self.is_primary = is_primary
        self.shared_env = shared_env
        
        # 多機器人支援
        self.robot_id = robot_id if robot_id is not None else (0 if is_primary else 1)
        self.num_robots = num_robots if num_robots is not None else 2
        self.all_robots = []
        self.other_robots = []
        
        # 確保位置都是2D座標
        self.robot_position = np.zeros(2, dtype=np.int64)
        self.other_robot_position = None
        
        if not shared_env and not is_primary:
            raise ValueError("Non-primary robot must have a shared environment")
        if shared_env and is_primary:
            raise ValueError("Primary robot cannot have a shared environment")
        
        self.lethal_cost = 100
        self.decay_factor = 3
        self.inflation_radius = ROBOT_CONFIG['robot_size'] * 1.5
        
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
                
            # 初始化地圖
            self.global_map, self.initial_positions = self.map_setup(
                os.path.join(self.map_dir, self.map_list[self.li_map])
            )
            
            # 確保有足夠的起始位置
            if len(self.initial_positions) < self.num_robots:
                self._generate_additional_positions()
            
            # 設置機器人位置（確保是2D）
            pos = np.array(self.initial_positions[self.robot_id]).flatten()
            self.robot_position = pos[:2].astype(np.int64)
            
            # 向後兼容
            if self.num_robots >= 2:
                other_id = 1 if self.robot_id == 0 else 0
                other_pos = np.array(self.initial_positions[other_id]).flatten()
                self.other_robot_position = other_pos[:2].astype(np.int64)
            
            # 初始化地圖相關資源
            self.map_size = self.global_map.shape
            self.op_map = np.ones(self.global_map.shape) * 127
            self.t = self.map_points(self.global_map)
            self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())
            
        else:
            # 共享環境設置
            self.li_map = self.shared_env.li_map
            self.global_map = self.shared_env.global_map
            self.op_map = self.shared_env.op_map
            self.map_size = self.shared_env.map_size
            self.map_dir = self.shared_env.map_dir
            self.map_list = self.shared_env.map_list
            self.map_number = self.shared_env.map_number
            
            # 使用不同的起始位置（確保是2D）
            if hasattr(self.shared_env, 'initial_positions') and len(self.shared_env.initial_positions) > self.robot_id:
                pos = np.array(self.shared_env.initial_positions[self.robot_id]).flatten()
                self.robot_position = pos[:2].astype(np.int64)
            else:
                shared_pos = np.array(self.shared_env.other_robot_position).flatten()
                self.robot_position = shared_pos[:2].copy()
            
            primary_pos = np.array(self.shared_env.robot_position).flatten()
            self.other_robot_position = primary_pos[:2].copy()
            
            # 共享資源
            self.t = self.shared_env.t
            self.free_tree = self.shared_env.free_tree
        
        # 機器人配置
        self.robot_size = ROBOT_CONFIG['robot_size']
        self.sensor_range = ROBOT_CONFIG['sensor_range']
        self.local_size = ROBOT_CONFIG['local_size']
        self.movement_step = ROBOT_CONFIG['movement_step']
        self.finish_percent = ROBOT_CONFIG['finish_percent']
        
        # 初始化狀態
        self.old_position = np.zeros(2, dtype=np.float64)
        self.old_op_map = None  # 將在begin()方法中正確初始化
        self.current_target_frontier = None
        self.is_moving_to_target = False
        self.current_path = None
        self.current_path_index = 0
        self.steps = 0
        
        # 可視化相關
        if self.plot:
            self.xPoint = np.array([])
            self.yPoint = np.array([])
            self.fig = None
            self.initialize_visualization()

    def _generate_additional_positions(self):
        """生成額外的起始位置"""
        free_positions = self.free_points(self.global_map)
        needed_positions = self.num_robots - len(self.initial_positions)
        min_distance = self.robot_size * 3
        
        existing_positions = list(self.initial_positions)
        additional_positions = []
        
        for _ in range(needed_positions):
            max_attempts = 100
            best_pos = None
            best_min_distance = 0
            
            for _ in range(max_attempts):
                candidate = free_positions[np.random.randint(len(free_positions))]
                
                # 計算與現有位置的最小距離
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

    def get_all_other_robots_positions(self):
        """獲取所有其他機器人的位置（確保返回2D座標）"""
        positions = []
        
        if not hasattr(self, 'other_robots') or not self.other_robots:
            # 向後兼容：返回單個其他機器人位置
            if hasattr(self, 'other_robot_position') and self.other_robot_position is not None:
                pos = np.array(self.other_robot_position).flatten()
                
                # 確保是2D座標
                if len(pos) >= 2:
                    pos = pos[:2]
                    positions.append(pos.astype(np.float64))
            
            return positions
        
        for robot in self.other_robots:
            if robot is not None and hasattr(robot, 'robot_position'):
                pos = np.array(robot.robot_position).flatten()
                
                # 確保是2D座標
                if len(pos) >= 2:
                    pos = pos[:2]
                    positions.append(pos.astype(np.float64))
        
        return positions
    
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
        # 強制轉換：確保target_position是2D座標
        target_position = np.array(target_position).flatten()
        
        # 只取前兩個元素作為x, y座標
        if len(target_position) >= 2:
            target_position = target_position[:2]
        else:
            return target_position
        
        target_position = target_position.astype(np.float64)
        
        other_positions = self.get_all_other_robots_positions()
        min_distance = self.robot_size * 2
        
        for other_pos in other_positions:
            # 強制轉換：確保other_pos也是2D座標
            other_pos = np.array(other_pos).flatten()
            
            # 只取前兩個元素作為x, y座標
            if len(other_pos) >= 2:
                other_pos = other_pos[:2]
            else:
                continue
                
            other_pos = other_pos.astype(np.float64)
            
            distance = np.linalg.norm(target_position - other_pos)
            if distance < min_distance:
                # 計算避開的方向
                direction = target_position - other_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    target_position = other_pos + direction * min_distance
                    
        return target_position

    def get_other_robot_pos(self):
        """獲取另一個機器人的位置（確保返回2D座標）"""
        if hasattr(self, 'other_robot_position') and self.other_robot_position is not None:
            pos = np.array(self.other_robot_position).flatten()
            return pos[:2] if len(pos) >= 2 else pos
        elif hasattr(self, 'other_robots') and self.other_robots:
            pos = np.array(self.other_robots[0].robot_position).flatten()
            return pos[:2] if len(pos) >= 2 else pos
        return None

    def update_other_robot_pos(self, pos):
        """更新另一個機器人的位置（確保是2D座標）"""
        if hasattr(self, 'other_robot_position'):
            pos = np.array(pos).flatten()
            self.other_robot_position = pos[:2] if len(pos) >= 2 else pos

    def get_normalized_position(self):
        """獲取標準化位置（2D座標）"""
        pos = np.array(self.robot_position).flatten()
        pos_2d = pos[:2] if len(pos) >= 2 else pos
        map_dims = np.array([float(self.map_size[1]), float(self.map_size[0])])
        return pos_2d / map_dims

    @property
    def other_robot(self):
        """向後兼容：獲取另一個機器人"""
        if hasattr(self, 'other_robots') and self.other_robots:
            return self.other_robots[0]
        elif hasattr(self, '_other_robot'):
            return self._other_robot
        return None
    
    @other_robot.setter
    def other_robot(self, value):
        """向後兼容：設置另一個機器人"""
        self._other_robot = value
        
    def cleanup_visualization(self):
        """清理可視化資源"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            plt.clf()
            self.fig = None

    def robot_model(self, position, robot_size, points, map_glo):
        """機器人模型"""
        map_copy = map_glo.copy()
        x, y = int(position[0]), int(position[1])
        if 0 <= x < map_copy.shape[1] and 0 <= y < map_copy.shape[0]:
            map_copy[y, x] = 76
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
        
        # 路徑檢查
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return np.array([[-1, -1]]).reshape(1, 2), False
            
        x_step = dx / steps
        y_step = dy / steps
        
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
        """逆向傳感器模型"""
        return inverse_sensor_model(
            int(robot_position[0]), int(robot_position[1]), 
            sensor_range, op_map, map_glo)

    def frontier(self, op_map, map_size, points):
        """前沿檢測"""
        y_len, x_len = map_size
        mapping = (op_map == 127).astype(int)
        
        mapping = np.pad(mapping, ((1,1), (1,1)), 'constant')
        
        fro_map = (
            mapping[2:, 1:x_len+1] +    # 下
            mapping[:y_len, 1:x_len+1] +  # 上
            mapping[1:y_len+1, 2:] +      # 右
            mapping[1:y_len+1, :x_len] +  # 左
            mapping[2:, 2:] +             # 右下
            mapping[2:, :x_len] +         # 左下
            mapping[:y_len, 2:] +         # 右上
            mapping[:y_len, :x_len]       # 左上
        )
        
        ind_map = np.logical_and(fro_map > 0, op_map == 255)
        
        # 獲取frontier點
        frontier_y, frontier_x = np.where(ind_map)
        frontiers = np.column_stack((frontier_x, frontier_y))
        
        # 聚類frontier點
        if len(frontiers) > 0:
            clustered_frontiers = self.cluster_frontiers(frontiers)
            return clustered_frontiers
        
        return np.array([])

    def cluster_frontiers(self, frontiers, min_cluster_size=5, cluster_radius=10):
        """聚類frontier點"""
        if len(frontiers) == 0:
            return np.array([])
        
        clusters = []
        used = np.zeros(len(frontiers), dtype=bool)
        
        for i, frontier in enumerate(frontiers):
            if used[i]:
                continue
                
            # 找到附近的點
            distances = np.linalg.norm(frontiers - frontier, axis=1)
            nearby = distances <= cluster_radius
            cluster_points = frontiers[nearby]
            
            if len(cluster_points) >= min_cluster_size:
                # 計算聚類中心
                center = np.mean(cluster_points, axis=0)
                clusters.append(center)
                used[nearby] = True
        
        return np.array(clusters)

    def get_frontiers(self):
        """獲取frontier點，確保返回2D座標"""
        frontiers = self.frontier(self.op_map, self.map_size, self.t)
        
        # 確保所有frontier點都是2D座標
        if len(frontiers) > 0:
            frontiers_2d = []
            for frontier in frontiers:
                frontier = np.array(frontier).flatten()
                if len(frontier) >= 2:
                    frontiers_2d.append(frontier[:2])
            return np.array(frontiers_2d) if frontiers_2d else np.array([])
        
        return frontiers

    def astar_path(self, op_map, start, goal, safety_distance=1):
        """A*路徑規劃"""
        # 確保start和goal是2D座標
        start = np.array(start).flatten()[:2].astype(np.int32)
        goal = np.array(goal).flatten()[:2].astype(np.int32)
        
        # 檢查起點和終點是否有效
        if (start[0] < 0 or start[0] >= op_map.shape[1] or 
            start[1] < 0 or start[1] >= op_map.shape[0] or
            goal[0] < 0 or goal[0] >= op_map.shape[1] or 
            goal[1] < 0 or goal[1] >= op_map.shape[0]):
            return None
        
        # A*算法實現
        open_set = []
        heappush(open_set, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): np.linalg.norm(start - goal)}
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while open_set:
            current = heappop(open_set)[1]
            current_array = np.array(current)
            
            if np.linalg.norm(current_array - goal) < 2:
                # 重建路徑
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(tuple(start))
                return np.array(path[::-1])
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (neighbor[0] < 0 or neighbor[0] >= op_map.shape[1] or
                    neighbor[1] < 0 or neighbor[1] >= op_map.shape[0]):
                    continue
                
                # 檢查是否為已知的自由空間
                if op_map[neighbor[1], neighbor[0]] != 255:
                    continue
                
                tentative_g_score = g_score[current] + np.linalg.norm([dx, dy])
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None

    def simplify_path(self, path, threshold=2.0):
        """簡化路徑"""
        if path is None or len(path) <= 2:
            return path
        
        # 確保路徑點都是2D
        path_2d = []
        for point in path:
            point = np.array(point).flatten()
            if len(point) >= 2:
                path_2d.append(point[:2])
        
        if len(path_2d) <= 2:
            return np.array(path_2d)
        
        path = np.array(path_2d)
        
        def point_line_distance(point, line_start, line_end):
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                return np.linalg.norm(point_vec)
            line_unit_vec = line_vec / line_len
            projection_length = np.dot(point_vec, line_unit_vec)
            projection_length = max(min(projection_length, line_len), 0)
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
        
        mask = np.ones(len(path), dtype=bool)
        simplify_recursive(path, threshold, mask)
        
        return path[mask]

    def move_to_frontier(self, target_frontier):
        """移動到frontier，一次移動一步，直到到達目標或確定無法到達"""
        # 確保target_frontier是2D座標
        target_frontier = np.array(target_frontier).flatten()
        if len(target_frontier) >= 2:
            target_frontier = target_frontier[:2]
        else:
            return self.get_observation(), -1, True
        target_frontier = target_frontier.astype(np.float64)
        
        # 保存當前目標
        self.current_target_frontier = target_frontier
        
        # 如果沒有當前路徑或需要重新規劃路徑
        if not hasattr(self, 'current_path') or self.current_path is None:
            # 檢查目標是否可達
            known_path = self.astar_path(
                self.op_map,
                self.robot_position.astype(np.int32),
                target_frontier.astype(np.int32),
                safety_distance=ROBOT_CONFIG['safety_distance']
            )
            
            if known_path is None:
                # 無法找到路徑
                self.current_path = None
                return self.get_observation(), -1, True
                
            # 保存路徑
            self.current_path = self.simplify_path(known_path)
            self.current_path_index = 0
            
        # 執行一步移動
        if self.current_path_index < len(self.current_path) - 1:
            self.current_path_index += 1
            next_position = np.array(self.current_path[self.current_path_index])
            
            # 修復：確保next_position是2D座標
            next_position = np.array(next_position).flatten()
            if len(next_position) >= 2:
                next_position = next_position[:2]
            else:
                return self.get_observation(), -1, True
            next_position = next_position.astype(np.float64)
            
            # 避免與其他機器人碰撞
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

    def execute_movement(self, move_vector):
        """移動"""
        # 確保move_vector是2D
        move_vector = np.array(move_vector).flatten()
        if len(move_vector) >= 2:
            move_vector = move_vector[:2]
        else:
            return self.get_observation(), REWARD_CONFIG['collision_penalty'], True
        
        old_position = self.robot_position.copy()
        old_op_map = self.op_map.copy()
        
        # 更新位置
        new_position = self.robot_position + move_vector
        self.robot_position = np.round(new_position).astype(np.int64)
        
        # 確保robot_position是2D
        if len(self.robot_position) > 2:
            self.robot_position = self.robot_position[:2]
        
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
        # 確保輸入地圖的形狀一致
        if old_op_map.shape != new_op_map.shape:
            # 如果舊地圖無效，使用新地圖作為基準
            old_op_map = np.ones_like(new_op_map) * 127
        
        # 1. 探索獎勵
        new_explored = np.sum(new_op_map == 255) - np.sum(old_op_map == 255)
        exploration_reward = new_explored / 14000.0 * REWARD_CONFIG.get('exploration_weight', 1.0)
        
        # 2. 移動效率獎勵
        movement_length = np.linalg.norm(move_vector)
        efficiency_reward = (0 if new_explored > 0 
                            else REWARD_CONFIG.get('movement_penalty', 0.1) * movement_length)

        # 3. 協作獎勵
        cooperation_reward = 0.0
        if hasattr(self, 'other_robot') and self.other_robot is not None:
            if hasattr(self.other_robot, 'op_map'):
                try:
                    other_explored = np.sum(self.other_robot.op_map == 255)
                    total_explored = np.sum(new_op_map == 255)
                    if total_explored > 0:
                        cooperation_reward = 0.1 * (other_explored / total_explored)
                except:
                    cooperation_reward = 0.0
        
        return exploration_reward - efficiency_reward + cooperation_reward

    def calculate_reward(self):
        """計算獎勵，確保位置處理正確"""
        # 確保當前位置是2D
        current_pos = np.array(self.robot_position).flatten()[:2]
        
        # 基本探索獎勵
        new_explored = np.sum(self.op_map == 255)
        
        # 檢查old_op_map是否有效
        if hasattr(self, 'old_op_map') and len(self.old_op_map) > 0 and self.old_op_map.shape == self.op_map.shape:
            old_explored = np.sum(self.old_op_map == 255)
            exploration_reward = (new_explored - old_explored) / 1000.0
        else:
            # 如果old_op_map無效，只計算當前探索量
            exploration_reward = new_explored / 10000.0
        
        # 移動效率獎勵
        movement_penalty = 0
        if hasattr(self, 'old_position') and len(self.old_position) >= 2:
            old_pos = np.array(self.old_position).flatten()[:2]
            if len(old_pos) == 2:
                movement_distance = np.linalg.norm(current_pos - old_pos)
                movement_penalty = -0.01 * movement_distance
        
        # 與其他機器人的距離獎勵/懲罰
        distance_penalty = 0
        other_positions = self.get_all_other_robots_positions()
        for other_pos in other_positions:
            distance = np.linalg.norm(current_pos - other_pos)
            if distance < self.robot_size * 3:
                distance_penalty -= 0.1  # 太近懲罰
            elif distance > self.sensor_range * 3:
                distance_penalty -= 0.05  # 太遠懲罰
        
        total_reward = exploration_reward + movement_penalty + distance_penalty
        
        # 更新old_op_map以供下次使用
        self.old_op_map = self.op_map.copy()
        self.old_position = current_pos.copy()
        
        return total_reward

    def check_completion(self):
        """檢查探索是否完成"""
        exploration_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        
        if exploration_ratio > self.finish_percent:
            self.li_map += 1
            if self.li_map >= self.map_number:
                self.li_map = 0
                return True, True  # 完成所有地圖
                
            self.__init__(self.li_map, self.mode, self.plot)
            return True, False  # 完成當前地圖
            
        return False, False
    
    def check_done(self):
        """檢查是否需要結束當前回合"""
        # 檢查探索進度
        exploration_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        if exploration_ratio > self.finish_percent:
            return True
            
        # 檢查是否還有可探索的frontiers
        frontiers = self.get_frontiers()
        if len(frontiers) == 0:
            return True
            
        return False

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
            
            # 確保有足夠的起始位置
            if len(self.initial_positions) < self.num_robots:
                self._generate_additional_positions()
            
            # 設置新的起始位置（確保是2D）
            pos = np.array(self.initial_positions[self.robot_id]).flatten()
            self.robot_position = pos[:2].astype(np.int64)
            
            # 保持向後兼容
            if self.num_robots >= 2:
                other_id = 1 if self.robot_id == 0 else 0
                other_pos = np.array(self.initial_positions[other_id]).flatten()
                self.other_robot_position = other_pos[:2].astype(np.int64)
            
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
            
            # 使用不同的起始位置（確保是2D）
            if hasattr(self.shared_env, 'initial_positions') and len(self.shared_env.initial_positions) > self.robot_id:
                pos = np.array(self.shared_env.initial_positions[self.robot_id]).flatten()
                self.robot_position = pos[:2].astype(np.int64)
            else:
                shared_pos = np.array(self.shared_env.other_robot_position).flatten()
                self.robot_position = shared_pos[:2].copy()
            
            self_pos = np.array(self.shared_env.robot_position).flatten()
            self.other_robot_position = self_pos[:2].copy()
            
            # 共享KD樹和空閒空間點
            self.t = self.shared_env.t
            self.free_tree = self.shared_env.free_tree
        
        # 重置路徑規劃和frontier相關的狀態
        self.old_position = np.zeros(2, dtype=np.float64)
        self.old_op_map = None  # 將在begin()方法中正確初始化
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

    def begin(self):
        """初始化並返回初始狀態"""
        self.op_map = self.inverse_sensor(
            self.robot_position, self.sensor_range, self.op_map, self.global_map)
        
        # 初始化old_op_map和old_position
        self.old_op_map = self.op_map.copy()
        self.old_position = self.robot_position.copy()
            
        step_map = self.robot_model(
            self.robot_position, self.robot_size, self.t, self.op_map)
            
        resized_map = resize(step_map, (84, 84))
        state = np.expand_dims(resized_map, axis=-1)
        
        if self.plot:
            self.plot_env()
            
        return state

    def get_observation(self):
        """獲取當前觀察狀態"""
        step_map = self.robot_model(
            self.robot_position, self.robot_size, self.t, self.op_map)
            
        resized_map = resize(step_map, (84, 84))
        state = np.expand_dims(resized_map, axis=-1)
        
        return state

    def map_setup(self, map_dir):
        """地圖設置"""
        # 載入地圖
        map_img = io.imread(map_dir, as_gray=True)
        binary_map = (map_img > 0.5).astype(np.uint8)
        
        # 找到自由空間
        free_space = (binary_map == 1)
        free_points = np.column_stack(np.where(free_space))
        
        # 隨機選擇起始位置
        num_start_positions = max(2, self.num_robots)
        if len(free_points) < num_start_positions:
            raise ValueError("Not enough free space for robots")
        
        # 選擇起始位置
        indices = np.random.choice(len(free_points), size=num_start_positions, replace=False)
        start_positions = free_points[indices]
        
        # 轉換為x,y格式
        start_positions = start_positions[:, [1, 0]]  # y,x -> x,y
        
        return binary_map, start_positions

    def map_points(self, map_glo):
        """獲取地圖點"""
        map_x, map_y = np.meshgrid(np.arange(map_glo.shape[1]), np.arange(map_glo.shape[0]))
        return np.column_stack((map_x.ravel(), map_y.ravel()))

    def free_points(self, map_glo):
        """獲取自由空間點"""
        free_y, free_x = np.where(map_glo == 1)
        return np.column_stack((free_x, free_y))

    def nearest_free(self, free_tree, collision_points):
        """找到最近的自由空間點"""
        if len(collision_points) == 0:
            return self.robot_position
        
        collision_point = collision_points[0]
        _, nearest_idx = free_tree.query(collision_point)
        return free_tree.data[nearest_idx]

    def initialize_visualization(self):
        """初始化可視化"""
        if not self.plot:
            return
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.xPoint = np.array([])
        self.yPoint = np.array([])

    def plot_env(self):
        """繪製環境"""
        if not self.plot or not hasattr(self, 'fig'):
            return
        
        self.ax.clear()
        
        # 繪製地圖
        self.ax.imshow(self.op_map, cmap='gray', origin='lower')
        
        # 繪製機器人位置
        self.ax.plot(self.robot_position[0], self.robot_position[1], 'ro', markersize=10, label='Robot')
        
        # 繪製軌跡
        if len(self.xPoint) > 0:
            self.ax.plot(self.xPoint, self.yPoint, 'r-', alpha=0.7, label='Path')
        
        # 繪製其他機器人
        if hasattr(self, 'other_robot_position') and self.other_robot_position is not None:
            self.ax.plot(self.other_robot_position[0], self.other_robot_position[1], 'bo', markersize=10, label='Other Robot')
        
        # 繪製frontiers
        frontiers = self.get_frontiers()
        if len(frontiers) > 0:
            self.ax.scatter(frontiers[:, 0], frontiers[:, 1], c='yellow', s=20, alpha=0.7, label='Frontiers')
        
        self.ax.legend()
        self.ax.set_title(f'Robot {self.robot_id} - Step {self.steps}')
        plt.pause(0.001)



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