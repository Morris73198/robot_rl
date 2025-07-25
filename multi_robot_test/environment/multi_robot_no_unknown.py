import os
import numpy as np
import numpy.ma as ma
from scipy import spatial
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') 
from ..utils.inverse_sensor_model import inverse_sensor_model
from scipy.ndimage import distance_transform_edt
import random
from heapq import heappush, heappop
from ..config import ROBOT_CONFIG, REWARD_CONFIG

import heapq


class Robot:
    @classmethod
    def create_shared_robots(cls, index_map, num_robots=5, train=True, plot=True):
        """創建共享環境的機器人實例
        
        Args:
            index_map: 地圖索引
            num_robots: 機器人數量（默認為2）
            train: 是否處於訓練模式
            plot: 是否繪製可視化
            
        Returns:
            list: 機器人實例列表
        """
        if num_robots < 1:
            raise ValueError("Number of robots must be at least 1")
            
        print(f"Creating {num_robots} robots with shared environment...")

        # 創建第一個機器人，它會加載和初始化地圖
        robots = []
        robot1 = cls(index_map, train, plot, is_primary=True, robot_id=0, num_robots=num_robots)
        robots.append(robot1)
        
        # 創建其他機器人，共享第一個機器人的地圖和相關資源
        for i in range(1, num_robots):
            robot = cls(index_map, train, plot, is_primary=False, shared_env=robot1, 
                       robot_id=i, num_robots=num_robots)
            robots.append(robot)
        
        # 設置機器人間的引用關係
        for i, robot in enumerate(robots):
            robot.other_robots = [r for j, r in enumerate(robots) if j != i]
            robot.other_robots_positions = [r.robot_position.copy() for r in robot.other_robots]
            
        print(f"Successfully created {num_robots} robots")
        for i, robot in enumerate(robots):
            print(f"Robot{i} other_robots count: {len(robot.other_robots)}")
        
        return robots
    
    def __init__(self, index_map, train, plot, is_primary=True, shared_env=None, 
                 robot_id=0, num_robots=2):
        """初始化機器人環境
        
        Args:
            index_map: 地圖索引
            train: 是否處於訓練模式
            plot: 是否繪製可視化
            is_primary: 是否為主要機器人(負責加載地圖)
            shared_env: 共享環境的機器人實例
            robot_id: 機器人ID
            num_robots: 總機器人數量
        """
        if not shared_env and not is_primary:
            raise ValueError("Non-primary robot must have a shared environment")
        if shared_env and is_primary:
            raise ValueError("Primary robot cannot have a shared environment")
            
        self.mode = train
        self.plot = plot
        self.is_primary = is_primary
        self.robot_id = robot_id
        self.num_robots = num_robots
        self.shared_env = shared_env 
        
        # 初始化其他機器人相關屬性
        self.other_robots = []
        self.other_robots_positions = []
        
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
            
            # 為所有機器人選擇不同的起始位置
            self.robot_position = self.initial_positions[0].astype(np.int64)
            # 存儲所有機器人的初始位置
            self.all_robots_initial_positions = [pos.astype(np.int64) for pos in self.initial_positions]
            
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
            
            # 使用對應的起始位置
            self.robot_position = shared_env.all_robots_initial_positions[robot_id].copy()
            self.all_robots_initial_positions = shared_env.all_robots_initial_positions
            
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
                
                
                
    def begin(self):
        """初始化並返回初始狀態"""
        self.op_map = self.inverse_sensor(
            self.robot_position, self.sensor_range, self.op_map, self.global_map)
            
        step_map = self.robot_model(
            self.robot_position, self.robot_size, self.t, self.op_map)
            
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
            self.current_path = self.simplify_path(known_path, ROBOT_CONFIG['path_simplification'])
            self.current_path_index = 0
            
            # 立即更新可視化以顯示新規劃的路徑
            if self.plot:
                self.plot_env()

        # 檢查是否還有路徑點要處理
        if self.current_path_index >= len(self.current_path.T):
            # 路徑執行完畢，檢查是否到達目標
            dist_to_target = np.linalg.norm(self.robot_position - target_frontier)
            if dist_to_target < ROBOT_CONFIG['target_reach_threshold']:
                # 成功到達目標
                self.current_path = None
                self.current_target_frontier = None
                return self.get_observation(), 1.0, True
            else:
                # 需要重新規劃路徑
                self.current_path = None
                return self.get_observation(), -0.1, True

        # 獲取下一個路徑點
        next_point = self.current_path[:, self.current_path_index]
        
        # 計算移動向量
        move_vector = next_point - self.robot_position
        dist = np.linalg.norm(move_vector)
        
        # 確保最小移動距離
        MIN_MOVEMENT = 1.0  # 最小移動距離
        if dist < MIN_MOVEMENT:
            self.current_path_index += 1
            return self.get_observation(), 0, False
        
        # 調整步長
        if dist > ROBOT_CONFIG['movement_step']:
            move_vector = move_vector * (ROBOT_CONFIG['movement_step'] / dist)
        
        # 執行一步移動
        old_position = self.robot_position.copy()
        old_op_map = self.op_map.copy()
        
        # 更新位置
        new_position = self.robot_position + move_vector
        self.robot_position = np.round(new_position).astype(np.int64)
        
        # 記錄路徑點
        self.xPoint = np.append(self.xPoint, self.robot_position[0])
        self.yPoint = np.append(self.yPoint, self.robot_position[1])
        
        # 邊界檢查
        self.robot_position[0] = np.clip(self.robot_position[0], 0, self.map_size[1]-1)
        self.robot_position[1] = np.clip(self.robot_position[1], 0, self.map_size[0]-1)
        
        # 碰撞檢查
        collision_points, collision_index = self.fast_collision_check(
            old_position, self.robot_position, self.map_size, self.global_map
        )
        
        if collision_index:
            # 發生碰撞，任務失敗
            self.robot_position = self.nearest_free(self.free_tree, collision_points)
            self.current_path = None
            self.current_target_frontier = None
            return self.get_observation(), REWARD_CONFIG['collision_penalty'], True
        
        # 更新地圖和獎勵
        self.op_map = self.inverse_sensor(
            self.robot_position, self.sensor_range, 
            self.op_map, self.global_map
        )
        
        reward = self.calculate_fast_reward(old_op_map, self.op_map, move_vector)
        
        # 更新路徑索引
        if dist <= ROBOT_CONFIG['movement_step']:
            self.current_path_index += 1
        
        # 檢查是否發現了新的障礙物，需要重新規劃路徑
        if self.should_replan_path(self.current_path[:, self.current_path_index:]):
            self.current_path = None
            return self.get_observation(), reward, True
        
        # 更新可視化
        self.steps += 1
        if self.plot and self.steps % ROBOT_CONFIG['plot_interval'] == 0:
            self.plot_env()
        
        # 繼續執行，未完成
        return self.get_observation(), reward, False


    def should_replan_path(self, remaining_path):
        """檢查是否需要重新規劃路徑"""
        if len(remaining_path.T) == 0:
            return True
            
        # 檢查剩餘路徑是否被阻擋
        for i in range(len(remaining_path.T) - 1):
            start = remaining_path[:, i]
            end = remaining_path[:, i + 1]
            collision_points, collision_index = self.fast_collision_check(
                start, end, self.map_size, self.op_map
            )
            if collision_index:
                return True
                
        return False

    def check_path_blocked(self, path_points):
        """檢查路徑是否被阻擋"""
        if len(path_points) < 2:
            return False
            
        for i in range(len(path_points) - 1):
            start = path_points[i]
            end = path_points[i + 1]
            
            collision_points, collision_index = self.fast_collision_check(
                start, end, self.map_size, self.op_map
            )
            
            if collision_index:
                return True
                
        return False

    def execute_movement(self, move_vector):
        """移動"""
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
            
            # 計算與其他機器人的距離懲罰
            path_overlap_penalty = 0.0
            for other_pos in self.other_robots_positions:
                distance_to_other = np.linalg.norm(self.robot_position - other_pos)
                if distance_to_other < ROBOT_CONFIG['sensor_range'] * 2:
                    path_overlap_penalty += -0.1
            
            reward = self.calculate_fast_reward(old_op_map, self.op_map, move_vector) + path_overlap_penalty
            done = False
        
        self.steps += 1
        if self.plot and self.steps % ROBOT_CONFIG['plot_interval'] == 0:
            self.xPoint = np.append(self.xPoint, self.robot_position[0])
            self.yPoint = np.append(self.yPoint, self.robot_position[1])
            self.plot_env()
        
        return self.get_observation(), reward, done

    def calculate_fast_reward(self, old_op_map, new_op_map, move_vector):
        """計算獎勵，考慮所有其他機器人"""
        # 現有的獎勵組件
        new_explored = np.sum(new_op_map == 255) - np.sum(old_op_map == 255)
        exploration_reward = new_explored / 14000.0 * REWARD_CONFIG['exploration_weight']
        
        movement_length = np.linalg.norm(move_vector)
        efficiency_reward = (0 if new_explored > 0 
                            else REWARD_CONFIG['movement_penalty'] * movement_length)

        other_path_penalty = 0
        current_pos = np.array([self.robot_position[0], self.robot_position[1]])
        
        # 計算與所有其他機器人路徑的懲罰
        for other_robot in self.other_robots:
            if (hasattr(other_robot, 'xPoint') and 
                hasattr(other_robot, 'yPoint') and 
                len(other_robot.xPoint) > 0):
                
                recent_history = 500
                start_idx = max(0, len(other_robot.xPoint) - recent_history)
                
                other_path = np.column_stack((
                    other_robot.xPoint[start_idx:],
                    other_robot.yPoint[start_idx:]
                ))
                
                distances = np.linalg.norm(other_path - current_pos, axis=1)
                min_distance = np.min(distances)
                
                safe_distance = ROBOT_CONFIG['sensor_range'] * 1.5
                if min_distance < safe_distance:
                    other_path_penalty += -4 * np.exp(-min_distance/safe_distance)
                else:
                    other_path_penalty += 1
        
        # 計算與其他機器人的協同距離獎勵
        distance_reward = 0
        if len(self.other_robots_positions) > 0:
            for other_pos in self.other_robots_positions:
                distance_to_other = np.linalg.norm(self.robot_position - other_pos)
                optimal_distance = self.sensor_range * 2
                distance_reward += -0.5 * abs(distance_to_other - optimal_distance) / optimal_distance
            # 平均化距離獎勵
            distance_reward /= len(self.other_robots_positions)
        
        # 新增：Local map 交集懲罰
        overlap_penalty = 0
        if hasattr(self, 'map_tracker') and self.map_tracker is not None:
            overlap_ratio = self.map_tracker.calculate_overlap()
            overlap_penalty = -30.0 * overlap_ratio
        
        # 將新的交集懲罰加入總獎勵
        total_reward = (
            exploration_reward +
            efficiency_reward +
            other_path_penalty +
            overlap_penalty
        )
        
        return np.clip(total_reward, -10, 10)

    def map_setup(self, location):
        """設置地圖和機器人初始位置，支援任意數量機器人"""
        global_map = (io.imread(location, 1) * 255).astype(int)
        
        # 尋找所有可能的起始位置(值為208的點)
        start_positions = np.where(global_map == 208)
        start_positions = np.array([start_positions[1], start_positions[0]]).T
        
        if len(start_positions) < self.num_robots:
            # 如果沒有足夠的預定起始點，在自由空間中選擇起始點
            free_space = np.where(global_map > 150)
            free_positions = np.array([free_space[1], free_space[0]]).T
            
            if len(free_positions) < self.num_robots:
                raise ValueError(f"Map does not have enough free space for {self.num_robots} robots")
            
            # 為所有機器人選擇相距足夠遠的點
            min_distance = 20  # 最小距離閾值
            valid_positions = []
            
            # 選擇第一個位置
            first_pos = free_positions[np.random.randint(len(free_positions))]
            valid_positions.append(first_pos)
            
            # 為其餘機器人選擇位置
            for i in range(1, self.num_robots):
                max_attempts = 1000
                attempts = 0
                found_valid_pos = False
                
                while attempts < max_attempts and not found_valid_pos:
                    candidate_pos = free_positions[np.random.randint(len(free_positions))]
                    
                    # 檢查與所有已選位置的距離
                    valid = True
                    for existing_pos in valid_positions:
                        if np.linalg.norm(candidate_pos - existing_pos) < min_distance:
                            valid = False
                            break
                    
                    if valid:
                        valid_positions.append(candidate_pos)
                        found_valid_pos = True
                    
                    attempts += 1
                
                if not found_valid_pos:
                    # 如果找不到足夠遠的位置，降低距離要求
                    min_distance *= 0.8
                    print(f"Warning: Reducing minimum distance to {min_distance} for robot {i}")
                    # 重新嘗試
                    for candidate_pos in free_positions:
                        valid = True
                        for existing_pos in valid_positions:
                            if np.linalg.norm(candidate_pos - existing_pos) < min_distance:
                                valid = False
                                break
                        if valid:
                            valid_positions.append(candidate_pos)
                            break
                    else:
                        raise ValueError(f"Could not find suitable starting positions for {self.num_robots} robots")
            
            initial_positions = np.array(valid_positions)
        else:
            # 如果有足夠的預定起始點，使用它們
            initial_positions = start_positions[:self.num_robots]
        
        # 處理地圖
        global_map = (global_map > 150)
        global_map = global_map * 254 + 1
        
        return global_map, initial_positions

    def map_points(self, map_glo):
        """生成地圖"""
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
        index = np.where(op_map == 255)
        return np.asarray([index[1], index[0]]).T

    def nearest_free(self, tree, point):
        pts = np.atleast_2d(point)
        index = tuple(tree.query(pts)[1])
        return tree.data[index]

    def robot_model(self, position, robot_size, points, map_glo):
        map_copy = map_glo.copy()
        return map_copy

    def range_search(self, position, r, points):
        diff = points - position
        dist_sq = np.sum(diff * diff, axis=1)
        return points[dist_sq <= r * r]

    def fast_collision_check(self, start_point, end_point, map_size, map_glo):
        start = np.round(start_point).astype(int)
        end = np.round(end_point).astype(int)
        
        if not (0 <= end[0] < map_size[1] and 0 <= end[1] < map_size[0]):
            return np.array([end]).reshape(1, 2), True
            
        if map_glo[end[1], end[0]] == 1:
            return np.array([end]).reshape(1, 2), True
        
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
        return inverse_sensor_model(
            int(robot_position[0]), int(robot_position[1]), 
            sensor_range, op_map, map_glo)

    def frontier(self, op_map, map_size, points):
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
        """取得當前可用的frontier點，考慮所有其他機器人的位置"""
        if self.is_moving_to_target and self.current_target_frontier is not None:
            return np.array([self.current_target_frontier])
            
        frontiers = self.frontier(self.op_map, self.map_size, self.t)
        if len(frontiers) == 0:
            return np.zeros((0, 2))
            
        # 計算到自己的距離
        distances = np.linalg.norm(frontiers - self.robot_position, axis=1)
        
        # 計算到所有其他機器人的距離，並取平均
        if len(self.other_robots_positions) > 0:
            other_distances_sum = np.zeros(len(frontiers))
            for other_pos in self.other_robots_positions:
                other_distances_sum += np.linalg.norm(frontiers - other_pos, axis=1)
            other_distances = other_distances_sum / len(self.other_robots_positions)
        else:
            other_distances = np.zeros(len(frontiers))
        
        # 根據距離對frontier進行排序
        scores = distances - 0.5 * other_distances
        sorted_indices = np.argsort(scores)
        
        return frontiers[sorted_indices]

    def plot_env(self):
        """繪製環境和所有機器人"""
        plt.figure(self.fig.number)
        plt.clf()
        
        # 1. 繪製基礎地圖
        plt.imshow(self.op_map, cmap='gray')
        plt.axis((0, self.map_size[1], self.map_size[0], 0))
        
        # 2. 定義顏色列表
        colors = ['#800080', '#FFA500', '#FF6347', '#4169E1', '#32CD32', 
                 '#FFD700', '#FF69B4', '#00CED1', '#FF4500', '#9370DB']
        path_color = colors[self.robot_id % len(colors)]
        
        # 3. 繪製路徑軌跡
        if len(self.xPoint) > 1:
            plt.plot(self.xPoint, self.yPoint, color=path_color, 
                    linewidth=2, label=f'Robot{self.robot_id} Path')
        
        # 4. 繪製 frontier 點
        frontiers = self.get_frontiers()
        if len(frontiers) > 0:
            plt.scatter(frontiers[:, 0], frontiers[:, 1], 
                    c='red', marker='*', s=100, label='Frontiers')
        
        # 5. 繪製目標 frontier 和規劃路徑
        if self.current_target_frontier is not None:
            plt.plot(self.current_target_frontier[0], self.current_target_frontier[1], 
                    marker='^', color=path_color, markersize=15, 
                    label=f'Robot{self.robot_id} Target')
            
            if self.current_path is not None and self.current_path.shape[1] > self.current_path_index:
                remaining_path = self.current_path[:, self.current_path_index:]
                plt.plot(remaining_path[0, :], remaining_path[1, :], '--', 
                        color=path_color, linewidth=2, alpha=0.8, label='Planned Path')
                
                if remaining_path.shape[1] > 1:
                    plt.plot(remaining_path[0, 1], remaining_path[1, 1], 'x', 
                            color=path_color, markersize=8, label='Next Point')
                plt.plot(remaining_path[0, -1], remaining_path[1, -1], 's',
                         color=path_color, markersize=10, label='Goal')
        
        # 6. 繪製當前位置
        plt.plot(self.robot_position[0], self.robot_position[1], 
                'o', color=path_color, markersize=8, label=f'Robot{self.robot_id} Current')
        
        # 7. 繪製起始位置
        if len(self.xPoint) > 0:
            plt.plot(self.xPoint[0], self.yPoint[0], 
                    'o', color='cyan', markersize=8, label='Start Position')
        
        # 8. 繪製所有其他機器人的位置
        for i, other_pos in enumerate(self.other_robots_positions):
            other_color = colors[(self.robot_id + i + 1) % len(colors)]
            plt.plot(other_pos[0], other_pos[1], 
                    'o', color=other_color, markersize=8, 
                    label=f'Robot{(self.robot_id + i + 1) % self.num_robots}')
        
        # 9. 添加圖例和標題
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
        explored_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        plt.title(f'Robot{self.robot_id} Exploration Progress: {explored_ratio:.1%}')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_training_progress(self):
        """繪製訓練進度圖"""
        fig, axs = plt.subplots(6, 1, figsize=(12, 20))
        
        episodes = range(1, len(self.training_history['episode_rewards']) + 1)
        
        # 繪製總獎勵
        axs[0].plot(episodes, self.training_history['episode_rewards'], 
                    color='#4B0082')
        axs[0].set_title('總獎勵')
        axs[0].set_xlabel('輪數')
        axs[0].set_ylabel('獎勵')
        axs[0].grid(True)
        
        # 繪製各機器人獎勵（支援任意數量）
        colors = ['#800080', '#FFA500', '#FF6347', '#4169E1', '#32CD32']
        for i in range(self.num_robots):
            if f'robot{i}_rewards' in self.training_history:
                color = colors[i % len(colors)]
                axs[1].plot(episodes, self.training_history[f'robot{i}_rewards'], 
                           color=color, label=f'Robot{i}', alpha=0.8)
        axs[1].set_title('各機器人獎勵')
        axs[1].set_xlabel('輪數')
        axs[1].set_ylabel('獎勵')
        axs[1].legend()
        axs[1].grid(True)
        
        # 其餘圖表保持不變
        axs[2].plot(episodes, self.training_history['episode_lengths'], color='#4169E1')
        axs[2].set_title('每輪步數')
        axs[2].set_xlabel('輪數')
        axs[2].set_ylabel('步數')
        axs[2].grid(True)
        
        axs[3].plot(episodes, self.training_history['exploration_rates'], color='#228B22')
        axs[3].set_title('探索率')
        axs[3].set_xlabel('輪數')
        axs[3].set_ylabel('Epsilon')
        axs[3].grid(True)
        
        axs[4].plot(episodes, self.training_history['losses'], color='#B22222')
        axs[4].set_title('訓練損失')
        axs[4].set_xlabel('輪數')
        axs[4].set_ylabel('損失值')
        axs[4].grid(True)
        
        axs[5].plot(episodes, self.training_history['exploration_progress'], color='#2F4F4F')
        axs[5].set_title('探索進度')
        axs[5].set_xlabel('輪數')
        axs[5].set_ylabel('探索完成率')
        axs[5].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 以下方法保持不變...
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
        
        if (cost_map[int(start[1]), int(start[0])] >= self.lethal_cost or
            cost_map[int(goal[1]), int(goal[0])] >= self.lethal_cost):
            return None
            
        start = tuple(start)
        goal = tuple(goal)
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return np.array(path).T
                
            for next_pos in self.get_neighbors(current, cost_map):
                if op_map[next_pos[1], next_pos[0]] != 255:
                    continue
                    
                movement_cost = 1.0
                if abs(next_pos[0] - current[0]) + abs(next_pos[1] - current[1]) == 2:
                    movement_cost = ROBOT_CONFIG['diagonal_weight']
                    
                inflation_cost = cost_map[next_pos[1], next_pos[0]] / self.lethal_cost
                new_cost = cost_so_far[current] + movement_cost * (1 + inflation_cost)
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal) * (1 + inflation_cost)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
                    
        return None

    def get_neighbors(self, pos, cost_map):
        """獲取當前位置的有效鄰居節點，只考慮已知區域"""
        x, y = pos
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

    def astar(self, op_map, start, goal):
        """A*路徑規劃，只走已知區域"""
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
                
                move_cost = ROBOT_CONFIG['diagonal_weight'] if dx != 0 and dy != 0 else 1
                tentative_g_score = gscore[current] + move_cost
                
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        return None

    def heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        D = 1
        D2 = ROBOT_CONFIG['diagonal_weight']
        return D * max(dx, dy) + (D2 - D) * min(dx, dy)

    def astar_path(self, op_map, start, goal, safety_distance=None):
        """获取考虑膨胀的A*路径"""
        if safety_distance is None:
            safety_distance = ROBOT_CONFIG['safety_distance']
            
        path = self.astar_with_inflation(start, goal, op_map)
        if path is None:
            return None
            
        return self.simplify_path(path, ROBOT_CONFIG['path_simplification'])

    def simplify_path(self, path, threshold):
        """路径简化"""
        if path.shape[1] <= 2:
            return path
            
        def point_line_distance(point, start, end):
            if np.all(start == end):
                return np.linalg.norm(point - start)
                
            line_vec = end - start
            point_vec = point - start
            line_len = np.linalg.norm(line_vec)
            line_unit_vec = line_vec / line_len
            projection_length = np.dot(point_vec, line_unit_vec)
            
            if projection_length < 0:
                return np.linalg.norm(point - start)
            elif projection_length > line_len:
                return np.linalg.norm(point - end)
            else:
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

    def check_completion(self):
        """检查探索是否完成"""
        exploration_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        
        if exploration_ratio > self.finish_percent:
            self.li_map += 1
            if self.li_map >= self.map_number:
                self.li_map = 0
                return True, True
                
            self.__init__(self.li_map, self.mode, self.plot)
            return True, False
            
        return False, False
    
    def reset(self):
        """重置環境到新地圖，並確保所有機器人使用相同的地圖"""
        if self.plot:
            self.cleanup_visualization()

        if self.is_primary:
            self.li_map += 1
            if self.li_map >= self.map_number:
                self.li_map = 0
                
            self.global_map, self.initial_positions = self.map_setup(
                os.path.join(self.map_dir, self.map_list[self.li_map])
            )
            
            self.robot_position = self.initial_positions[0].astype(np.int64)
            self.all_robots_initial_positions = [pos.astype(np.int64) for pos in self.initial_positions]
            
            self.op_map = np.ones(self.global_map.shape) * 127
            self.t = self.map_points(self.global_map)
            self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())
        else:
            self.li_map = self.shared_env.li_map
            self.global_map = self.shared_env.global_map
            self.op_map = self.shared_env.op_map
            self.map_size = self.shared_env.map_size
            
            self.robot_position = self.shared_env.all_robots_initial_positions[self.robot_id].copy()
            self.all_robots_initial_positions = self.shared_env.all_robots_initial_positions
            
            self.t = self.shared_env.t
            self.free_tree = self.shared_env.free_tree
        
        # 重置其他狀態
        self.old_position = np.zeros([2])
        self.old_op_map = np.empty([0])
        self.current_target_frontier = None
        self.is_moving_to_target = False
        self.current_path = None
        self.current_path_index = 0
        self.steps = 0
        
        if self.plot:
            self.initialize_visualization()
        
        return self.begin()

    def check_done(self):
        """检查是否需要结束当前回合"""
        exploration_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        if exploration_ratio > self.finish_percent:
            return True
            
        frontiers = self.get_frontiers()
        if len(frontiers) == 0:
            return True
            
        return False

    def get_observation(self):
        """获取当前观察状态"""
        step_map = self.robot_model(
            self.robot_position, self.robot_size, self.t, self.op_map)
            
        resized_map = resize(step_map, (84, 84))
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
            self.robot_position[0] / float(self.map_size[1]),
            self.robot_position[1] / float(self.map_size[0])
        ])
        
    def initialize_visualization(self):
        """初始化可視化相關的屬性"""
        self.xPoint = np.array([self.robot_position[0]])
        self.yPoint = np.array([self.robot_position[1]])
        self.x2frontier = np.empty([0])
        self.y2frontier = np.empty([0])
        
        self.fig = plt.figure(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title(f'Robot{self.robot_id} Exploration')

    def get_other_robots_positions(self):
        """獲取所有其他機器人的位置"""
        return [pos.copy() for pos in self.other_robots_positions]

    def update_other_robots_positions(self, positions_list):
        """更新所有其他機器人的位置"""
        self.other_robots_positions = [np.array(pos) for pos in positions_list]
        
    def cleanup_visualization(self):
        """清理可視化資源"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            plt.clf()
            self.fig = None