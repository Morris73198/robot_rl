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

class Robot:
    def __init__(self, index_map, train, plot):
        """初始化機器人環境"""
        self.mode = train
        self.plot = plot
        
        
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
        
        # 初始化地圖和機器人位置
        self.global_map, self.robot_position = self.map_setup(
            os.path.join(self.map_dir, self.map_list[self.li_map]))
        self.robot_position = self.robot_position.astype(np.int64)

        self.op_map = np.ones(self.global_map.shape) * 127
        self.map_size = np.shape(self.global_map)
        
        # 加载參數
        self.movement_step = ROBOT_CONFIG['movement_step']
        self.finish_percent = ROBOT_CONFIG['finish_percent']
        self.sensor_range = ROBOT_CONFIG['sensor_range']
        self.robot_size = ROBOT_CONFIG['robot_size']
        self.local_size = ROBOT_CONFIG['local_size']
        
        # 膨脹參數
        self.inflation_radius = self.robot_size * 10  # 膨胀半径为机器人尺寸的1.5倍
        self.lethal_cost = 100  # 致命障碍物代价
        self.decay_factor = 3  # 代价衰减因子
        
        
        self.old_position = np.zeros([2])
        self.old_op_map = np.empty([0])
        self.current_target_frontier = None
        self.is_moving_to_target = False
        self.steps = 0
        
        # map and free space
        self.t = self.map_points(self.global_map)
        self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())
        
        # vizsualize
        if self.plot:
            self.xPoint = np.array([self.robot_position[0]])
            self.yPoint = np.array([self.robot_position[1]])
            self.x2frontier = np.empty([0])
            self.y2frontier = np.empty([0])

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
        """改進的移動到frontier方法"""
        # 如果正在執行移動且目標點沒有改變,繼續使用當前目標
        if self.is_moving_to_target and np.array_equal(self.current_target_frontier, target_frontier):
            target_frontier = self.current_target_frontier
        else:
            # 新的移動任務
            self.current_target_frontier = target_frontier.copy()
            self.is_moving_to_target = True
        
        # 初始路徑規劃
        path = self.astar_path(
            self.op_map, 
            self.robot_position.astype(np.int32),
            target_frontier.astype(np.int32),
            safety_distance=ROBOT_CONFIG['safety_distance']
        )
        
        if path is None:
            self.is_moving_to_target = False
            return self.get_observation(), -1, True
        
        path = self.simplify_path(path, ROBOT_CONFIG['path_simplification'])
        
        total_reward = 0
        done = False
        next_state = None
        
        # 改進的路徑跟隨邏輯
        path_points = path.T
        current_path_index = 0
        look_ahead_distance = ROBOT_CONFIG['movement_step'] * 1.5  # 前瞻距離
        last_check_position = self.robot_position.copy()
        check_interval = ROBOT_CONFIG['movement_step'] * 2  # 檢查間隔
        
        while current_path_index < len(path_points):
            # 檢查是否需要重新規劃路徑
            current_distance = np.linalg.norm(self.robot_position - last_check_position)
            if current_distance >= check_interval:
                # 檢查當前路徑是否被阻擋
                current_to_goal = path_points[current_path_index:] 
                path_blocked = self.check_path_blocked(current_to_goal)
                
                if path_blocked:
                    # 重新規劃路徑
                    new_path = self.astar_path(
                        self.op_map,
                        self.robot_position.astype(np.int32),
                        target_frontier.astype(np.int32),
                        safety_distance=ROBOT_CONFIG['safety_distance']
                    )
                    
                    if new_path is None:
                        # 無法找到新路徑
                        self.is_moving_to_target = False
                        return self.get_observation(), -1, True
                    
                    # 更新路徑
                    path = self.simplify_path(new_path, ROBOT_CONFIG['path_simplification'])
                    path_points = path.T
                    current_path_index = 0
                
                last_check_position = self.robot_position.copy()
            
            # 找到前瞻點
            look_ahead_point = None
            accumulated_distance = 0
            
            for i in range(current_path_index, len(path_points)):
                if i == current_path_index:
                    continue
                point_distance = np.linalg.norm(
                    path_points[i] - path_points[i-1]
                )
                accumulated_distance += point_distance
                
                if accumulated_distance >= look_ahead_distance:
                    look_ahead_point = path_points[i]
                    break
            
            if look_ahead_point is None:
                look_ahead_point = path_points[-1]
            
            # 計算移動向量
            move_vector = look_ahead_point - self.robot_position
            dist = np.linalg.norm(move_vector)
            
            # 根據距離動態調整步長
            if dist > ROBOT_CONFIG['movement_step']:
                # 保持固定步長
                move_vector = move_vector * (ROBOT_CONFIG['movement_step'] / dist)
            else:
                # 接近目標時使用較小步長
                move_vector = move_vector * 0.5
            
            # 執行移動
            next_state, reward, step_done = self.execute_movement(move_vector)
            total_reward += reward
            
            if step_done:
                done = True
                break
            
            # 更新當前路徑索引
            current_position = self.robot_position
            min_distance = float('inf')
            new_index = current_path_index
            
            # 找到最近的路徑點作為新的索引
            for i in range(current_path_index, len(path_points)):
                distance = np.linalg.norm(path_points[i] - current_position)
                if distance < min_distance:
                    min_distance = distance
                    new_index = i
            
            current_path_index = new_index
            
            # 檢查是否到達目標點附近
            dist_to_target = np.linalg.norm(self.robot_position - target_frontier)
            if dist_to_target < ROBOT_CONFIG['target_reach_threshold']:
                done = True
                break
        
        # 如果完成移動或遇到障礙,重置移動狀態
        if done:
            self.is_moving_to_target = False
            self.current_target_frontier = None
            
        return next_state, total_reward, done

    def check_path_blocked(self, path_points):
        """檢查路徑是否被阻擋
        
        Args:
            path_points: nx2的數組，包含路徑點
            
        Returns:
            bool: True表示路徑被阻擋，False表示路徑暢通
        """
        if len(path_points) < 2:
            return False
            
        for i in range(len(path_points) - 1):
            start = path_points[i]
            end = path_points[i + 1]
            
            # 檢查這段路徑是否碰到障礙物
            collision_points, collision_index = self.fast_collision_check(
                start, 
                end, 
                self.map_size,
                self.op_map  # 使用當前觀測到的地圖
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
            self.op_map = self.inverse_sensor(
                self.robot_position, self.sensor_range, 
                self.op_map, self.global_map
            )
            reward = self.calculate_fast_reward(old_op_map, self.op_map, move_vector)
            done = False
        
        self.steps += 1
        if self.plot and self.steps % ROBOT_CONFIG['plot_interval'] == 0:
            self.xPoint = np.append(self.xPoint, self.robot_position[0])
            self.yPoint = np.append(self.yPoint, self.robot_position[1])
            self.plot_env()
        
        return self.get_observation(), reward, done

    def calculate_fast_reward(self, old_op_map, new_op_map, move_vector):
        """計算獎勵"""
        # 探索獎勵
        explored_diff = float(
            np.sum(new_op_map == 255) - np.sum(old_op_map == 255)
        ) / 14000 * REWARD_CONFIG['exploration_weight']
        
        # 移動懲罰
        movement_cost = REWARD_CONFIG['movement_penalty'] * np.linalg.norm(move_vector)
        
        # 離目標點越近獎勵越高（之後可能會拿掉）
        if self.current_target_frontier is not None:
            distance_to_target = np.linalg.norm(
                self.current_target_frontier - self.robot_position)
            progress_reward = -0.0001 * distance_to_target
        else:
            progress_reward = 0
        
        total_reward = explored_diff + movement_cost + progress_reward
        return np.clip(total_reward, -1, 1)

    def map_setup(self, location):
        """設置地圖和機器人初始位置"""
        global_map = (io.imread(location, 1) * 255).astype(int)
        robot_location = np.nonzero(global_map == 208)
        robot_location = np.array([np.array(robot_location)[1, 0], 
                                 np.array(robot_location)[0, 0]])
        global_map = (global_map > 150)
        global_map = global_map * 254 + 1
        return global_map, robot_location

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
        # robot_points = self.range_search(position, robot_size, points)
        # for point in robot_points:
        #     y, x = point[::-1].astype(int) #(x,y)轉（y,x）
        #     if 0 <= y < map_copy.shape[0] and 0 <= x < map_copy.shape[1]:
        #         map_copy[y, x] = 76 # 機器人位置標記為 76
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
        if self.is_moving_to_target and self.current_target_frontier is not None:
            return np.array([self.current_target_frontier])
            
        frontiers = self.frontier(self.op_map, self.map_size, self.t)
        if len(frontiers) == 0:
            return np.zeros((0, 2))
            
        distances = np.linalg.norm(frontiers - self.robot_position, axis=1)
        sorted_indices = np.argsort(distances)
        return frontiers[sorted_indices]

    def plot_env(self):
        plt.cla()
        plt.imshow(self.op_map, cmap='gray')
        plt.axis((0, self.map_size[1], self.map_size[0], 0))
        
        plt.plot(self.xPoint, self.yPoint, 'b-', linewidth=2, label='Robot Path')
        
        frontiers = self.get_frontiers()
        if len(frontiers) > 0:
            plt.scatter(frontiers[:, 0], frontiers[:, 1], 
                    c='red', marker='*', s=100, label='Frontiers')
        
        if self.current_target_frontier is not None:
            plt.plot(self.current_target_frontier[0], self.current_target_frontier[1], 
                    'go', markersize=10, label='Target Frontier')
            
            path = self.astar_path(
                self.op_map,
                self.robot_position.astype(np.int32),
                self.current_target_frontier.astype(np.int32)
            )
            
            if path is not None and path.shape[1] > 1:
                path_x = path[0, :]
                path_y = path[1, :]
                plt.plot(path_x, path_y, 'g--', linewidth=2, 
                        alpha=0.8, label='Planned Path')
                
                if len(path_x) > 1:
                    direction_x = path_x[1] - self.robot_position[0]
                    direction_y = path_y[1] - self.robot_position[1]
                    
                    magnitude = np.sqrt(direction_x**2 + direction_y**2)
                    if magnitude > 0:
                        direction_x = direction_x / magnitude * 20  
                        direction_y = direction_y / magnitude * 20
                        
                        plt.arrow(self.robot_position[0], self.robot_position[1],
                                direction_x, direction_y,
                                head_width=3, head_length=3, fc='yellow', ec='yellow',
                                label='Movement Direction', zorder=5)
        
        plt.plot(self.robot_position[0], self.robot_position[1], 
                'mo', markersize=8, label='Current Position')
        plt.plot(self.xPoint[0], self.yPoint[0], 
                'co', markersize=8, label='Start Position')
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        explored_ratio = np.sum(self.op_map == 255) / np.sum(self.global_map == 255)
        plt.title(f'Exploration Progress: {explored_ratio:.1%}')
        
        plt.pause(0.01)














    def inflate_map(self, binary_map):
        """
        膨胀地图以创建代价地图
        
        Args:
            binary_map: 二值地图 (0: 自由空间, 1: 障碍物)
            
        Returns:
            cost_map: 带有膨胀障碍物的代价地图
        """
        # 创建障碍物地图
        obstacle_map = (binary_map == 1)
        
        # 计算距离变换
        distances = distance_transform_edt(~obstacle_map)
        
        # 创建代价地图
        cost_map = np.zeros_like(distances)
        
        # 设置致命障碍物
        cost_map[obstacle_map] = self.lethal_cost
        
        # 计算膨胀代价
        inflation_mask = (distances > 0) & (distances <= self.inflation_radius)
        cost_map[inflation_mask] = self.lethal_cost * np.exp(
            -self.decay_factor * distances[inflation_mask] / self.inflation_radius
        )
        
        return cost_map

    def astar_with_inflation(self, start, goal, op_map):
        """
        考虑膨胀的A*路径规划
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            op_map: 观测地图
            
        Returns:
            path: 路径点列表，如果没找到则返回None
        """
        # 创建二值地图
        binary_map = (op_map == 1).astype(int)
        
        # 获取膨胀后的代价地图
        cost_map = self.inflate_map(binary_map)
        
        # 检查起点和终点是否在安全区域
        if (cost_map[int(start[1]), int(start[0])] >= self.lethal_cost or
            cost_map[int(goal[1]), int(goal[0])] >= self.lethal_cost):
            return None
            
        # 初始化开放和关闭列表
        start = tuple(start)
        goal = tuple(goal)
        frontier = []
        heappush(frontier, (0, start))  # 使用导入的heappush而不是heapq.heappush
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heappop(frontier)[1]  # 使用导入的heappop而不是heapq.heappop
            
            if current == goal:
                # 重建路径
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return np.array(path).T
                
            for next_pos in self.get_neighbors(current, cost_map):
                # 计算新代价，包括膨胀代价
                movement_cost = 1.0
                if abs(next_pos[0] - current[0]) + abs(next_pos[1] - current[1]) == 2:
                    movement_cost = ROBOT_CONFIG['diagonal_weight']
                    
                inflation_cost = cost_map[next_pos[1], next_pos[0]] / self.lethal_cost
                new_cost = cost_so_far[current] + movement_cost * (1 + inflation_cost)
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal) * (1 + inflation_cost)
                    heappush(frontier, (priority, next_pos))  # 使用导入的heappush
                    came_from[next_pos] = current
                    
        return None





    def get_neighbors(self, pos, cost_map):
        """
        获取当前位置的有效邻居节点
        
        Args:
            pos: 当前位置 (x, y)
            cost_map: 包含障碍物和膨胀区域的代价地图
            
        Returns:
            neighbors: 有效的邻居位置列表
        """
        x, y = pos
        neighbors = []
        
        # 8个方向的邻居：上下左右和对角线
        directions = [
            (0, 1),   # 右
            (1, 0),   # 下
            (0, -1),  # 左
            (-1, 0),  # 上
            (1, 1),   # 右下
            (-1, 1),  # 右上
            (1, -1),  # 左下
            (-1, -1)  # 左上
        ]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            # 检查边界
            if not (0 <= new_x < self.map_size[1] and 0 <= new_y < self.map_size[0]):
                continue
                
            # 检查是否在安全区域（代价小于致命代价）
            if cost_map[new_y, new_x] < self.lethal_cost:
                neighbors.append((new_x, new_y))
                
        return neighbors

















    def astar(self, op_map, start, goal):
        """A*路径规划"""
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
                
                # 边界检查
                if not (0 <= neighbor[0] < self.map_size[1] and 
                       0 <= neighbor[1] < self.map_size[0]):
                    continue
                    
                # 障碍物检查
                if op_map[neighbor[1]][neighbor[0]] == 1:
                    continue
                
                # 移动代价计算，对角线移动代价更高
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
        """启发式函数：使用对角线距离"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        D = 1  # 直线移动代价
        D2 = ROBOT_CONFIG['diagonal_weight']  # 对角线移动代价
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
                return True, True  # 完成所有地图
                
            self.__init__(self.li_map, self.mode, self.plot)
            return True, False  # 完成当前地图
            
        return False, False
    
    
    def reset(self):
        """重置环境到新地图"""
        self.li_map += 1
        if self.li_map >= self.map_number:
            self.li_map = 0
            
        # 初始化新地图
        self.__init__(self.li_map, self.mode, self.plot)
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