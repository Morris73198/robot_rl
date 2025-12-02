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
import networkx as nx
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology, segmentation
from two_robot_cnndqn_attention.environment.multi_robot_no_unknown import Robot
from two_robot_cnndqn_attention.config import MODEL_CONFIG, MODEL_DIR, ROBOT_CONFIG
from two_robot_cnndqn_attention.environment.robot_local_map_tracker import RobotIndividualMapTracker

class SegmentationExplorer:
    """基於環境分割的多機器人探索協調器"""
    
    def __init__(self, robot1, robot2, output_dir='results_segmentation'):
        """初始化分割探索協調器
        
        Args:
            robot1: 第一個機器人實例
            robot2: 第二個機器人實例
            output_dir: 輸出目錄
        """
        self.robot1 = robot1
        self.robot2 = robot2
        self.output_dir = output_dir
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 分割圖相關屬性
        self.segments = None
        self.segment_labels = None
        self.segment_frontiers = {}  # 每個分段的frontier點
        self.robot_segment_assignments = {
            'robot1': None,
            'robot2': None
        }
        
        # 可視化相關
        self.fig = None
        self.axes = None
    
    def segment_environment(self):
        """對環境進行分割"""
        print("開始對環境進行分割...")
        
        # 獲取當前探索地圖
        op_map = self.robot1.op_map.copy()
        
        # 創建二值地圖 (255: 自由空間, 其他: 障礙物或未知)
        binary_map = (op_map == 255).astype(np.uint8)
        
        # 使用距離變換生成Voronoi圖
        dist_transform = distance_transform_edt(binary_map)
        
        # 標識狹窄通道和門口 (局部最小值)
        # 論文中提到這些是分割的關鍵點
        local_minima = self._find_critical_points(dist_transform, binary_map)
        
        # 使用局部最小值作為種子進行分水嶺分割
        markers = np.zeros_like(binary_map, dtype=np.int32)
        
        # 標記種子 (每個房間一個標記)
        rooms = measure.label(binary_map, connectivity=2)
        max_room_id = np.max(rooms)
        
        # 如果只有一個連通區域，使用K-means或其他方法進行分割
        if max_room_id <= 1:
            # 嘗試使用形狀特徵進行分割
            self.segments, self.segment_labels = self._segment_large_area(binary_map)
        else:
            # 以房間為基礎進行分割
            self.segments = rooms
            self.segment_labels = np.unique(rooms)[1:]  # 忽略背景 (0)
        
        print(f"分割完成，共分為 {len(self.segment_labels)} 個區域")
        
        # 保存分割圖
        plt.figure(figsize=(12, 10))
        plt.imshow(self.segments, cmap='nipy_spectral')
        plt.colorbar(label='Segment ID')
        plt.title('Environment Segmentation')
        plt.savefig(os.path.join(self.output_dir, 'segmentation.png'), dpi=300)
        plt.close()
        
        return self.segments, self.segment_labels
    
    def _find_critical_points(self, dist_transform, binary_map):
        """查找臨界點 (通常是門口或狹窄通道)
        
        Args:
            dist_transform: 距離變換圖
            binary_map: 二值地圖 (True: 自由空間, False: 障礙物或未知)
            
        Returns:
            critical_points: 臨界點列表 [(x1,y1), (x2,y2), ...]
        """
        # 論文中的方法：尋找局部最小值（到最近障礙物距離的局部最小點）
        # 對距離圖應用高斯平滑以減少噪聲
        from scipy.ndimage import gaussian_filter
        smoothed_dist = gaussian_filter(dist_transform, sigma=1.0)
        
        # 構建圖形以查找度為2的節點（兩個相連節點）和附近的度為3的結點（交叉點）
        # 這是論文中提到的臨界點條件
        G = nx.Graph()
        
        # 在距離圖上查找局部極小值
        min_pool_size = 3
        from scipy.ndimage import minimum_filter
        local_min = minimum_filter(smoothed_dist, size=min_pool_size)
        min_mask = (smoothed_dist == local_min) & (smoothed_dist > 0)
        
        # 只考慮距離足夠小的點（代表狹窄區域）
        narrow_threshold = np.mean(smoothed_dist[binary_map]) * 0.6
        narrow_areas = min_mask & (smoothed_dist < narrow_threshold)
        
        # 找到這些點的坐標
        critical_points = list(zip(*np.where(narrow_areas)))
        
        # 打印找到的臨界點數量
        print(f"找到 {len(critical_points)} 個臨界點")
        
        return critical_points
    
    def _segment_large_area(self, binary_map):
        """對大區域進行分割，使用骨架特徵
        
        Args:
            binary_map: 二值地圖
            
        Returns:
            segments: 分割後的標籤圖
            segment_labels: 分割標籤列表
        """
        # 計算骨架
        skeleton = morphology.skeletonize(binary_map)
        
        # 使用骨架分叉點作為種子
        from scipy import ndimage
        # 計算每個點的鄰居數量
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        neighbors = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        # 分叉點有3個或更多鄰居
        branch_points = (neighbors >= 3) & skeleton
        
        # 如果找不到足夠的分叉點，使用K-means進行分割
        if np.sum(branch_points) < 2:
            print("使用K-means進行分割...")
            # 獲取自由空間點的坐標
            y, x = np.where(binary_map)
            points = np.column_stack([x, y])
            
            # 根據地圖大小估計合適的分割數
            map_size = binary_map.shape
            # 計算地圖的對角線長度，用於估計區域數量
            diag_length = np.sqrt(map_size[0]**2 + map_size[1]**2)
            k = max(2, min(5, int(diag_length / 200)))  # 至少2個區域，最多5個
            
            # 使用K-means進行分割
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=0).fit(points)
            
            # 創建分割標籤圖
            segments = np.zeros_like(binary_map, dtype=int)
            segments[y, x] = kmeans.labels_ + 1  # 從1開始標記
            
            segment_labels = np.arange(1, k+1)
        else:
            # 使用分叉點和端點作為標記
            end_points = (neighbors == 1) & skeleton
            markers = np.zeros_like(binary_map, dtype=int)
            
            # 為每個分叉點分配一個唯一標籤
            branch_labels = measure.label(branch_points, connectivity=2)
            max_branch = np.max(branch_labels)
            
            # 為每個端點分配一個唯一標籤
            end_labels = measure.label(end_points, connectivity=2)
            max_end = np.max(end_labels)
            
            # 組合標記
            markers[branch_points] = branch_labels[branch_points]
            markers[end_points] = max_branch + end_labels[end_points]
            
            # 使用分水嶺算法進行分割
            segments = segmentation.watershed(-distance_transform_edt(binary_map), markers, mask=binary_map)
            segment_labels = np.unique(segments)[1:]  # 忽略背景 (0)
            
            # 如果分割過多，合併小區域
            if len(segment_labels) > 5:
                # 計算每個區域的大小
                segment_sizes = {label: np.sum(segments == label) for label in segment_labels}
                # 計算平均大小的30%作為閾值
                size_threshold = 0.3 * (np.sum(binary_map) / len(segment_labels))
                
                # 合併小區域到相鄰的大區域
                for label in segment_labels:
                    if segment_sizes[label] < size_threshold:
                        # 找到與該小區域相鄰的最大區域
                        dilated = morphology.binary_dilation(segments == label)
                        neighbors = segments[dilated & (segments != label)]
                        if len(neighbors) > 0:
                            neighbor_labels = np.unique(neighbors)
                            if len(neighbor_labels) > 0:
                                # 找到最大的相鄰區域
                                largest_neighbor = max(neighbor_labels, key=lambda x: segment_sizes.get(x, 0))
                                # 合併
                                segments[segments == label] = largest_neighbor
                
                # 更新標籤
                segment_labels = np.unique(segments)[1:]
        
        print(f"大區域分割完成，共分為 {len(segment_labels)} 個區域")
        return segments, segment_labels
    
    def identify_segment_frontiers(self):
        """為每個分段識別frontier點"""
        # 確保已經進行分割
        if self.segments is None:
            self.segment_environment()
        
        # 獲取所有frontier點
        all_frontiers = self.robot1.get_frontiers()
        
        # 初始化每個分段的frontier字典
        self.segment_frontiers = {label: [] for label in self.segment_labels}
        
        # 為每個frontier點分配分段
        for frontier in all_frontiers:
            x, y = int(frontier[0]), int(frontier[1])
            # 檢查座標是否在地圖範圍內
            if 0 <= x < self.segments.shape[1] and 0 <= y < self.segments.shape[0]:
                segment_id = self.segments[y, x]
                # 只考慮有效分段
                if segment_id in self.segment_labels:
                    self.segment_frontiers[segment_id].append(frontier)
        
        # 打印每個分段的frontier數量
        for segment_id in self.segment_labels:
            frontiers = self.segment_frontiers.get(segment_id, [])
            print(f"分段 {segment_id} 包含 {len(frontiers)} 個frontier點")
        
        return self.segment_frontiers
    
    def assign_robots_to_segments(self):
        """使用匈牙利算法將機器人分配到不同的分段"""
        print("開始分配機器人到分段...")
        
        # 確保已經識別了分段的frontier點
        if not self.segment_frontiers:
            self.identify_segment_frontiers()
        
        # 篩選掉沒有frontier的分段
        valid_segments = [segment_id for segment_id in self.segment_labels 
                          if len(self.segment_frontiers.get(segment_id, [])) > 0]
        
        # 如果沒有有效分段，返回
        if not valid_segments:
            print("沒有包含frontier的有效分段")
            return False
        
        # 獲取機器人位置
        robot1_pos = self.robot1.robot_position
        robot2_pos = self.robot2.robot_position
        
        # 計算每個機器人到每個分段的代價
        cost_matrix = np.zeros((2, len(valid_segments)))
        
        for i, segment_id in enumerate(valid_segments):
            # 使用該分段中所有frontier點到機器人的最小距離作為代價
            frontiers = self.segment_frontiers.get(segment_id, [])
            
            if frontiers:
                # 計算到robot1的最小距離
                dist_to_robot1 = min(np.linalg.norm(np.array(frontier) - robot1_pos) 
                                    for frontier in frontiers)
                # 計算到robot2的最小距離
                dist_to_robot2 = min(np.linalg.norm(np.array(frontier) - robot2_pos) 
                                    for frontier in frontiers)
                
                # 檢查機器人是否已經在該分段中
                segment_map = (self.segments == segment_id)
                robot1_in_segment = segment_map[int(robot1_pos[1]), int(robot1_pos[0])]
                robot2_in_segment = segment_map[int(robot2_pos[1]), int(robot2_pos[0])]
                
                # 如果機器人已經在該分段中，給予折扣（論文中的方法）
                discount_factor = 0.7  # 折扣因子
                if robot1_in_segment:
                    dist_to_robot1 *= discount_factor
                if robot2_in_segment:
                    dist_to_robot2 *= discount_factor
                
                cost_matrix[0, i] = dist_to_robot1
                cost_matrix[1, i] = dist_to_robot2
            else:
                # 如果分段沒有frontier，設置高代價
                cost_matrix[0, i] = 1e6
                cost_matrix[1, i] = 1e6
        
        # 使用匈牙利算法進行分配
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 根據分配結果更新機器人分段分配
        self.robot_segment_assignments = {
            'robot1': valid_segments[col_ind[0]] if len(col_ind) > 0 else None,
            'robot2': valid_segments[col_ind[1]] if len(col_ind) > 1 else None
        }
        
        print(f"機器人1被分配到分段 {self.robot_segment_assignments['robot1']}")
        print(f"機器人2被分配到分段 {self.robot_segment_assignments['robot2']}")
        
        return True
    
    def get_targets_for_robots(self):
        """根據分段分配獲取機器人的目標frontier點"""
        # 確保已經進行了機器人到分段的分配
        if self.robot_segment_assignments['robot1'] is None and self.robot_segment_assignments['robot2'] is None:
            # 如果分配失敗，使用常規frontier方法
            frontiers = self.robot1.get_frontiers()
            if len(frontiers) >= 2:
                return frontiers[0], frontiers[1]
            elif len(frontiers) == 1:
                return frontiers[0], frontiers[0]
            else:
                return None, None
        
        # 獲取每個機器人的分段frontier點
        robot1_segment = self.robot_segment_assignments['robot1']
        robot2_segment = self.robot_segment_assignments['robot2']
        
        robot1_frontiers = self.segment_frontiers.get(robot1_segment, []) if robot1_segment else []
        robot2_frontiers = self.segment_frontiers.get(robot2_segment, []) if robot2_segment else []
        
        # 選擇每個機器人在其分段中的最佳frontier點
        robot1_target = None
        if robot1_frontiers:
            # 選擇距離機器人1最近的frontier
            distances = [np.linalg.norm(np.array(f) - self.robot1.robot_position) for f in robot1_frontiers]
            robot1_target = robot1_frontiers[np.argmin(distances)]
        
        robot2_target = None
        if robot2_frontiers:
            # 選擇距離機器人2最近的frontier
            distances = [np.linalg.norm(np.array(f) - self.robot2.robot_position) for f in robot2_frontiers]
            robot2_target = robot2_frontiers[np.argmin(distances)]
        
        # 如果沒有為某個機器人找到目標，使用另一個機器人的分段
        if robot1_target is None and robot2_target is not None:
            robot1_target = robot2_target
        elif robot2_target is None and robot1_target is not None:
            robot2_target = robot1_target
        elif robot1_target is None and robot2_target is None:
            # 如果兩個機器人都沒有目標，使用常規frontier方法
            frontiers = self.robot1.get_frontiers()
            if len(frontiers) >= 2:
                robot1_target = frontiers[0]
                robot2_target = frontiers[1]
            elif len(frontiers) == 1:
                robot1_target = frontiers[0]
                robot2_target = frontiers[0]
        
        return robot1_target, robot2_target
    
    def visualize_segmentation(self, step, save=True):
        """可視化分割和機器人分配"""
        if self.segments is None:
            return
        
        plt.figure(figsize=(15, 10))
        
        # 顯示已探索地圖
        plt.subplot(121)
        plt.imshow(self.robot1.op_map, cmap='gray')
        plt.title('Explored Map')
        
        # 顯示帶有分段標籤和機器人位置的分割圖
        plt.subplot(122)
        
        # 創建彩色分割圖
        from matplotlib import colors
        # 創建隨機顏色
        np.random.seed(42)  # 固定隨機數種子以獲得一致的顏色
        cmap = colors.ListedColormap(np.random.rand(256, 3))
        bounds = np.linspace(0, 256, 257)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        plt.imshow(self.segments, cmap=cmap, norm=norm)
        plt.title('Environment Segmentation')
        
        # 顯示機器人位置
        robot1_pos = self.robot1.robot_position
        robot2_pos = self.robot2.robot_position
        plt.plot(robot1_pos[0], robot1_pos[1], 'bo', markersize=10, label='Robot 1')
        plt.plot(robot2_pos[0], robot2_pos[1], 'ro', markersize=10, label='Robot 2')
        
        # 顯示分配給機器人的目標點
        robot1_target, robot2_target = self.get_targets_for_robots()
        if robot1_target is not None:
            plt.plot(robot1_target[0], robot1_target[1], 'bx', markersize=8, label='Robot 1 Target')
        if robot2_target is not None:
            plt.plot(robot2_target[0], robot2_target[1], 'rx', markersize=8, label='Robot 2 Target')
        
        # 高亮顯示分配給每個機器人的分段
        if self.robot_segment_assignments['robot1'] is not None:
            segment1 = self.robot_segment_assignments['robot1']
            segment1_mask = (self.segments == segment1)
            # 使用半透明的藍色遮罩
            plt.contour(segment1_mask, colors='blue', linewidths=3, alpha=0.7)
        
        if self.robot_segment_assignments['robot2'] is not None:
            segment2 = self.robot_segment_assignments['robot2']
            segment2_mask = (self.segments == segment2)
            # 使用半透明的紅色遮罩
            plt.contour(segment2_mask, colors='red', linewidths=3, alpha=0.7)
        
        plt.legend()
        
        # 保存或顯示
        if save:
            plt.savefig(os.path.join(self.output_dir, f'segmentation_step_{step:04d}.png'), dpi=150)
            plt.close()
        else:
            plt.show()
    
    def update_segmentation(self):
        """根據最新的地圖更新分割"""
        # 每隔一定步數更新分割
        self.segment_environment()
        self.identify_segment_frontiers()
        self.assign_robots_to_segments()

def save_plot(robot, step, output_path):
    """儲存單個機器人的繪圖"""
    plt.figure(figsize=(10, 10))
    robot.plot_env()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')  # 關閉所有圖形以釋放記憶體

def create_robots_with_custom_positions(map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
    """創建使用特定地圖檔案和自定義起始位置的機器人"""
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

def test_segmentation_exploration(map_file_path, start_points_list, output_dir='results_segmentation', 
                              max_steps=100000, segment_update_interval=40):
    """使用基於分割的探索策略測試多個起始點位置
    
    Args:
        map_file_path: 要使用的特定地圖檔案的路徑
        start_points_list: 包含多個起始點的列表，每個元素是 [robot1_pos, robot2_pos]
        output_dir: 輸出目錄
        max_steps: 每個測試的最大步數
        segment_update_interval: 更新分割的步數間隔
    """
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 創建CSV檔案來記錄每個步驟的覆蓋率數據
    csv_path = os.path.join(output_dir, 'coverage_data.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['StartPoint', 'Step', 'Robot1Coverage', 'Robot2Coverage', 
                     'IntersectionCoverage', 'UnionCoverage', 'Strategy']
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
        
        # 創建分割可視化目錄
        segmentation_dir = os.path.join(current_output_dir, 'segmentation')
        if not os.path.exists(segmentation_dir):
            os.makedirs(segmentation_dir)
        
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
        
        # 創建分割探索協調器
        segmentation_explorer = SegmentationExplorer(
            robot1, 
            robot2, 
            output_dir=segmentation_dir
        )
        
        try:
            # 初始化環境
            state = robot1.begin()
            robot2.begin()
            
            # 開始追蹤個人地圖
            tracker.start_tracking()
            
            # 儲存初始狀態
            save_plot(robot1, 0, os.path.join(current_output_dir, 'robot1_step_0000.png'))
            save_plot(robot2, 0, os.path.join(current_output_dir, 'robot2_step_0000.png'))
            
            # 更新並儲存初始個人地圖
            tracker.update()
            tracker.save_current_maps(0)
            
            # 進行初始分割
            segmentation_explorer.segment_environment()
            segmentation_explorer.identify_segment_frontiers()
            segmentation_explorer.assign_robots_to_segments()
            segmentation_explorer.visualize_segmentation(0)
            
            steps = 0
            exploration_data = []
            
            # 主探索循環
            while steps < max_steps:
                # 檢查是否需要更新分割
                if steps % segment_update_interval == 0 and steps > 0:
                    print(f"步驟 {steps}: 更新環境分割")
                    segmentation_explorer.update_segmentation()
                    segmentation_explorer.visualize_segmentation(steps)
                
                # 檢查是否完成探索
                if robot1.check_done() or robot2.check_done():
                    print(f"步驟 {steps}: 探索完成")
                    break
                
                # 從分割協調器獲取目標
                robot1_target, robot2_target = segmentation_explorer.get_targets_for_robots()
                
                if robot1_target is None and robot2_target is None:
                    print(f"步驟 {steps}: 沒有可用的frontier點，探索結束")
                    break
                
                # 打印機器人選擇的目標和距離
                print(f"步驟 {steps}:")
                if robot1_target is not None:
                    robot1_segment = segmentation_explorer.robot_segment_assignments['robot1']
                    print(f"  機器人1 (分段 {robot1_segment}) 目標: {robot1_target}, " 
                         f"距離: {np.linalg.norm(robot1_target - robot1.robot_position):.2f}")
                
                if robot2_target is not None:
                    robot2_segment = segmentation_explorer.robot_segment_assignments['robot2']
                    print(f"  機器人2 (分段 {robot2_segment}) 目標: {robot2_target}, "
                         f"距離: {np.linalg.norm(robot2_target - robot2.robot_position):.2f}")
                
                # 移動機器人
                if robot1_target is not None:
                    next_state1, r1, d1 = robot1.move_to_frontier(robot1_target)
                else:
                    d1 = False
                    
                # 同步地圖
                robot2.op_map = robot1.op_map.copy()
                
                if robot2_target is not None:
                    next_state2, r2, d2 = robot2.move_to_frontier(robot2_target)
                else:
                    d2 = False
                
                # 再次同步地圖
                robot1.op_map = robot2.op_map.copy()
                
                # 更新位置
                robot1.other_robot_position = robot2.robot_position.copy()
                robot2.other_robot_position = robot1.robot_position.copy()
                
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
                    
                    # 記錄數據
                    exploration_data.append({
                        'step': steps,
                        'robot1_coverage': robot1_coverage,
                        'robot2_coverage': robot2_coverage,
                        'intersection_coverage': intersection_coverage,
                        'union_coverage': union_coverage
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
                            'Strategy': 'Segmentation'
                        })
                
                steps += 1
                
                # 每10步儲存繪圖和個人地圖
                if steps % 10 == 0:
                    save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_step_{steps:04d}.png'))
                    save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_step_{steps:04d}.png'))
                    
                    # 儲存當前個人地圖
                    tracker.save_current_maps(steps)
                    
                    # 輸出進度信息
                    coverage_data = exploration_data[-1]
                    print(f"步數: {steps}, 聯合覆蓋率: {coverage_data['union_coverage']:.1%}")
            
            # 儲存最終狀態
            save_plot(robot1, steps, os.path.join(current_output_dir, f'robot1_final_step_{steps:04d}.png'))
            save_plot(robot2, steps, os.path.join(current_output_dir, f'robot2_final_step_{steps:04d}.png'))
            
            # 儲存最終個人地圖
            tracker.save_current_maps(steps)
            
            # 儲存最終分割可視化
            segmentation_explorer.visualize_segmentation(steps)
            
            # 生成每個起始點的覆蓋率時間變化圖表
            plt.figure(figsize=(12, 8))
            steps_data = [data['step'] for data in exploration_data]
            robot1_coverage = [data['robot1_coverage'] for data in exploration_data]
            robot2_coverage = [data['robot2_coverage'] for data in exploration_data]
            intersection_coverage = [data['intersection_coverage'] for data in exploration_data]
            union_coverage = [data['union_coverage'] for data in exploration_data]
            
            # 繪製四條曲線
            plt.plot(steps_data, robot1_coverage, 'b-', linewidth=2, label='Robot 1')
            plt.plot(steps_data, robot2_coverage, 'r-', linewidth=2, label='Robot 2')
            plt.plot(steps_data, intersection_coverage, 'g-', linewidth=2, label='Intersection')
            plt.plot(steps_data, union_coverage, 'k-', linewidth=2, label='Union')
            
            # 添加標籤和標題
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Coverage', fontsize=14)
            plt.title(f'Time-Coverage Analysis - Segmentation Strategy (Start Point {start_idx+1})', fontsize=16)
            
            # 添加網格和圖例
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            # 設置y軸範圍
            plt.ylim(0, 1.05)
            
            plt.savefig(os.path.join(current_output_dir, 'time_coverage_analysis.png'), dpi=300)
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
                
                all_data[start_point]['steps'].append(step)
                all_data[start_point]['robot1'].append(float(row['Robot1Coverage']))
                all_data[start_point]['robot2'].append(float(row['Robot2Coverage']))
                all_data[start_point]['intersection'].append(float(row['IntersectionCoverage']))
                all_data[start_point]['union'].append(float(row['UnionCoverage']))
        
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
            plt.title(f'Time-Coverage Analysis - Segmentation Strategy (Start Point {start_point})', fontsize=16)
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
        plt.title('Total Coverage Comparison - Segmentation Strategy', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.ylim(0, 1.05)
        
        plt.savefig(os.path.join(output_dir, 'all_total_coverage_comparison.png'), dpi=300)
        plt.close()
        
        print(f"已生成所有起始點的覆蓋率分析圖表")
        
    except Exception as e:
        print(f"生成比較圖表時出錯: {str(e)}")
    
    print(f"\n===== 完成所有起始點的測試 =====")
    print(f"結果儲存在: {output_dir}")
    print(f"覆蓋率數據儲存在: {csv_path}")
    print(f"測試完成！共測試了 {len(start_points_list)} 組不同起始點")

def test_and_compare_strategies(map_file_path, start_points_list, base_output_dir='comparison_results'):
    """測試和比較不同的探索策略
    
    Args:
        map_file_path: 要使用的特定地圖檔案的路徑
        start_points_list: 包含多個起始點的列表，每個元素是 [robot1_pos, robot2_pos]
        base_output_dir: 基礎輸出目錄
    """
    # 確保輸出目錄存在
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # 測試基於分割的探索策略
    segmentation_output_dir = os.path.join(base_output_dir, 'segmentation_strategy')
    test_segmentation_exploration(map_file_path, start_points_list, segmentation_output_dir)
    
    # 測試基於貪婪frontier的探索策略（直接使用test.py中的實現）
    # 這裡您需要導入test.py中的函數
    try:
        from test import test_multiple_start_points_greedy_same_target_random
        
        greedy_output_dir = os.path.join(base_output_dir, 'greedy_strategy')
        test_multiple_start_points_greedy_same_target_random(map_file_path, start_points_list, greedy_output_dir)
        
        # 比較兩種策略
        compare_strategies(segmentation_output_dir, greedy_output_dir, os.path.join(base_output_dir, 'strategy_comparison'))
    except ImportError:
        print("警告: 無法導入test.py中的函數，只運行分割策略測試")

def compare_strategies(segmentation_dir, greedy_dir, output_dir):
    """比較不同探索策略的性能
    
    Args:
        segmentation_dir: 分割策略的結果目錄
        greedy_dir: 貪婪策略的結果目錄
        output_dir: 輸出比較結果的目錄
    """
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 讀取兩種策略的覆蓋率數據
    segmentation_data = {}
    greedy_data = {}
    
    try:
        # 讀取分割策略的數據
        with open(os.path.join(segmentation_dir, 'coverage_data.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                start_point = int(row['StartPoint'])
                step = int(row['Step'])
                
                if start_point not in segmentation_data:
                    segmentation_data[start_point] = {'steps': [], 'union': []}
                
                segmentation_data[start_point]['steps'].append(step)
                segmentation_data[start_point]['union'].append(float(row['UnionCoverage']))
        
        # 讀取貪婪策略的數據
        with open(os.path.join(greedy_dir, 'coverage_data.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                start_point = int(row['StartPoint'])
                step = int(row['Step'])
                
                if start_point not in greedy_data:
                    greedy_data[start_point] = {'steps': [], 'union': []}
                
                greedy_data[start_point]['steps'].append(step)
                greedy_data[start_point]['union'].append(float(row['UnionCoverage']))
        
        # 為每個起始點創建比較圖表
        for start_point in sorted(set(segmentation_data.keys()) & set(greedy_data.keys())):
            plt.figure(figsize=(12, 8))
            
            # 繪製分割策略曲線
            plt.plot(segmentation_data[start_point]['steps'], segmentation_data[start_point]['union'], 
                    'b-', linewidth=2, label='Segmentation Strategy')
            
            # 繪製貪婪策略曲線
            plt.plot(greedy_data[start_point]['steps'], greedy_data[start_point]['union'], 
                    'r-', linewidth=2, label='Greedy Strategy')
            
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Total Coverage (Union)', fontsize=14)
            plt.title(f'Strategy Comparison - Start Point {start_point}', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.ylim(0, 1.05)
            
            plt.savefig(os.path.join(output_dir, f'strategy_comparison_startpoint_{start_point}.png'), dpi=300)
            plt.close()
        
        # 創建所有起始點的平均比較圖表
        plt.figure(figsize=(12, 8))
        
        # 計算每個策略的平均覆蓋率
        seg_avg_coverage = calculate_average_coverage(segmentation_data)
        greedy_avg_coverage = calculate_average_coverage(greedy_data)
        
        # 繪製平均曲線
        plt.plot(seg_avg_coverage['steps'], seg_avg_coverage['union'], 
                'b-', linewidth=3, label='Segmentation Strategy (Avg)')
        plt.plot(greedy_avg_coverage['steps'], greedy_avg_coverage['union'], 
                'r-', linewidth=3, label='Greedy Strategy (Avg)')
        
        plt.xlabel('Time (steps)', fontsize=14)
        plt.ylabel('Average Total Coverage', fontsize=14)
        plt.title('Average Coverage Comparison Between Strategies', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.ylim(0, 1.05)
        
        plt.savefig(os.path.join(output_dir, 'average_coverage_comparison.png'), dpi=300)
        plt.close()
        
        print(f"已生成策略比較圖表，儲存在: {output_dir}")
    
    except Exception as e:
        print(f"生成策略比較圖表時出錯: {str(e)}")
        import traceback
        traceback.print_exc()

def calculate_average_coverage(data_dict):
    """計算平均覆蓋率
    
    Args:
        data_dict: 包含各起始點覆蓋率數據的字典
        
    Returns:
        包含平均步數和覆蓋率的字典
    """
    # 找出最大步數
    max_steps = max(max(data['steps']) for data in data_dict.values())
    
    # 初始化結果
    result = {'steps': list(range(max_steps + 1)), 'union': [0] * (max_steps + 1)}
    counts = [0] * (max_steps + 1)
    
    # 累加每個起始點的覆蓋率
    for start_point, data in data_dict.items():
        for step, coverage in zip(data['steps'], data['union']):
            result['union'][step] += coverage
            counts[step] += 1
    
    # 計算平均值
    for step in range(max_steps + 1):
        if counts[step] > 0:
            result['union'][step] /= counts[step]
        else:
            # 如果沒有數據，使用前一步的覆蓋率
            if step > 0:
                result['union'][step] = result['union'][step - 1]
    
    return result

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
        [[90, 300], [90, 300]],   # 起始點 9
        [[260, 200], [260, 200]],  # 起始點 10
    ]
    
    # 設置輸出目錄
    output_dir = 'results_segmentation_exploration2'
    
    # 運行分割策略測試
    test_segmentation_exploration(map_file_path, start_points, output_dir)
    
    # 如果要比較不同策略，取消下面代碼的註釋
    # comparison_dir = 'strategy_comparison_results'
    # test_and_compare_strategies(map_file_path, start_points, comparison_dir)

if __name__ == '__main__':
    main()