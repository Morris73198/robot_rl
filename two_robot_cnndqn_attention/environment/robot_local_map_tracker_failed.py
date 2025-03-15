import numpy as np
import matplotlib.pyplot as plt
import os

class RobotIndividualMapTracker:
    """
    追蹤並紀錄兩個機器人的個人探索地圖（只包含自己探索的區域）
    地圖大小與op_map相同
    兼容 multi_robot_with_unknown 和 multi_robot_no_unknown
    """
    
    def __init__(self, robot1, robot2, save_dir='robot_individual_maps'):
        """
        初始化追蹤器
        
        參數:
            robot1: 第一個機器人實例
            robot2: 第二個機器人實例
            save_dir: 保存地圖的目錄
        """
        self.robot1 = robot1
        self.robot2 = robot2
        self.save_dir = save_dir
        
        # 確保保存目錄存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 初始化每個機器人的個人地圖（只有自己探索的區域）
        self.robot1_individual_map = None
        self.robot2_individual_map = None
        
        # 記錄前一步的位置，用於計算新探索區域
        self.last_robot1_position = None
        self.last_robot2_position = None
        
        # 記錄地圖的歷史
        self.robot1_maps = []
        self.robot2_maps = []
        
        # 是否正在追蹤
        self.is_tracking = False
        
        # 可視化相關設置
        self.fig = None
        self.axes = None
    
    def start_tracking(self):
        """開始追蹤個人地圖"""
        self.is_tracking = True
        
        # 初始化個人地圖為未知區域 (值為127)
        map_shape = self.robot1.op_map.shape
        self.robot1_individual_map = np.ones(map_shape) * 127
        self.robot2_individual_map = np.ones(map_shape) * 127
        
        # 記錄初始位置
        self.last_robot1_position = self.robot1.robot_position.copy()
        self.last_robot2_position = self.robot2.robot_position.copy()
        
        # 清空地圖歷史
        self.robot1_maps = []
        self.robot2_maps = []
        
        print("開始追蹤機器人個人探索地圖")
    
    def stop_tracking(self):
        """停止追蹤個人地圖"""
        self.is_tracking = False
        print(f"停止追蹤機器人個人探索地圖，共記錄了 {len(self.robot1_maps)} 個時間點")
    
    def update(self):
        """更新兩個機器人的個人探索地圖"""
        if not self.is_tracking:
            return
        
        # 檢查機器人是否已經初始化
        if self.robot1_individual_map is None or self.robot2_individual_map is None:
            self.start_tracking()
        
        # 獲取當前位置
        robot1_position = self.robot1.robot_position.copy()
        robot2_position = self.robot2.robot_position.copy()
        
        # 檢測機器人1的移動並更新其個人地圖
        if not np.array_equal(robot1_position, self.last_robot1_position):
            # 計算機器人1新探索的區域（只更新自己的感知範圍）
            self._update_individual_map(
                self.robot1, 
                self.robot1_individual_map,
                robot1_position, 
                self.last_robot1_position
            )
            self.last_robot1_position = robot1_position.copy()
        
        # 檢測機器人2的移動並更新其個人地圖
        if not np.array_equal(robot2_position, self.last_robot2_position):
            # 計算機器人2新探索的區域（只更新自己的感知範圍）
            self._update_individual_map(
                self.robot2, 
                self.robot2_individual_map,
                robot2_position, 
                self.last_robot2_position
            )
            self.last_robot2_position = robot2_position.copy()
        
        # 獲取當前地圖並標記機器人位置
        robot1_map = self._get_map_with_robot(self.robot1_individual_map, robot1_position)
        robot2_map = self._get_map_with_robot(self.robot2_individual_map, robot2_position)
        
        # 保存到歷史記錄
        self.robot1_maps.append(robot1_map)
        self.robot2_maps.append(robot2_map)
    
    def _update_individual_map(self, robot, individual_map, current_position, last_position):
        """更新機器人的個人探索地圖
        
        參數:
            robot: 機器人實例
            individual_map: 機器人的個人地圖
            current_position: 當前位置
            last_position: 上一個位置
        """
        # 獲取感知範圍
        sensor_range = robot.sensor_range
        
        # 計算感知區域的範圍
        x, y = int(current_position[0]), int(current_position[1])
        min_x = max(0, x - sensor_range)
        max_x = min(individual_map.shape[1] - 1, x + sensor_range)
        min_y = max(0, y - sensor_range)
        max_y = min(individual_map.shape[0] - 1, y + sensor_range)
        
        # 只更新感知範圍內的區域
        local_area = robot.op_map[min_y:max_y+1, min_x:max_x+1]
        individual_map[min_y:max_y+1, min_x:max_x+1] = local_area
    
    def _get_map_with_robot(self, map_data, position):
        """在地圖上標記機器人位置
        
        參數:
            map_data: 地圖數據
            position: 機器人位置
            
        返回:
            帶有機器人標記的地圖副本
        """
        # 創建副本避免修改原始地圖
        map_copy = map_data.copy()
        
        # 標記機器人位置
        x, y = int(position[0]), int(position[1])
        robot_size = 3  # 標記大小
        
        # 確保不超出地圖邊界
        min_x = max(0, x - robot_size)
        max_x = min(map_copy.shape[1] - 1, x + robot_size)
        min_y = max(0, y - robot_size)
        max_y = min(map_copy.shape[0] - 1, y + robot_size)
        
        # 標記機器人位置為76
        map_copy[min_y:max_y+1, min_x:max_x+1] = 76
        
        return map_copy
    
    def save_current_maps(self, step):
        """保存當前的個人探索地圖"""
        if not self.is_tracking or not self.robot1_maps or not self.robot2_maps:
            return
        
        # 獲取最新的地圖
        robot1_map = self.robot1_maps[-1]
        robot2_map = self.robot2_maps[-1]
        
        # 保存為圖片
        plt.figure(figsize=(20, 10))
        
        plt.subplot(1, 2, 1)
        plt.imshow(robot1_map, cmap='gray')
        plt.title('Robot1 Individual Exploration')
        plt.colorbar(label='Map Value')
        
        plt.subplot(1, 2, 2)
        plt.imshow(robot2_map, cmap='gray')
        plt.title('Robot2 Individual Exploration')
        plt.colorbar(label='Map Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'individual_maps_step_{step:04d}.png'), dpi=150)
        plt.close()
    
    def visualize_maps(self):
        """實時可視化兩個機器人的個人探索地圖"""
        if not self.is_tracking or not self.robot1_maps or not self.robot2_maps:
            return
        
        # 獲取最新的地圖
        robot1_map = self.robot1_maps[-1]
        robot2_map = self.robot2_maps[-1]
        
        # 初始化圖形
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(20, 10))
            plt.ion()  # 開啟互動模式
        
        # 更新圖形
        self.axes[0].clear()
        im1 = self.axes[0].imshow(robot1_map, cmap='gray')
        self.axes[0].set_title('Robot1 Individual Exploration')
        plt.colorbar(im1, ax=self.axes[0], label='Map Value')
        
        self.axes[1].clear()
        im2 = self.axes[1].imshow(robot2_map, cmap='gray')
        self.axes[1].set_title('Robot2 Individual Exploration')
        plt.colorbar(im2, ax=self.axes[1], label='Map Value')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def save_map_history(self, interval=10):
        """保存所有記錄的個人地圖歷史（每隔interval步保存一次）"""
        if not self.robot1_maps or not self.robot2_maps:
            print("沒有記錄的地圖歷史")
            return
        
        history_dir = os.path.join(self.save_dir, 'history')
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        # 保存地圖歷史
        for i in range(0, len(self.robot1_maps), interval):
            if i < len(self.robot1_maps) and i < len(self.robot2_maps):
                robot1_map = self.robot1_maps[i]
                robot2_map = self.robot2_maps[i]
                
                plt.figure(figsize=(20, 10))
                
                plt.subplot(1, 2, 1)
                plt.imshow(robot1_map, cmap='gray')
                plt.title(f'Robot1 Individual Map - Step {i}')
                plt.colorbar(label='Map Value')
                
                plt.subplot(1, 2, 2)
                plt.imshow(robot2_map, cmap='gray')
                plt.title(f'Robot2 Individual Map - Step {i}')
                plt.colorbar(label='Map Value')
                
                plt.tight_layout()
                plt.savefig(os.path.join(history_dir, f'individual_maps_history_{i:04d}.png'), dpi=150)
                plt.close()
        
        print(f"已保存個人地圖歷史，共 {len(range(0, len(self.robot1_maps), interval))} 幀")
    
    def generate_exploration_video(self):
        """生成個人探索過程的視頻（需要安裝 imageio 和 ffmpeg）"""
        try:
            import imageio
            import subprocess
        except ImportError:
            print("需要安裝 imageio 庫才能生成視頻。請執行 'pip install imageio-ffmpeg imageio'")
            return
        
        if not self.robot1_maps or not self.robot2_maps:
            print("沒有記錄的地圖歷史，無法生成視頻")
            return
        
        video_dir = os.path.join(self.save_dir, 'video')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        
        # 為 robot1 生成視頻
        print("正在生成 Robot1 個人探索視頻...")
        writer = imageio.get_writer(os.path.join(video_dir, 'robot1_individual_exploration.mp4'), fps=5)
        
        for map_data in self.robot1_maps:
            plt.figure(figsize=(12, 10))
            plt.imshow(map_data, cmap='gray')
            plt.title('Robot1 Individual Exploration')
            plt.colorbar(label='Map Value')
            
            # 保存為臨時文件
            plt.savefig('temp_frame.png')
            plt.close()
            
            # 添加到視頻
            writer.append_data(imageio.imread('temp_frame.png'))
        
        writer.close()
        
        # 為 robot2 生成視頻
        print("正在生成 Robot2 個人探索視頻...")
        writer = imageio.get_writer(os.path.join(video_dir, 'robot2_individual_exploration.mp4'), fps=5)
        
        for map_data in self.robot2_maps:
            plt.figure(figsize=(12, 10))
            plt.imshow(map_data, cmap='gray')
            plt.title('Robot2 Individual Exploration')
            plt.colorbar(label='Map Value')
            
            # 保存為臨時文件
            plt.savefig('temp_frame.png')
            plt.close()
            
            # 添加到視頻
            writer.append_data(imageio.imread('temp_frame.png'))
        
        writer.close()
        
        # 刪除臨時文件
        if os.path.exists('temp_frame.png'):
            os.remove('temp_frame.png')
        
        print(f"視頻已生成，保存在 {video_dir} 目錄")
    
    def get_exploration_ratio(self):
        """獲取兩個機器人的個人探索比例"""
        if self.robot1_individual_map is None or self.robot2_individual_map is None:
            return 0, 0
        
        # 計算已探索區域（值為255的區域）
        robot1_explored = np.sum(self.robot1_individual_map == 255)
        robot2_explored = np.sum(self.robot2_individual_map == 255)
        
        # 計算全局地圖的可探索區域總數
        total_explorable = np.sum(self.robot1.global_map == 255)
        
        # 計算比例
        robot1_ratio = robot1_explored / total_explorable if total_explorable > 0 else 0
        robot2_ratio = robot2_explored / total_explorable if total_explorable > 0 else 0
        
        return robot1_ratio, robot2_ratio
    
    def calculate_overlap(self):
        """計算兩個機器人探索區域的重疊程度"""
        if self.robot1_individual_map is None or self.robot2_individual_map is None:
            return 0
        
        # 計算兩個機器人都探索過的區域（兩個地圖中值都是255的區域）
        overlap = np.sum((self.robot1_individual_map == 255) & (self.robot2_individual_map == 255))
        
        # 計算任一機器人探索過的區域
        any_explored = np.sum((self.robot1_individual_map == 255) | (self.robot2_individual_map == 255))
        
        # 計算重疊比例
        overlap_ratio = overlap / any_explored if any_explored > 0 else 0
        
        return overlap_ratio
    
    def cleanup(self):
        """清理資源"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
        
        # 釋放記憶體
        self.robot1_maps = []
        self.robot2_maps = []
        self.robot1_individual_map = None
        self.robot2_individual_map = None
        self.is_tracking = False