import os
import cv2
import numpy as np
from tqdm import tqdm

def create_video(image_folder, output_path, fps=3):
    """
    將圖片序列轉換為影片
    
    參數:
        image_folder: 包含圖片序列的資料夾路徑
        output_path: 輸出影片的路徑
        fps: 影片幀率
    """
    # 獲取所有圖片檔案
    images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    images.sort()  # 確保按正確順序排序
    
    if not images:
        print(f"在 {image_folder} 中未找到圖片")
        return
    
    # 讀取第一張圖片以獲取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    
    # 定義影片編碼器和輸出影片物件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"正在從 {len(images)} 幀創建影片...")
    for image in tqdm(images):
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)
    
    # 釋放影片物件
    video.release()
    print(f"影片已保存至 {output_path}")

def process_all_episodes(base_dir="test_file", output_dir="videos"):
    """
    處理 test_file 目錄下的所有 episode 資料夾
    
    參數:
        base_dir: 基礎目錄，包含所有 episode 資料夾
        output_dir: 輸出影片的目錄
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取所有 episode 資料夾
    episode_folders = [d for d in os.listdir(base_dir) if d.startswith("episode_")]
    episode_folders.sort(key=lambda x: int(x.split("_")[1]))
    
    print(f"找到 {len(episode_folders)} 個 episodes 待處理")
    
    # 處理每個 episode
    for episode_folder in episode_folders:
        print(f"\n正在處理 {episode_folder}...")
        
        input_folder = os.path.join(base_dir, episode_folder)
        output_path = os.path.join(output_dir, f"{episode_folder}.mp4")
        
        create_video(input_folder, output_path)

if __name__ == "__main__":
    # 設置基本參數
    BASE_DIR = "test_file"  # 包含所有 episode 資料夾的目錄
    OUTPUT_DIR = "videos"   # 輸出影片的目錄
    
    # 檢查基礎目錄是否存在
    if not os.path.exists(BASE_DIR):
        print(f"錯誤：基礎目錄 {BASE_DIR} 不存在！")
        exit(1)
    
    # 處理所有 episodes
    process_all_episodes(BASE_DIR, OUTPUT_DIR)
    
    print("\n影片創建完成！")