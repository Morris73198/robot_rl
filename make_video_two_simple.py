import os
import cv2
import numpy as np
from glob import glob
import argparse
from tqdm import tqdm

def create_side_by_side_video(episode_dir, output_path, fps=10, frame_size=None):
    """
    創建兩個機器人並排的影片
    
    Args:
        episode_dir: episode目錄路徑
        output_path: 輸出影片的路徑
        fps: 影片的幀率
        frame_size: 每個機器人視圖的尺寸 (width, height)，如果為None則使用原始尺寸
    """
    # 獲取兩個機器人的圖片列表
    robot1_images = sorted(glob(os.path.join(episode_dir, 'robot1_*.png')))
    robot2_images = sorted(glob(os.path.join(episode_dir, 'robot2_*.png')))
    
    if not robot1_images or not robot2_images:
        print(f"No images found in {episode_dir}")
        return False
    
    # 確保兩個機器人有相同數量的圖片
    if len(robot1_images) != len(robot2_images):
        print("Warning: Different number of images for robots")
        min_images = min(len(robot1_images), len(robot2_images))
        robot1_images = robot1_images[:min_images]
        robot2_images = robot2_images[:min_images]
    
    # 讀取第一張圖片來獲取尺寸
    frame1 = cv2.imread(robot1_images[0])
    frame2 = cv2.imread(robot2_images[0])
    
    if frame1 is None or frame2 is None:
        print("Failed to read initial frames")
        return False
    
    # 獲取原始尺寸
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # 如果沒有指定尺寸，使用原始尺寸
    if frame_size is None:
        # 使用兩個圖片中較大的尺寸
        frame_size = (max(w1, w2), max(h1, h2))
    
    # 設置最終的尺寸
    single_width, single_height = frame_size
    total_width = single_width * 2  # 兩個圖片並排
    
    # 創建影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (total_width, single_height))
    
    # 添加標題設置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    thickness = 2
    
    print(f"Creating video from {len(robot1_images)} frames...")
    for img1_path, img2_path in tqdm(zip(robot1_images, robot2_images)):
        # 讀取兩個機器人的圖片
        frame1 = cv2.imread(img1_path)
        frame2 = cv2.imread(img2_path)
        
        if frame1 is None or frame2 is None:
            print(f"Warning: Failed to read frames: {img1_path} or {img2_path}")
            continue
        
        # 調整兩個圖片到相同的尺寸
        frame1 = cv2.resize(frame1, (single_width, single_height))
        frame2 = cv2.resize(frame2, (single_width, single_height))
        
        # 創建並排的圖片
        combined_frame = np.zeros((single_height, total_width, 3), dtype=np.uint8)
        combined_frame[:, :single_width] = frame1
        combined_frame[:, single_width:] = frame2
        
        # 添加標題
        cv2.putText(combined_frame, 'Robot 1', (single_width//4, 30), 
                    font, font_scale, font_color, thickness)
        cv2.putText(combined_frame, 'Robot 2', (single_width + single_width//4, 30), 
                    font, font_scale, font_color, thickness)
        
        # 寫入影片
        video.write(combined_frame)
    
    # 釋放資源
    video.release()
    print(f"Video saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create side-by-side videos from exploration step images')
    parser.add_argument('--input', default='result_simple',
                        help='Input directory containing episode folders')
    parser.add_argument('--output', default='videos',
                        help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for output video')
    parser.add_argument('--width', type=int, default=None,
                        help='Width for each robot view (optional)')
    parser.add_argument('--height', type=int, default=None,
                        help='Height for each robot view (optional)')
    parser.add_argument('--episode', type=str, default=None,
                        help='Process specific episode (e.g., "episode_001")')
    
    args = parser.parse_args()
    
    # 確保輸出目錄存在
    os.makedirs(args.output, exist_ok=True)
    
    # 設置每個視圖的尺寸
    frame_size = None
    if args.width and args.height:
        frame_size = (args.width, args.height)
    
    # 處理指定的episode或所有episodes
    if args.episode:
        episode_dir = os.path.join(args.input, args.episode)
        if os.path.exists(episode_dir):
            output_path = os.path.join(args.output, f'{args.episode}.mp4')
            create_side_by_side_video(episode_dir, output_path, args.fps, frame_size)
        else:
            print(f"Episode directory not found: {episode_dir}")
    else:
        # 處理所有episodes
        episodes = sorted(glob(os.path.join(args.input, 'episode_*')))
        for episode_dir in episodes:
            episode_name = os.path.basename(episode_dir)
            print(f"\nProcessing {episode_name}...")
            output_path = os.path.join(args.output, f'{episode_name}.mp4')
            create_side_by_side_video(episode_dir, output_path, args.fps, frame_size)

if __name__ == '__main__':
    main()