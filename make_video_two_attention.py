import os
import cv2
import numpy as np
from glob import glob
import argparse
from tqdm import tqdm

def create_test_video(episode_dir, output_path, fps=10, frame_size=None):
    """
    Create a side-by-side video from test output images
    
    Args:
        episode_dir: Path to episode directory containing test images
        output_path: Path for the output video
        fps: Video frame rate
        frame_size: Size for each robot view (width, height), if None uses original size
    """
    # Get image lists for both robots
    robot1_images = sorted(glob(os.path.join(episode_dir, 'robot1_*.png')))
    robot2_images = sorted(glob(os.path.join(episode_dir, 'robot2_*.png')))
    
    if not robot1_images or not robot2_images:
        print(f"No images found in {episode_dir}")
        return False
    
    # Ensure equal number of images for both robots
    if len(robot1_images) != len(robot2_images):
        print("Warning: Different number of images for robots")
        min_images = min(len(robot1_images), len(robot2_images))
        robot1_images = robot1_images[:min_images]
        robot2_images = robot2_images[:min_images]
    
    # Read first frames to get dimensions
    frame1 = cv2.imread(robot1_images[0])
    frame2 = cv2.imread(robot2_images[0])
    
    if frame1 is None or frame2 is None:
        print("Failed to read initial frames")
        return False
    
    # Get original dimensions
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Use original size if not specified
    if frame_size is None:
        frame_size = (max(w1, w2), max(h1, h2))
    
    # Set final dimensions
    single_width, single_height = frame_size
    total_width = single_width * 2  # Two images side by side
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (total_width, single_height))
    
    # Font settings for titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White
    thickness = 2
    
    # Process frames
    print(f"\nCreating video with {len(robot1_images)} frames...")
    for img1_path, img2_path in tqdm(zip(robot1_images, robot2_images)):
        # Extract step number from filename for progress indicator
        step = os.path.basename(img1_path).split('_')[2].split('.')[0]
        
        # Read frames
        frame1 = cv2.imread(img1_path)
        frame2 = cv2.imread(img2_path)
        
        if frame1 is None or frame2 is None:
            print(f"Warning: Failed to read frames: {img1_path} or {img2_path}")
            continue
        
        # Resize frames
        frame1 = cv2.resize(frame1, (single_width, single_height))
        frame2 = cv2.resize(frame2, (single_width, single_height))
        
        # Create combined frame
        combined_frame = np.zeros((single_height, total_width, 3), dtype=np.uint8)
        combined_frame[:, :single_width] = frame1
        combined_frame[:, single_width:] = frame2
        
        # Add titles and step counter
        cv2.putText(combined_frame, 'Robot 1', (single_width//4, 30), 
                    font, font_scale, font_color, thickness)
        cv2.putText(combined_frame, 'Robot 2', (single_width + single_width//4, 30), 
                    font, font_scale, font_color, thickness)
        
        # Add step counter at the bottom
        cv2.putText(combined_frame, f'Step: {step}', (total_width//2 - 100, single_height - 20),
                    font, font_scale, font_color, thickness)
        
        # Write frame
        video.write(combined_frame)
    
    # Release resources
    video.release()
    print(f"Video saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create videos from test output images')
    parser.add_argument('--input', default='result_attention',
                        help='Input directory containing episode folders')
    parser.add_argument('--output', default='test_videos',
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
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Set frame size if specified
    frame_size = None
    if args.width and args.height:
        frame_size = (args.width, args.height)
    
    # Process specified episode or all episodes
    if args.episode:
        episode_dir = os.path.join(args.input, args.episode)
        if os.path.exists(episode_dir):
            output_path = os.path.join(args.output, f'test_{args.episode}.mp4')
            create_test_video(episode_dir, output_path, args.fps, frame_size)
        else:
            print(f"Episode directory not found: {episode_dir}")
    else:
        # Process all episodes
        episodes = sorted(glob(os.path.join(args.input, 'episode_*')))
        for episode_dir in episodes:
            episode_name = os.path.basename(episode_dir)
            print(f"\nProcessing {episode_name}...")
            output_path = os.path.join(args.output, f'test_{episode_name}.mp4')
            create_test_video(episode_dir, output_path, args.fps, frame_size)

if __name__ == '__main__':
    main()