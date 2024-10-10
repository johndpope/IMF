import os
import cv2
from tqdm import tqdm
import random

def extract_quadrants(image):
    height, width = image.shape[:2]
    mid_x, mid_y = width // 2, height // 2
    
    top_left = image[0:mid_y, 0:mid_x]
    top_right = image[0:mid_y, mid_x:width]
    bottom_left = image[mid_y:height, 0:mid_x]
    bottom_right = image[mid_y:height, mid_x:width]
    
    return [top_left, top_right, bottom_left, bottom_right]

def extract_frames(video_path, output_folder, frame_skip=0, max_frames=None, save_quadrants=False):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    success, image = video.read()
    count = 0
    saved_count = 0
    
    for _ in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}"):
        if success:
            if count % (frame_skip + 1) == 0:
                frame_name = f"frame_{saved_count:06d}.png"
                cv2.imwrite(os.path.join(output_folder, frame_name), image)
                
                if save_quadrants:
                    quadrants = extract_quadrants(image)
                    for i, quadrant in enumerate(quadrants):
                        quadrant_name = f"frame_{saved_count:06d}_quadrant_{i}.png"
                        cv2.imwrite(os.path.join(output_folder, quadrant_name), quadrant)
                
                saved_count += 1
                if max_frames and saved_count >= max_frames:
                    break
            count += 1
        success, image = video.read()
    
    video.release()
    print(f"Extracted {saved_count} frames from {os.path.basename(video_path)}")

def process_videos(input_folder, output_base_folder, frame_skip=0, max_videos=1000, max_frames_per_video=100, save_quadrants=False):
    video_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    # Shuffle and limit the number of videos
    random.shuffle(video_files)
    video_files = video_files[:max_videos]
    
    for video_path in video_files:
        relative_path = os.path.relpath(video_path, input_folder)
        video_name = os.path.splitext(relative_path)[0]
        output_folder = os.path.join(output_base_folder, video_name)
        extract_frames(video_path, output_folder, frame_skip, max_frames_per_video, save_quadrants)

# Usage
input_folder = './'
output_base_folder = './data'
frame_skip = 0  # Adjust this if you want to skip frames
max_videos = 100  # Limit the number of videos to process
max_frames_per_video = 1000  # Limit the number of frames per video
save_quadrants = True  # Set to True to save quadrants as separate PNG files

process_videos(input_folder, output_base_folder, frame_skip, max_videos, max_frames_per_video, save_quadrants)