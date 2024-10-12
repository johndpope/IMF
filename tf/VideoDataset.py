import os
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Any

class VideoDataset(tf.keras.utils.Sequence):
    def __init__(self, root_dir, transform=None, frame_skip=0, num_frames=400):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.video_folders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
        print(f"Found {len(self.video_folders)} video folders in {root_dir}")
        self.video_frames = self.count_frames()
        print(f"Total frames across all videos: {sum(self.video_frames)}")

    def count_frames(self):
        video_frames = []
        for folder in self.video_folders:
            frames = [f for f in os.listdir(folder) if f.endswith('.png')]
            video_frames.append(len(frames))
            print(f"Folder {folder}: {len(frames)} frames")
        return video_frames

    def __len__(self):
        return len(self.video_folders)

    def load_and_transform_frame(self, frame_path):
        try:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = img.resize((256, 256))  # Default resizing if no transform is provided
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            return img
        except (OSError, IOError) as e:
            print(f"Error loading image {frame_path}: {str(e)}")
            raise  # Re-raise the exception to be caught in __getitem__

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.png')])
        print(f"Processing video folder: {video_folder}")
        print(f"Number of frames found: {len(frames)}")

        if not frames:
            print(f"No frames found in folder: {video_folder}")
            return self.__getitem__((idx + 1) % len(self))  # Move to next item

        total_frames = len(frames)
        if total_frames < self.num_frames:
            frame_indices = [i % total_frames for i in range(self.num_frames)]
        else:
            start_idx = random.randint(0, total_frames - self.num_frames)
            frame_indices = range(start_idx, start_idx + self.num_frames, self.frame_skip + 1)

        loaded_frames = []
        for i in frame_indices:
            img_path = os.path.join(video_folder, frames[i % total_frames])
            try:
                frame = self.load_and_transform_frame(img_path)
                loaded_frames.append(frame)
            except (OSError, IOError) as e:
                print(f"Error loading image {img_path}: {str(e)}")
                continue  # Skip this frame and continue with the next

        if not loaded_frames:
            print(f"No valid frames loaded from {video_folder}")
            return self.__getitem__((idx + 1) % len(self))  # Move to next item

        frames_tensor = np.stack(loaded_frames)  # Shape: (num_frames, height, width, channels)
        # Transpose the tensor to match the expected shape (frames, channels, height, width)
        frames_tensor = np.transpose(frames_tensor, (0, 3, 1, 2))
        print(f"Loaded {frames_tensor.shape[0]} frames for video {os.path.basename(video_folder)}")
        print(f"Frames tensor shape: {frames_tensor.shape}")

        return {
            "frames": frames_tensor,  # NumPy array with shape (frames, channels, height, width)
            "video_name": os.path.basename(video_folder)
        }

# Example usage
if __name__ == "__main__":
    # Define a simple transform function (optional)
    def transform(image):
        return image.resize((256, 256))

    dataset = VideoDataset(
        root_dir="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/images",
        transform=transform,
        frame_skip=0,
        num_frames=300
    )
    print(f"Total videos in dataset: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Number of frames: {sample['frames'].shape[0]}")
    print(f"Frames tensor shape: {sample['frames'].shape}")
    print(f"Video name: {sample['video_name']}")