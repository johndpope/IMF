import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
import os
from typing import List, Tuple, Dict, Any
from decord import VideoReader, cpu
import torch.nn.functional as F


class VideoAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_skip=0, num_frames=400):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip
        self.num_frames = num_frames
        
        # Load dataset metadata
        with open(os.path.join(root_dir, 'dataset.json'), 'r') as f:
            self.dataset_metadata = json.load(f)
            
        self.videos = self.dataset_metadata['videos']
        self.frame_rate = self.dataset_metadata['frame_rate']
        self.audio_sample_rate = self.dataset_metadata['audio_sample_rate']

    def __len__(self):
        return len(self.videos)

    def load_frame(self, frame_path: str):
        """Load and transform a single frame"""
        img = Image.open(frame_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def load_audio_chunk(self, chunk_path: str) -> np.ndarray:
        """Load an audio chunk"""
        return np.load(chunk_path)

    def __getitem__(self, idx):
        video_metadata = self.videos[idx]
        video_folder = os.path.join(self.root_dir, 
                                  os.path.splitext(video_metadata['video_path'])[0])
                                  
        # Get frame data
        frame_metadata = video_metadata['frames']
        total_frames = frame_metadata['total_frames']
        
        # Select frame range
        if total_frames < self.num_frames:
            frame_indices = [i % total_frames for i in range(self.num_frames)]
        else:
            start_idx = random.randint(0, total_frames - self.num_frames)
            frame_indices = range(start_idx, start_idx + self.num_frames)

        # Load frames
        frames = []
        for frame_idx in frame_indices:
            frame_info = frame_metadata['frames'][frame_idx]
            frame_path = os.path.join(video_folder, frame_info['path'])
            frame = self.load_frame(frame_path)
            frames.append(frame)

        # Load corresponding audio chunks
        audio_metadata = video_metadata['audio']
        audio_chunks = []
        for frame_idx in frame_indices:
            chunk_info = audio_metadata['chunks'][frame_idx]
            chunk_path = os.path.join(video_folder, chunk_info['path'])
            chunk = self.load_audio_chunk(chunk_path)
            audio_chunks.append(chunk)

        return {
            "frames": torch.stack(frames) if isinstance(frames[0], torch.Tensor) else frames,
            "audio": torch.from_numpy(np.stack(audio_chunks)),
            "video_name": os.path.basename(video_folder),
            "frame_indices": frame_indices,
            "sample_rate": self.audio_sample_rate,
            "frame_rate": self.frame_rate
        }