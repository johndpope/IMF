


import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from google.cloud import storage
from io import BytesIO
from typing import List,  Dict, Any
import torch.nn.functional as F

def gpu_padded_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    assert isinstance(batch, list), "Batch should be a list"
    
    # Separate source and reference frames
    source_frames = [item['source_frame'] for item in batch]
    ref_frames = [item['ref_frame'] for item in batch]
    
    # Determine the maximum dimensions across all frames in the batch
    max_height = max(max(frame.shape[1] for frame in source_frames + ref_frames))
    max_width = max(max(frame.shape[2] for frame in source_frames + ref_frames))
    
    # Pad frames
    padded_source_frames = []
    padded_ref_frames = []
    
    for source, ref in zip(source_frames, ref_frames):
        # Pad source frame
        padded_source = F.pad(source, (0, max_width - source.shape[2], 0, max_height - source.shape[1]))
        padded_source_frames.append(padded_source)
        
        # Pad reference frame
        padded_ref = F.pad(ref, (0, max_width - ref.shape[2], 0, max_height - ref.shape[1]))
        padded_ref_frames.append(padded_ref)
    
    # Stack all padded frame tensors in the batch
    source_tensor = torch.stack(padded_source_frames)
    ref_tensor = torch.stack(padded_ref_frames)
    
    # Collect other metadata
    video_names = [item['video_name'] for item in batch]
    source_indices = torch.tensor([item['source_index'] for item in batch])
    ref_indices = torch.tensor([item['ref_index'] for item in batch])
    
    return {
        'source_frames': source_tensor,
        'ref_frames': ref_tensor,
        'video_names': video_names,
        'source_indices': source_indices,
        'ref_indices': ref_indices
    }



class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_folders = self._list_folders()
        self.video_frames = self._count_frames()

    def _list_folders(self):
        return [f.path for f in os.scandir(self.root_dir) if f.is_dir()]

    def _count_frames(self):
        video_frames = []
        for folder in self.video_folders:
            frames = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            video_frames.append(len(frames))
        return video_frames

    def __len__(self):
        return sum(self.video_frames)

    def __getitem__(self, idx):
        # Find which video and frame this index corresponds to
        video_idx = 0
        while idx >= self.video_frames[video_idx]:
            idx -= self.video_frames[video_idx]
            video_idx += 1

        video_folder = self.video_folders[video_idx]
        frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
        
        # Get the current frame as source
        source_frame = frames[idx]
        source_path = os.path.join(video_folder, source_frame)
        source_img = Image.open(source_path).convert('RGB')

        # Randomly select a reference frame from the same video
        ref_idx = random.randint(0, len(frames) - 1)
        while ref_idx == idx:  # Ensure reference frame is different from source
            ref_idx = random.randint(0, len(frames) - 1)
        ref_frame = frames[ref_idx]
        ref_path = os.path.join(video_folder, ref_frame)
        ref_img = Image.open(ref_path).convert('RGB')

        if self.transform:
            source_tensor = self.transform(source_img)
            ref_tensor = self.transform(ref_img)
        else:
            source_tensor = transforms.ToTensor()(source_img)
            ref_tensor = transforms.ToTensor()(ref_img)

        return {
            "source_frame": source_tensor,
            "ref_frame": ref_tensor,
            "video_name": os.path.basename(video_folder),
            "source_index": idx,
            "ref_index": ref_idx
        }
    
# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Assume the GCS bucket is mounted at /mnt/gcs_bucket
    dataset = VideoDataset("/mnt/gcs_bucket/CelebV-HQ/celebvhq/35666/images", 
                           transform=transform, 
                           frame_skip=0, 
                           num_frames=2)
    print(f"Total videos in dataset: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Number of frames: {sample['frames'].shape[0]}")
    print(f"Tensor shape: {sample['frames'].shape}")
    print(f"Video name: {sample['video_name']}")