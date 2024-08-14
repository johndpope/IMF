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
    
    # Determine the maximum number of frames across all videos in the batch
    max_frames = max(len(item['frames']) for item in batch)
    
    # Determine the maximum dimensions across all frames in the batch
    max_height = max(max(frame.shape[1] for frame in item['frames']) for item in batch)
    max_width = max(max(frame.shape[2] for frame in item['frames']) for item in batch)
    
    # Pad and stack frames for each item in the batch
    padded_frames = []
    for item in batch:
        frames = item['frames']
        # Pad each frame to max_height and max_width
        padded_item_frames = [F.pad(frame, (0, max_width - frame.shape[2], 0, max_height - frame.shape[1])) for frame in frames]
        # Pad the number of frames to max_frames
        while len(padded_item_frames) < max_frames:
            padded_item_frames.append(torch.zeros_like(padded_item_frames[0]))
        # Stack frames for this item
        padded_frames.append(torch.stack(padded_item_frames))
    
    # Stack all padded frame tensors in the batch
    frames_tensor = torch.stack(padded_frames)
    
    # Assert the correct shape of the output tensor
    assert frames_tensor.ndim == 5, "Frames tensor should be 5D (batch, frames, channels, height, width)"
    
    return {'frames': frames_tensor}

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_skip=0, num_frames=2):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.is_gcs = root_dir.startswith('gs://')
        if self.is_gcs:
            self.client = storage.Client()
            self.bucket_name = root_dir.split('/')[2]
            self.bucket = self.client.get_bucket(self.bucket_name)
            self.prefix = '/'.join(root_dir.split('/')[3:])
        self.video_folders = self._list_folders()
        self.video_frames = self._count_frames()

    def _list_folders(self):
        if self.is_gcs:
            blobs = self.bucket.list_blobs(prefix=self.prefix, delimiter='/')
            return [f"gs://{self.bucket_name}/{prefix}" for prefix in blobs.prefixes]
        else:
            return [f.path for f in os.scandir(self.root_dir) if f.is_dir()]

    def _count_frames(self):
        video_frames = []
        for folder in self.video_folders:
            if self.is_gcs:
                blobs = self.bucket.list_blobs(prefix=folder.split(f"gs://{self.bucket_name}/")[1])
                frames = [blob.name for blob in blobs if blob.name.endswith('.jpg')]
            else:
                frames = [f for f in os.listdir(folder) if f.endswith('.jpg')]
            video_frames.append(len(frames))
        return video_frames

    def __len__(self):
        return len(self.video_folders)

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        transformed_images = [transform(img) for img in images]
        return torch.stack(transformed_images, dim=0)  # (f, c, h, w)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        if self.is_gcs:
            blobs = list(self.bucket.list_blobs(prefix=video_folder.split(f"gs://{self.bucket_name}/")[1]))
            frames = sorted([blob.name for blob in blobs if blob.name.endswith('.jpg')])
        else:
            frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
        
        if len(frames) < self.num_frames:
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))
        
        start_idx = random.randint(0, len(frames) - self.num_frames)
        frame_indices = range(start_idx, start_idx + self.num_frames)
        
        vid_pil_image_list = []
        for i in frame_indices:
            if self.is_gcs:
                blob = self.bucket.blob(frames[i])
                img_bytes = blob.download_as_bytes()
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
            else:
                img_path = os.path.join(video_folder, frames[i])
                img = Image.open(img_path).convert('RGB')
            vid_pil_image_list.append(img)

        if self.transform:
            state = torch.get_rng_state()
            vid_tensor = self.augmentation(vid_pil_image_list, self.transform, state)
        else:
            vid_tensor = torch.stack([transforms.ToTensor()(img) for img in vid_pil_image_list])

        return {
            "frames": vid_tensor,
            "video_name": os.path.basename(video_folder)
        }


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = VideoDataset("/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/images", 
                           transform=transform, 
                           frame_skip=0, 
                           num_frames=2)
    print(f"Total videos in dataset: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Number of PIL images: {len(sample['frames'])}")
    print(f"Tensor shape: {sample['tensor'].shape}")
    print(f"Video name: {sample['video_name']}")