
import numpy as np

from typing import List, Dict, Any
import asyncio
from PIL import Image
import io
import json
from model import IMFModel
import base64
import ssl
import uvicorn
import logging
import torchvision.transforms as transforms
from VideoDataset import VideoDataset
from pathlib import Path
from starlette.websockets import WebSocketState
from fastapi.responses import JSONResponse
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
from typing import Dict, Optional
import time
class TokenCache:
    def __init__(self, max_size: int = 1000, cache_dir: str = "./cache"):
        self.max_size = max_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate storage for reference features and frame tokens
        self.reference_features: Dict[int, List[np.ndarray]] = {}  # video_id -> reference features
        self.video_tokens: Dict[int, Dict[int, np.ndarray]] = {}  # video_id -> {frame_id -> tokens}
        self.generation_status: Dict[int, Dict[int, bool]] = {}  # video_id -> {frame_id -> is_generated}
        self.lock = threading.Lock()
        
        # Load existing cache
        self.load_cache()
        
        # Set up periodic saving
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes

    def _maybe_save_cache(self):
        """Check if it's time to save the cache and save if needed"""
        current_time = time.time()
        if (current_time - self.last_save_time) >= self.save_interval:
            self.save_cache()

    def set_tokens(self, video_id: int, frame_id: int, tokens: Dict):
        """Set tokens in cache with LRU eviction"""
        with self.lock:
            if video_id not in self.video_tokens:
                self.video_tokens[video_id] = {}
            
            self.video_tokens[video_id][frame_id] = tokens
            
            # Update generation status
            if video_id not in self.generation_status:
                self.generation_status[video_id] = {}
            self.generation_status[video_id][frame_id] = True
            
            # LRU eviction if needed
            total_tokens = sum(len(frames) for frames in self.video_tokens.values())
            if total_tokens > self.max_size:
                self._perform_lru_eviction()
            
            # Maybe save to disk
            self._maybe_save_cache()

    def set_reference_features(self, video_id: int, features: List[np.ndarray]):
        """Set reference features for a video"""
        with self.lock:
            self.reference_features[video_id] = features
            # Save reference features immediately as they're used frequently
            self.save_reference_features(video_id)
            self._maybe_save_cache()


    def is_generated(self, video_id: int, frame_id: int) -> bool:
        """Check if tokens have been generated"""
        return self.generation_status.get(video_id, {}).get(frame_id, False)
    

    def get_tokens(self, video_id: int, frame_id: int) -> Optional[Dict]:
        """Get tokens from cache"""
        if video_id in self.video_tokens:
            return self.video_tokens[video_id].get(frame_id)
        return None


    def get_generation_progress(self, video_id: int) -> float:
        """Get token generation progress for a video"""
        if video_id not in self.generation_status:
            return 0.0
        
        total_frames = len(self.generation_status[video_id])
        if total_frames == 0:
            return 0.0
            
        generated_frames = sum(
            1 for is_generated in self.generation_status[video_id].values()
            if is_generated
        )
        return generated_frames / total_frames
    
   
                        
    def get_cache_path(self, video_id: int, is_reference: bool = False) -> Path:
        """Get path for cache file"""
        if is_reference:
            return self.cache_dir / f"video_{video_id}_reference.npz"
        return self.cache_dir / f"video_{video_id}_tokens.npz"

 

    def get_reference_features(self, video_id: int) -> Optional[List[np.ndarray]]:
        """Get reference features for a video"""
        return self.reference_features.get(video_id)

    def set_frame_tokens(self, video_id: int, frame_id: int, tokens: np.ndarray):
        """Set tokens for a specific frame"""
        with self.lock:
            if video_id not in self.video_tokens:
                self.video_tokens[video_id] = {}
            
            self.video_tokens[video_id][frame_id] = tokens
            
            # Update generation status
            if video_id not in self.generation_status:
                self.generation_status[video_id] = {}
            self.generation_status[video_id][frame_id] = True
            
            # LRU eviction if needed
            total_tokens = sum(len(frames) for frames in self.video_tokens.values())
            if total_tokens > self.max_size:
                self._perform_lru_eviction()

            # Save periodically
            self.save_cache()

    def _perform_lru_eviction(self):
        """Perform LRU eviction of tokens"""
        total_tokens = sum(len(frames) for frames in self.video_tokens.values())
        while total_tokens > self.max_size:
            # Find oldest video
            oldest_video = next(iter(self.video_tokens))
            oldest_frames = self.video_tokens[oldest_video]
            if oldest_frames:
                # Remove oldest frame
                oldest_frame = next(iter(oldest_frames))
                del oldest_frames[oldest_frame]
                if oldest_video in self.generation_status:
                    self.generation_status[oldest_video].pop(oldest_frame, None)
                total_tokens -= 1
            if not oldest_frames:
                del self.video_tokens[oldest_video]
                self.generation_status.pop(oldest_video, None)

    def load_cache(self):
        """Load cache from disk"""
        logger.info("Loading token cache from disk...")
        try:
            # Load generation status
            index_path = self.cache_dir / "cache_index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    index = json.load(f)
                    self.generation_status = {
                        int(k): {int(fk): fv for fk, fv in v.items()}
                        for k, v in index['generation_status'].items()
                    }

            # Load reference features
            for ref_file in self.cache_dir.glob("video_*_reference.npz"):
                try:
                    video_id = int(ref_file.stem.split('_')[1])
                    with np.load(ref_file, allow_pickle=True) as data:
                        self.reference_features[video_id] = [
                            data[f'feature_{i}'] for i in range(len(data.files))
                        ]
                    logger.info(f"Loaded reference features for video {video_id}")
                except Exception as e:
                    logger.error(f"Error loading reference features for video {video_id}: {str(e)}")

            # Load frame tokens
            for token_file in self.cache_dir.glob("video_*_tokens.npz"):
                try:
                    video_id = int(token_file.stem.split('_')[1])
                    with np.load(token_file) as data:
                        self.video_tokens[video_id] = {
                            int(k): data[k] for k in data.files
                        }
                    logger.info(f"Loaded {len(self.video_tokens[video_id])} tokens for video {video_id}")
                except Exception as e:
                    logger.error(f"Error loading tokens for video {video_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self.reference_features = {}
            self.video_tokens = {}
            self.generation_status = {}

    def save_cache(self, force: bool = False):
        """Save cache to disk"""
        current_time = time.time()
        if not force and (current_time - self.last_save_time) < self.save_interval:
            return

        with self.lock:
            try:
                # Save generation status
                index_path = self.cache_dir / "cache_index.json"
                index_data = {
                    'generation_status': {
                        str(k): {str(fk): fv for fk, fv in v.items()}
                        for k, v in self.generation_status.items()
                    },
                    'last_updated': current_time
                }
                with open(index_path, 'w') as f:
                    json.dump(index_data, f)

                # Save frame tokens
                for video_id, tokens in self.video_tokens.items():
                    cache_path = self.get_cache_path(video_id)
                    try:
                        save_data = {
                            str(frame_id): token.astype(np.float32)
                            for frame_id, token in tokens.items()
                        }
                        np.savez_compressed(cache_path, **save_data)
                    except Exception as e:
                        logger.error(f"Error saving tokens for video {video_id}: {str(e)}")

                self.last_save_time = current_time
                logger.info("Cache save completed")

            except Exception as e:
                logger.error(f"Error saving cache: {str(e)}")

    def save_reference_features(self, video_id: int):
        """Save reference features for a video"""
        try:
            features = self.reference_features.get(video_id)
            if features is not None:
                cache_path = self.get_cache_path(video_id, is_reference=True)
                save_data = {
                    f'feature_{i}': feature.astype(np.float32)
                    for i, feature in enumerate(features)
                }
                np.savez_compressed(cache_path, **save_data)
                logger.info(f"Saved reference features for video {video_id}")
        except Exception as e:
            logger.error(f"Error saving reference features for video {video_id}: {str(e)}")

    def clear_video(self, video_id: int):
        """Clear cached tokens for a video and remove from disk"""
        with self.lock:
            if video_id in self.video_tokens:
                del self.video_tokens[video_id]
            if video_id in self.generation_status:
                del self.generation_status[video_id]
            
            # Remove cache file
            cache_path = self.get_cache_path(video_id)
            if cache_path.exists():
                cache_path.unlink()
            
            # Save updated index
            self.save_cache(force=True)