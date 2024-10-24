from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
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
class TokenCache:
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.video_tokens: Dict[int, Dict[int, Dict]] = {}  # video_id -> {frame_id -> tokens}
        self.generation_status: Dict[int, Dict[int, bool]] = {}  # video_id -> {frame_id -> is_generated}
        self.lock = threading.Lock()

    def get_tokens(self, video_id: int, frame_id: int) -> Optional[Dict]:
        """Get tokens from cache"""
        if video_id in self.video_tokens:
            return self.video_tokens[video_id].get(frame_id)
        return None

    def set_tokens(self, video_id: int, frame_id: int, tokens: Dict):
        """Set tokens in cache with LRU eviction"""
        with self.lock:
            if video_id not in self.video_tokens:
                self.video_tokens[video_id] = {}
            
            self.video_tokens[video_id][frame_id] = tokens
            
            # LRU eviction if needed
            total_tokens = sum(len(frames) for frames in self.video_tokens.values())
            if total_tokens > self.max_size:
                # Remove oldest entries
                while total_tokens > self.max_size:
                    oldest_video = next(iter(self.video_tokens))
                    oldest_frames = self.video_tokens[oldest_video]
                    if oldest_frames:
                        oldest_frame = next(iter(oldest_frames))
                        del oldest_frames[oldest_frame]
                        total_tokens -= 1
                    if not oldest_frames:
                        del self.video_tokens[oldest_video]
        
    def is_generated(self, video_id: int, frame_id: int) -> bool:
        """Check if tokens have been generated"""
        return self.generation_status.get(video_id, {}).get(frame_id, False)

    def clear_video(self, video_id: int):
        """Clear cached tokens for a video"""
        with self.lock:
            if video_id in self.video_tokens:
                del self.video_tokens[video_id]
            if video_id in self.generation_status:
                del self.generation_status[video_id]

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

# Decorator for caching frame loads
@lru_cache(maxsize=128)
def load_frame(frame_path: str):
    """Load and preprocess a frame with caching"""
    import cv2
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise ValueError(f"Failed to load frame: {frame_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

class IMFServer:
    def __init__(self, checkpoint_path: str = "./checkpoints/checkpoint.pth"):
        self.app = FastAPI()
        self.setup_cors()
        self.setup_routes()
        self.load_model(checkpoint_path)
        self.active_connections: List[WebSocket] = []
        self.token_cache = TokenCache(max_size=1000)
        self.background_tasks = set()

        # Initialize dataset
        videos_root = "/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/"
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.dataset = VideoDataset(
            root_dir= videos_root ,
            transform=self.transform
        )
        
        self.videos_root = Path(videos_root)
        logger.info(f"Loaded {len(self.dataset)} videos from {videos_root}")


        # Start background task to prepare all videos
        self.startup_complete = asyncio.Event()
        asyncio.create_task(self.prepare_all_videos())


          # Register startup event handler
        @self.app.on_event("startup")
        async def startup_event():
            logger.info("Starting server initialization...")
            await self.prepare_all_videos()
            logger.info("Server initialization complete")

    async def prepare_all_videos(self):
        """Prepare all videos on server startup"""
        try:
            logger.info("Starting preparation of all videos...")
            tasks = []
            
            # Create tasks for each video
            for video_id in range(len(self.dataset)):
                if not self.token_cache.is_generated(video_id, 0):  # Check if video is already processed
                    task = asyncio.create_task(self.generate_tokens_for_video(video_id))
                    tasks.append(task)
                    logger.info(f"Queued video {video_id} for processing")
            
            if tasks:
                # Process videos in parallel with a limit
                chunk_size = 1  # Process 3 videos at a time
                for i in range(0, len(tasks), chunk_size):
                    chunk = tasks[i:i + chunk_size]
                    await asyncio.gather(*chunk)
                    logger.info(f"Completed processing chunk {i//chunk_size + 1} of {(len(tasks) + chunk_size - 1)//chunk_size}")
            
            logger.info("All videos prepared successfully")
            
        except Exception as e:
            logger.error(f"Error preparing videos: {str(e)}")
        finally:
            # Signal that startup processing is complete
            self.startup_complete.set()

    async def generate_tokens_for_video(self, video_id: int):
        """Generate tokens for all frames in a video"""
        try:
            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            
            logger.info(f"Starting token generation for video {video_id} with {len(frames)} frames")
            
            # Initialize progress tracking
            if video_id not in self.token_cache.generation_status:
                self.token_cache.generation_status[video_id] = {}

            # Process frames in chunks with error handling for each frame
            chunk_size = 10
            for i in range(0, len(frames), chunk_size):
                chunk_frames = frames[i:i + chunk_size]
                tasks = []
                
                for frame_idx, frame_path in enumerate(chunk_frames, start=i):
                    if not self.token_cache.is_generated(video_id, frame_idx):
                        task = asyncio.create_task(self.generate_frame_tokens(video_id, frame_idx, frame_path))
                        tasks.append(task)
                
                try:
                    # Process chunk with individual error handling
                    for task in tasks:
                        try:
                            await task
                        except Exception as e:
                            logger.error(f"Failed to process frame in video {video_id}: {str(e)}")
                            continue
                    
                    processed_count = len([k for k in self.token_cache.video_tokens.get(video_id, {}).keys() if k < i + chunk_size])
                    logger.info(f"Generated tokens for frames {i} to {min(i + chunk_size, len(frames))} ({processed_count} frames processed)")
                
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk in video {video_id}: {str(chunk_error)}")
                    continue

            # Log completion status
            total_processed = len(self.token_cache.video_tokens.get(video_id, {}))
            total_frames = len(frames)
            logger.info(f"Completed token generation for video {video_id}. Processed {total_processed}/{total_frames} frames.")

        except Exception as e:
            logger.error(f"Error generating tokens for video {video_id}: {str(e)}")
            raise

    async def process_video_frames(self, video_id: int, current_frame: int, reference_frame: int) -> Dict:
        """Process frames using dataset loading"""
        try:
            if video_id < 0 or video_id >= len(self.dataset):
                raise ValueError(f"Invalid video_id: {video_id}")

            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            frame_count = len(frames)
            
            if current_frame >= frame_count or reference_frame >= frame_count:
                raise ValueError(f"Frame index out of range. Video {video_id} has {frame_count} frames")

            # Get cached tokens
            current_tokens = self.token_cache.get_tokens(video_id, current_frame)
            reference_tokens = self.token_cache.get_tokens(video_id, reference_frame)

            if current_tokens and reference_tokens:
                features_data = {
                    'reference_features': reference_tokens['features'],
                    'reference_token': reference_tokens['tokens'],
                    'current_token': current_tokens['tokens']
                }
                logger.info("Using cached tokens")
            else:
                # Load frames using dataset
                current_frame_path = frames[current_frame]
                reference_frame_path = frames[reference_frame]
                
                current_frame_tensor = self.dataset._load_and_transform_frame(current_frame_path).unsqueeze(0)
                reference_frame_tensor = self.dataset._load_and_transform_frame(reference_frame_path).unsqueeze(0)

                # Extract features and tokens
                with torch.no_grad():
                    f_r, t_r, t_c = self.model.tokens(current_frame_tensor, reference_frame_tensor)

                features_data = {
                    'reference_features': [f.cpu().numpy().reshape(1, *f.shape[1:]).tolist() for f in f_r],
                    'reference_token': t_r.cpu().numpy().reshape(1, -1).tolist(),
                    'current_token': t_c.cpu().numpy().reshape(1, -1).tolist()
                }

                # Cache tokens
                if not current_tokens:
                    self.token_cache.set_tokens(video_id, current_frame, {
                        'features': features_data['reference_features'],
                        'tokens': features_data['current_token']
                    })
                if not reference_tokens:
                    self.token_cache.set_tokens(video_id, reference_frame, {
                        'features': features_data['reference_features'],
                        'tokens': features_data['reference_token']
                    })

            # Prepare response with cached frames data
            video_cached_tokens = self.token_cache.video_tokens.get(video_id, {})
            cached_frame_indices = sorted(list(video_cached_tokens.keys()))
            
            return {
                "type": "frame_features",
                "video_id": video_id,
                "current_frame": current_frame,
                "reference_frame": reference_frame,
                "features": features_data,
                "cached_frames": {
                    str(idx): video_cached_tokens[idx] 
                    for idx in cached_frame_indices
                },
                "metadata": {
                    "frame_count": frame_count,
                    "cached": bool(current_tokens and reference_tokens),
                    "total_cached_frames": len(video_cached_tokens),
                    "processing_progress": self.token_cache.get_generation_progress(video_id)
                }
            }

        except Exception as e:
            logger.error(f"Error in process_video_frames: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "message": str(e)
            }
    
    async def generate_frame_tokens(self, video_id: int, frame_idx: int, frame_path: str):
        """Generate tokens for a single frame using the dataset's frame loading"""
        try:
            # Load frame using dataset's method
            frame_data = self.dataset._load_and_transform_frame(frame_path)
            if frame_data is None:
                raise ValueError(f"Failed to load frame {frame_idx}")

            # Add batch dimension if needed
            frame_tensor = frame_data.unsqueeze(0) if frame_data.dim() == 3 else frame_data
            
            # Use the same frame as both current and reference for token generation
            with torch.no_grad():
                features, tokens_ref, tokens_current = self.model.tokens(
                    frame_tensor,  # current frame
                    frame_tensor   # use same frame as reference
                )

            # Cache tokens and features
            self.token_cache.set_tokens(video_id, frame_idx, {
                'features': [f.cpu().numpy() for f in features],
                'tokens': tokens_current.cpu().numpy()
            })
                
            # Update generation status
            self.token_cache.generation_status[video_id][frame_idx] = True

            logger.info(f"Successfully generated tokens for frame {frame_idx} of video {video_id}")

        except Exception as e:
            logger.error(f"Error generating tokens for frame {frame_idx} of video {video_id}: {str(e)}")
            raise

        
    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://192.168.1.108:3001"],  # Specific origin instead of wildcard
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["*"]
        )

    def load_model(self, checkpoint_path: str):
        # Initialize model
        self.model = IMFModel()
        self.model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def setup_routes(self):

        @self.app.get("/videos/{video_id}/generation-status")
        async def get_generation_status(self,video_id: int):
            """Get token generation progress"""
            try:
                if video_id < 0 or video_id >= len(self.dataset):
                    raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

                progress = self.token_cache.get_generation_progress(video_id)
                return {
                    "video_id": video_id,
                    "progress": progress
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/videos")
        async def list_videos():
            """List all available videos"""
            try:
                videos = []
                for idx in range(len(self.dataset)):
                    video_folder = self.dataset.video_folders[idx]
                    video_name = Path(video_folder).name
                    num_frames = self.dataset.video_frames[idx]
                    videos.append({
                        "id": idx,
                        "name": video_name,
                        "frame_count": num_frames
                    })
                return {"videos": videos}
            except Exception as e:
                logger.error(f"Error listing videos: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/videos/{video_id}/prepare")
        async def prepare_video(video_id: int):
            """Start token generation for a video"""
            try:
                if video_id < 0 or video_id >= len(self.dataset):
                    raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

                # Start token generation in background
                task = asyncio.create_task(self.generate_tokens_for_video(video_id))
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)

                return {
                    "status": "started",
                    "message": f"Token generation started for video {video_id}"
                }

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            
        @self.app.get("/videos/{video_id}/frames/{frame_id}")
        async def get_frame(video_id: int, frame_id: int):
            """Get a specific frame from a video"""
            try:
                # Validate video_id
                if video_id < 0 or video_id >= len(self.dataset):
                    logger.warning(f"Invalid video_id requested: {video_id}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Video {video_id} not found. Available videos: 0-{len(self.dataset)-1}"
                    )

                # Get frame count for this video
                video_folder = self.dataset.video_folders[video_id]
                frames = sorted([f for f in Path(video_folder).glob("*.png")])
                frame_count = len(frames)
                
                # Validate frame_id
                if frame_id < 0 or frame_id >= frame_count:
                    logger.warning(f"Invalid frame_id requested: {frame_id} for video {video_id} with {frame_count} frames")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Frame {frame_id} not found in video {video_id}. Available frames: 0-{frame_count-1}"
                    )
                
                frame_path = frames[frame_id]
                if not frame_path.exists():
                    logger.error(f"Frame file missing: {frame_path}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Frame file not found: {frame_path}"
                    )

                # Load and convert frame
                try:
                    img = Image.open(frame_path)
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    return JSONResponse({
                        "frame": base64.b64encode(img_bytes.getvalue()).decode('utf-8'),
                        "metadata": {
                            "frame_number": frame_id,
                            "total_frames": frame_count,
                            "video_id": video_id
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing frame: {str(e)}"
                    )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Unexpected error in get_frame: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error: {str(e)}"
                )

        # Add an endpoint to get video metadata
        @self.app.get("/videos/{video_id}/metadata")
        async def get_video_metadata(video_id: int):
            """Get metadata for a specific video"""
            try:
                if video_id < 0 or video_id >= len(self.dataset):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Video {video_id} not found"
                    )

                video_folder = self.dataset.video_folders[video_id]
                frames = sorted([f for f in Path(video_folder).glob("*.png")])
                
                return {
                    "video_id": video_id,
                    "frame_count": len(frames),
                    "name": Path(video_folder).name
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=str(e)
                )
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket_connection(websocket)

        @self.app.post("/upload-video")
        async def upload_video(file: UploadFile = File(...)):
            return await self.handle_video_upload(file)

    async def handle_websocket_connection(self, websocket: WebSocket):
        logger.info("New WebSocket connection attempt...")
        try:
            await websocket.accept()
            logger.info("WebSocket connection accepted")
            self.active_connections.append(websocket)
            
            try:
                while True:
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        break

                    data = await websocket.receive_json()
                    logger.info(f"Received message: {data}")
                    
                    message_type = data.get("type")
                    if message_type == "init":
                        try:
                            # Handle initialization
                            payload = data.get("payload", {})
                            buffer_size = payload.get("bufferSize", 30)
                            fps = payload.get("fps", 30)
                            
                            # Send acknowledgment
                            await websocket.send_json({
                                "type": "init_response",
                                "status": "success",
                                "config": {
                                    "bufferSize": buffer_size,
                                    "fps": fps,
                                    "maxFrames": 300  # Or any server-side limit
                                }
                            })
                            logger.info(f"Client initialized with buffer_size={buffer_size}, fps={fps}")
                            
                        except Exception as e:
                            logger.error(f"Error during initialization: {str(e)}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Initialization failed: {str(e)}"
                            })

                    elif message_type == "process_frames":
                        try:
                            # Extract fields from payload
                            payload = data.get("payload", {})
                            video_id = payload.get("video_id")
                            current_frame = payload.get("current_frame")
                            reference_frame = payload.get("reference_frame")

                            # Validate required fields
                            if any(x is None for x in [video_id, current_frame, reference_frame]):
                                raise ValueError("Missing required fields in payload")

                            # Process frames
                            response = await self.process_video_frames(
                                video_id=video_id,
                                current_frame=current_frame,
                                reference_frame=reference_frame
                            )
                            
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_json(response)
                                
                        except ValueError as ve:
                            logger.error(f"Invalid message payload: {str(ve)}")
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Invalid message payload: {str(ve)}"
                                })
                        except Exception as e:
                            logger.error(f"Error processing frames: {str(e)}")
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": str(e)
                                })
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Unknown message type: {message_type}"
                            })
                    
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected normally")
            except Exception as e:
                logger.error(f"Error in WebSocket connection: {str(e)}")
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
                    
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {str(e)}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def process_video_frames(self, video_id: int, current_frame: int, reference_frame: int) -> Dict:
        """Process a pair of frames using cached tokens and return all cached frames for the specific video"""
        try:
            if video_id < 0 or video_id >= len(self.dataset):
                raise ValueError(f"Invalid video_id: {video_id}")

            # Get video folder and validate frame indices
            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            frame_count = len(frames)
            
            if current_frame >= frame_count or reference_frame >= frame_count:
                raise ValueError(f"Frame index out of range. Video {video_id} has {frame_count} frames")

            # Get all cached tokens for this specific video
            video_cached_tokens = self.token_cache.video_tokens.get(video_id, {})
            logger.info(f"Found {len(video_cached_tokens)} cached frames for video {video_id}")

            # Check if current and reference frames are cached
            current_tokens = self.token_cache.get_tokens(video_id, current_frame)
            reference_tokens = self.token_cache.get_tokens(video_id, reference_frame)

            if current_tokens and reference_tokens:
                # Use cached tokens for requested frames
                features_data = {
                    'reference_features': reference_tokens['features'],
                    'reference_token': reference_tokens['tokens'],
                    'current_token': current_tokens['tokens']
                }
                logger.info("Using cached tokens for requested frames")
            else:
                # Generate tokens on-the-fly for requested frames
                logger.info("Generating tokens on-the-fly for requested frames")
                
                # Load frames using PIL and transform
                current_frame_path = frames[current_frame]
                reference_frame_path = frames[reference_frame]
                
                def load_and_transform_frame(frame_path):
                    img = Image.open(frame_path).convert('RGB')
                    if self.dataset.transform:
                        return self.dataset.transform(img)
                    else:
                        transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                        ])
                        return transform(img)
                
                current_frame_tensor = load_and_transform_frame(current_frame_path)
                reference_frame_tensor = load_and_transform_frame(reference_frame_path)

                # Extract features and tokens
                with torch.no_grad():
                    f_r, t_r, t_c = self.model.tokens(
                        current_frame_tensor.unsqueeze(0),
                        reference_frame_tensor.unsqueeze(0)
                    )

                # Convert to serializable format
                features_data = {
                    'reference_features': [
                        f.cpu().numpy().reshape(1, *f.shape[1:]).tolist() 
                        for f in f_r
                    ],
                    'reference_token': t_r.cpu().numpy().reshape(1, -1).tolist(),
                    'current_token': t_c.cpu().numpy().reshape(1, -1).tolist()
                }

                # Cache the newly generated tokens
                if not current_tokens:
                    self.token_cache.set_tokens(video_id, current_frame, {
                        'features': features_data['reference_features'],
                        'tokens': features_data['current_token']
                    })
                if not reference_tokens:
                    self.token_cache.set_tokens(video_id, reference_frame, {
                        'features': features_data['reference_features'],
                        'tokens': features_data['reference_token']
                    })

                # Update video_cached_tokens with newly generated tokens
                video_cached_tokens = self.token_cache.video_tokens.get(video_id, {})
                logger.info(f"Generated and cached tokens for frames {current_frame} and {reference_frame}")

            # Get ordered list of cached frame indices
            cached_frame_indices = sorted(list(video_cached_tokens.keys()))

            # Get a continuous range of cached frames if they exist
            cached_frames_ranges = []
            range_start = None
            for i in range(len(cached_frame_indices)):
                if i == 0:
                    range_start = cached_frame_indices[i]
                elif cached_frame_indices[i] != cached_frame_indices[i-1] + 1:
                    cached_frames_ranges.append((range_start, cached_frame_indices[i-1]))
                    range_start = cached_frame_indices[i]
            if range_start is not None:
                cached_frames_ranges.append((range_start, cached_frame_indices[-1]))

            # Prepare continuous ranges of cached frames data
            cached_frames_data = {}
            for start, end in cached_frames_ranges:
                for frame_id in range(start, end + 1):
                    if frame_id in video_cached_tokens:
                        cached_frames_data[str(frame_id)] = {
                            'features': video_cached_tokens[frame_id]['features'],
                            'tokens': video_cached_tokens[frame_id]['tokens']
                        }

            return {
                "type": "frame_features",
                "video_id": video_id,
                "current_frame": current_frame,
                "reference_frame": reference_frame,
                "features": features_data,
                "cached_frames": cached_frames_data,
                "metadata": {
                    "frame_count": frame_count,
                    "cached": bool(current_tokens and reference_tokens),
                    "total_cached_frames": len(video_cached_tokens),
                    "cached_ranges": cached_frames_ranges,
                    "processing_progress": self.token_cache.get_generation_progress(video_id)
                }
            }

        except Exception as e:
            logger.error(f"Error in process_video_frames: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "message": str(e)
            }


    async def handle_video_upload(self, file: UploadFile):
        # Save video temporarily
        video_path = f"temp_{file.filename}"
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process video and extract frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Store frames in memory (or database for production)
        video_id = str(len(frames))  # Simple ID generation
        self.stored_frames = frames

        return {"video_id": video_id, "total_frames": len(frames)}

    @torch.no_grad()
    def extract_features(self, current_frame: np.ndarray, reference_frame: np.ndarray):
        logger.info("Extracting features from frames")
        
        def preprocess_frame(frame):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            frame = frame.unsqueeze(0)
            return frame

        x_current = preprocess_frame(current_frame)
        x_reference = preprocess_frame(reference_frame)

        # Extract features and tokens
        f_r, t_r, t_c = self.model.tokens(x_current, x_reference)

        # Log shapes for debugging
        logger.info(f"Current token shape: {t_c.shape}")
        logger.info(f"Reference token shape: {t_r.shape}")
        for i, f in enumerate(f_r):
            logger.info(f"Reference feature {i} shape: {f.shape}")

        # Convert to serializable format ensuring correct dimensions
        features_data = {
            "reference_features": [
                f.cpu().numpy().reshape(1, *f.shape[1:]).tolist() 
                for f in f_r
            ],
            "reference_token": t_r.cpu().numpy().reshape(1, -1).tolist(),
            "current_token": t_c.cpu().numpy().reshape(1, -1).tolist()
        }

        # Log the processed shapes
        logger.info("Processed shapes:")
        logger.info(f"Current token: {np.array(features_data['current_token']).shape}")
        logger.info(f"Reference token: {np.array(features_data['reference_token']).shape}")
        for i, f in enumerate(features_data['reference_features']):
            logger.info(f"Reference feature {i}: {np.array(f).shape}")

        logger.info("Features extracted successfully")
        return features_data

    async def process_frame_request(self, data: Dict[str, Any]):
        logger.info(f"Processing frame request: {data}")
        frame_idx = data["frame_index"]
        ref_frame_idx = data.get("reference_frame_index", 0)

        try:
            # Get frames
            current_frame = self.stored_frames[frame_idx]
            reference_frame = self.stored_frames[ref_frame_idx]

            # Extract features
            features_data = self.extract_features(current_frame, reference_frame)

            # Validate feature shapes
            reference_features = features_data["reference_features"]
            expected_shapes = [
                (1, 128, 64, 64),
                (1, 256, 32, 32),
                (1, 512, 16, 16),
                (1, 512, 8, 8)
            ]

            logger.info("Validating feature shapes:")
            for i, (feat, expected) in enumerate(zip(reference_features, expected_shapes)):
                feat_shape = np.array(feat).shape
                logger.info(f"Feature {i}: Got {feat_shape}, Expected {expected}")
                if feat_shape != expected:
                    raise ValueError(f"Feature {i} has wrong shape: {feat_shape} vs {expected}")

            logger.info(f"Reference token shape: {np.array(features_data['reference_token']).shape}")
            logger.info(f"Current token shape: {np.array(features_data['current_token']).shape}")

            return {
                "type": "frame_features",
                "frame_index": frame_idx,
                "reference_frame_index": ref_frame_idx,
                "features": features_data
            }
        except Exception as e:
            logger.error(f"Error processing frame request: {str(e)}")
            return {
                "type": "error",
                "message": str(e)
            }
        

    def run(self, 
            host: str = "0.0.0.0", 
            port: int = 8000,
            ssl_certfile: str = "192.168.1.108.pem",
            ssl_keyfile: str = "192.168.1.108-key.pem"):
        
        logger.info(f"Starting server on {host}:{port} with SSL")
        
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)
            
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile,
                log_level="debug",
                ws_ping_interval=30.0,
                ws_ping_timeout=10.0,
                timeout_keep_alive=30,
            )
            
            server = uvicorn.Server(config)
            server.run()
            
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            raise

if __name__ == "__main__":
    # Create an async function to run the server
    async def main():
        server = IMFServer()
        config = uvicorn.Config(
            app=server.app,
            host="0.0.0.0",
            port=8000,
            ssl_certfile="192.168.1.108.pem",
            ssl_keyfile="192.168.1.108-key.pem",
            log_level="debug",
            ws_ping_interval=30.0,
            ws_ping_timeout=10.0,
            timeout_keep_alive=30,
        )
        server_instance = uvicorn.Server(config)
        await server_instance.serve()

    # Run the async main function
    asyncio.run(main())