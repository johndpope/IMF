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
from imf_server_cache import TokenCache
from moviepy.editor import VideoFileClip
import numpy as np
import tempfile
import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack,RTCIceCandidate
from av import AudioFrame
import numpy as np

from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self, frames_queue):
        super().__init__()
        self.frames_queue = frames_queue
        self.sample_rate = 48000
        self.pts = 0
        
    async def recv(self):
        frame_data = await self.frames_queue.get()
        frame = AudioFrame.from_ndarray(
            frame_data,
            format='s16',
            layout='mono'
        )
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self.pts += frame.samples
        return frame


class MP4Handler:
    def __init__(self, video_path: str):
        self.video = VideoFileClip(video_path)
        self.audio = self.video.audio
        
    def extract_audio_chunk(self, start_time: float, duration: float) -> np.ndarray:
        """Extract audio chunk as numpy array"""
        return self.audio.subclip(start_time, start_time + duration).to_soundarray()
        
    def get_audio_params(self) -> dict:
        return {
            'fps': self.audio.fps,
            'duration': self.audio.duration,
            'nchannels': self.audio.nchannels
        }


ALLOWED_ORIGINS = [
            "https://192.168.1.108:3001",  # Your frontend origin
            "wss://192.168.1.108:8000"     # WebSocket server origin
] 
class IMFServer:
    def __init__(self, checkpoint_path: str = "./checkpoints/checkpoint.pth", cache_dir: str = "./token_cache"):
        self.app = FastAPI()
        self.setup_cors()
        self.setup_routes()
        self.load_model(checkpoint_path)
        self.active_connections: List[WebSocket] = []
        
        # Initialize persistent cache
        self.token_cache = TokenCache(max_size=1000, cache_dir=cache_dir)
        self.background_tasks = set()

        # Initialize dataset
        videos_root = "/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/"
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.dataset = VideoDataset(
            root_dir=videos_root,
            transform=self.transform
        )
        
        self.videos_root = Path(videos_root)
        logger.info(f"Loaded {len(self.dataset)} videos from {videos_root}")

        # Start background task to prepare unprocessed videos
        self.startup_complete = asyncio.Event()
        self.startup_complete.set()
        # asyncio.create_task(self.prepare_all_videos())

        # Add connection state tracking
        self.ice_gathering_state = {}
        self.ice_connection_states = {}
        self.connected_peers = set()
        self.data_channels = {}  # Store data channels by peer_id
        self.pending_messages = {}  # Store messages that need to be sent once channel is open


    async def generate_reference_features(self, video_id: int):
        """Generate and cache reference features for a video"""
        try:
            # Get first frame path
            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            if not frames:
                raise ValueError(f"No frames found for video {video_id}")
            
            reference_frame_path = frames[0]  # Use first frame as reference
            
            # Load and transform frame
            img = Image.open(reference_frame_path).convert('RGB')
            frame_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
            
            # Generate features
            with torch.no_grad():
                features = self.model.dense_feature_encoder(frame_tensor)
            
            # Cache reference features
            self.token_cache.set_reference_features(video_id, [
                f.cpu().numpy() for f in features
            ])
            
            logger.info(f"Generated and cached reference features for video {video_id}")
            return features
            
        except Exception as e:
            logger.error(f"🔥 🔥 Error generating reference features for video {video_id}: {str(e)}")
            raise

    async def generate_frame_tokens(self, video_id: int, frame_idx: int, frame_path: str):
        """Generate tokens for a single frame"""
        try:
            # Load and preprocess frame
            img = Image.open(frame_path).convert('RGB')
            frame_tensor = self.transform(img).unsqueeze(0)
            
            # Use the same frame as both current and reference for initial token generation
            with torch.no_grad():
                features, _, tokens = self.model.tokens(frame_tensor, frame_tensor)

            # Cache tokens and features
            if frame_idx == 0:  # Only store features for reference frame
                self.token_cache.set_reference_features(video_id, [
                    f.cpu().numpy() for f in features
                ])
            
            # Store frame tokens
            self.token_cache.set_tokens(video_id, frame_idx, tokens.cpu().numpy())
            
            # Update generation status
            if video_id not in self.token_cache.generation_status:
                self.token_cache.generation_status[video_id] = {}
            self.token_cache.generation_status[video_id][frame_idx] = True

            logger.info(f"Generated tokens for frame {frame_idx} of video {video_id}")

        except Exception as e:
            logger.error(f"🔥 🔥 Error generating tokens for frame {frame_idx} of video {video_id}: {str(e)}")
            raise


    async def prepare_all_videos(self):
        """Prepare all videos sequentially"""
        try:
            logger.info("Starting sequential video preparation...")
            for video_id in range(len(self.dataset)):
                try:
                    video_folder = self.dataset.video_folders[video_id]
                    frames = sorted([f for f in Path(video_folder).glob("*.png")])
                    
                    # Process reference frame first
                    if not self.token_cache.get_reference_features(video_id):
                        await self.generate_frame_tokens(video_id, 0, frames[0])
                        logger.info(f"Generated reference features for video {video_id}")
                    
                    # Then process remaining frames
                    for frame_idx, frame_path in enumerate(frames[1:], start=1):
                        if not self.token_cache.is_generated(video_id, frame_idx):
                            await self.generate_frame_tokens(video_id, frame_idx, frame_path)
                            logger.info(f"Generated tokens for frame {frame_idx} of video {video_id}")
                    
                    # Save cache after each video
                    self.token_cache.save_cache(force=True)
                    logger.info(f"Completed processing video {video_id}")
                    
                except Exception as e:
                    logger.error(f"🔥 🔥 Failed to process video {video_id}: {str(e)}")
                    continue

            logger.info("All videos prepared successfully")
            
        except Exception as e:
            logger.error(f"🔥 🔥 Error during video preparation: {str(e)}")
        finally:
            self.startup_complete.set()

    async def process_video_frames(self, video_id: int, current_frame: int, reference_frame: int) -> Dict:
        """Process frames using cached tokens"""
        try:
            if video_id < 0 or video_id >= len(self.dataset):
                raise ValueError(f"Invalid video_id: {video_id}")

            # Get video folder and validate frame indices
            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            frame_count = len(frames)
            
            if current_frame >= frame_count or reference_frame >= frame_count:
                raise ValueError(f"Frame index out of range. Video {video_id} has {frame_count} frames")

            # Get cached tokens
            video_cached_tokens = self.token_cache.video_tokens.get(video_id, {})
            logger.info(f"Found {len(video_cached_tokens)} cached frames for video {video_id}")

            # Get reference features (should be from frame 0)
            reference_features = None
            if 0 in video_cached_tokens:
                ref_token_data = video_cached_tokens[0]
                reference_features = ref_token_data.get('features', None)

            # Check if current frame has cached tokens
            current_tokens = None
            if current_frame in video_cached_tokens:
                current_tokens = video_cached_tokens[current_frame].get('tokens', None)

            if reference_features is None or current_tokens is None:
                # Generate tokens on-the-fly for requested frames
                logger.info("Generating tokens on-the-fly for requested frames")
                
                # Load frames using PIL and transform
                current_frame_path = frames[current_frame]
                reference_frame_path = frames[0]  # Always use first frame as reference
                
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

                    # Always store reference features from first frame
                    if reference_features is None:
                        reference_features = [f.cpu().numpy() for f in f_r]
                        self.token_cache.set_tokens(video_id, 0, {
                            'features': reference_features,
                            'tokens': t_r.cpu().numpy()
                        })

                    # Store current frame tokens
                    current_tokens = t_c.cpu().numpy()
                    self.token_cache.set_tokens(video_id, current_frame, {
                        'tokens': current_tokens
                    })

                    logger.info(f"Generated and cached tokens for frames {current_frame}")

            # Prepare response
            features_data = {
                'reference_features': [f.tolist() if isinstance(f, np.ndarray) else f for f in reference_features],
                'current_token': current_tokens.tolist() if isinstance(current_tokens, np.ndarray) else current_tokens
            }

            # Get ordered list of cached frames
            cached_frame_indices = sorted(list(video_cached_tokens.keys()))
            
            # Get continuous ranges
            cached_frames_ranges = []
            if cached_frame_indices:
                range_start = cached_frame_indices[0]
                prev = cached_frame_indices[0]
                
                for idx in cached_frame_indices[1:]:
                    if idx != prev + 1:
                        cached_frames_ranges.append((range_start, prev))
                        range_start = idx
                    prev = idx
                cached_frames_ranges.append((range_start, prev))

            return {
                "type": "frame_features",
                "video_id": video_id,
                "current_frame": current_frame,
                "reference_frame": reference_frame,
                "features": features_data,
                "metadata": {
                    "frame_count": frame_count,
                    "cached": bool(current_tokens is not None),
                    "total_cached_frames": len(video_cached_tokens),
                    "cached_ranges": cached_frames_ranges,
                    "processing_progress": self.token_cache.get_generation_progress(video_id)
                }
            }

        except Exception as e:
            logger.error(f"🔥 🔥 Error in process_video_frames: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "message": str(e)
            }

        
    async def get_media_chunk(self, video_id: int, chunk_index: int):
        try:
            video_path = self.dataset.video_folders[video_id]
            mp4_path = f"{video_path}/video.mp4"
            
            # Get handler (could be cached)
            handler = MP4Handler(mp4_path)
            
            # Calculate time for chunk
            chunk_duration = 1/24.0  # For 24fps
            start_time = chunk_index * chunk_duration
            
            # Get audio data
            audio_data = handler.extract_audio_chunk(start_time, chunk_duration)
            
            # Get frame token
            token_data = await self.get_frame_token(video_id, chunk_index)
            
            return {
                'timestamp': start_time * 1000,  # Convert to ms
                'audio': audio_data.tobytes(),
                'token': token_data,
                'duration': chunk_duration * 1000
            }
            
        except Exception as e:
            logger.error(f"🔥 🔥 Error getting media chunk: {e}")
            raise
    
   
    def setup_cors(self):
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,  # Pass the list of allowed origins
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


    async def handle_init_connection(self, websocket: WebSocket, message: Dict, peer_id: str) -> None:
        """Handle initial WebRTC connection setup with proper state tracking"""
        try:
            payload = message.get("payload", {})
            fps = payload.get("fps", 24)
            
            # Create proper RTCConfiguration object
            config = RTCConfiguration(
                iceServers=[
                    RTCIceServer(urls=["stun:stun.l.google.com:19302"])
                ]
            )
            
            # Create new peer connection with proper configuration
            pc = RTCPeerConnection(configuration=config)
            
            # Set up audio queue and track
            audio_queue = asyncio.Queue()
            pc.addTrack(AudioStreamTrack(audio_queue))
            
            # Set up data channel handler
            @pc.on("datachannel")
            def on_datachannel(channel):
                logger.info(f"Data channel established for peer {peer_id}: {channel.label}")
                
                @channel.on("open")
                def on_open():
                    logger.info(f"Data channel opened for peer {peer_id}")
                    self.data_channels[peer_id] = channel
                    
                    # Send any pending messages
                    if peer_id in self.pending_messages:
                        for msg in self.pending_messages[peer_id]:
                            try:
                                channel.send(json.dumps(msg))
                            except Exception as e:
                                logger.error(f"Error sending pending message: {e}")
                        del self.pending_messages[peer_id]

                @channel.on("message")
                async def on_message(msg):
                    try:
                        data = json.loads(msg)
                        if data["type"] == "start_stream":
                            await self.start_video_stream(channel, data["videoId"], peer_id)
                    except Exception as e:
                        logger.error(f"🔥 Error handling data channel message: {e}")

                @channel.on("close")
                def on_close():
                    logger.info(f"Data channel closed for peer {peer_id}")
                    if peer_id in self.data_channels:
                        del self.data_channels[peer_id]

            # Set up connection state monitoring
            @pc.on("connectionstatechange")
            async def on_connection_state_change():
                logger.info(f"Connection state changed to: {pc.connectionState} for peer {peer_id}")
                if pc.connectionState == "failed":
                    await self.handle_connection_failure(peer_id)
                elif pc.connectionState == "connected":
                    if peer_id not in self.data_channels:
                        data_channel = pc.createDataChannel("frames")
                        self.setup_data_channel(data_channel, peer_id)

            @pc.on("iceconnectionstatechange")
            async def on_ice_connection_state_change():
                logger.info(f"ICE connection state changed to: {pc.iceConnectionState} for peer {peer_id}")

            # Send immediate response to client
            logger.info(f"Sending init response to peer {peer_id}")
            await websocket.send_json({
                "type": "init_response",
                "status": "success",
                "payload": {
                    "rtcConfig": {
                        "iceServers": [
                            {"urls": ["stun:stun.l.google.com:19302"]}
                        ]
                    },
                    "fps": fps,
                    "maxFrames": 300
                }
            })

            return {
                "peer_connection": pc,
                "audio_queue": audio_queue,
                "fps": fps
            }

        except Exception as e:
            logger.error(f"🔥 Error in init connection for peer {peer_id}: {str(e)}", exc_info=True)
            await websocket.send_json({
                "type": "error",
                "payload": {
                    "message": f"Init failed: {str(e)}"
                }
            })
            raise



    def setup_routes(self):

        @self.app.websocket("/rtc")
        async def websocket_rtc(websocket: WebSocket):
            pc = None
            connection_data = None
            peer_id = str(id(websocket))
            
            try:
                await websocket.accept()
                logger.info(f"WebRTC WebSocket connection accepted for peer {peer_id}")
                
                while True:
                    try:
                        message = await websocket.receive_json()
                        logger.info(f"Received WebRTC message from peer {peer_id}: {message}")
                        
                        if message["type"] == "init":
                            # Initialize connection
                            logger.info(f"Initializing connection for peer {peer_id}")
                            connection_data = await self.handle_init_connection(
                                websocket=websocket, 
                                message=message,
                                peer_id=peer_id
                            )
                            pc = connection_data["peer_connection"]
                            logger.info(f"Connection initialized for peer {peer_id}")
                            
                        elif message["type"] == "offer":
                            if not pc:
                                logger.error(f"No peer connection for peer {peer_id}")
                                raise ValueError("No peer connection established")
                                
                            logger.info(f"Processing offer from peer {peer_id}")
                            offer = RTCSessionDescription(
                                sdp=message["payload"]["sdp"]["sdp"],
                                type=message["payload"]["sdp"]["type"]
                            )
                            
                            await pc.setRemoteDescription(offer)
                            answer = await pc.createAnswer()
                            await pc.setLocalDescription(answer)
                            
                            await websocket.send_json({
                                "type": "answer",
                                "payload": {
                                    "sdp": {
                                        "type": answer.type,
                                        "sdp": answer.sdp
                                    }
                                }
                            })
                            logger.info(f"Sent answer to peer {peer_id}")
                        
                        elif message["type"] == "ice-candidate":
                            if not pc:
                                logger.warning(f"Received ICE candidate before peer connection setup for peer {peer_id}")
                                continue
                                
                            try:
                                candidate_data = message["payload"]["candidate"]
                                if candidate_data and candidate_data.get("candidate"):
                                    await self.handle_ice_candidate(pc, candidate_data, peer_id)
                            except Exception as e:
                                logger.error(f"🔥 Error handling ICE candidate for peer {peer_id}: {e}")
                                
                    except WebSocketDisconnect:
                        logger.info(f"WebRTC WebSocket disconnected normally for peer {peer_id}")
                        break
                        
                    except Exception as e:
                        logger.error(f"🔥 Error handling message for peer {peer_id}: {e}")
                        if pc and pc.connectionState != "closed":
                            await pc.close()
                        break
                        
            except Exception as e:
                logger.error(f"🔥 Error in WebRTC connection for peer {peer_id}: {e}")
                
            finally:
                # Cleanup
                if peer_id:
                    # Clean up data channels
                    if peer_id in self.data_channels:
                        channel = self.data_channels[peer_id]
                        channel.close()
                        del self.data_channels[peer_id]
                    
                    # Clean up pending messages
                    if peer_id in self.pending_messages:
                        del self.pending_messages[peer_id]
                    
                    # Clean up connection states
                    if hasattr(self, 'ice_gathering_state'):
                        self.ice_gathering_state.pop(peer_id, None)
                    if hasattr(self, 'ice_connection_states'):
                        self.ice_connection_states.pop(peer_id, None)
                    if hasattr(self, 'connected_peers'):
                        self.connected_peers.discard(peer_id)
                    
                if pc:
                    logger.info(f"Closing peer connection for peer {peer_id}")
                    await pc.close()
                logger.info(f"Cleaned up connection for peer {peer_id}")

        async def handle_connection_failure(self, peer_id: str):
            """Handle failed connections with cleanup"""
            logger.error(f"Connection failed for peer {peer_id}")
            try:
                if peer_id in self.data_channels:
                    channel = self.data_channels[peer_id]
                    channel.close()
                    del self.data_channels[peer_id]

                if peer_id in self.pending_messages:
                    del self.pending_messages[peer_id]
                    
            except Exception as e:
                logger.error(f"Error during connection failure cleanup: {e}")


        @self.app.get("/videos/{video_id}/reference")
        async def get_reference_data(video_id: int):
            """Get reference features and token for a video"""
            try:
                reference_data = await self.get_video_reference_data(video_id)
                return JSONResponse(reference_data)
            except ValueError as ve:
                raise HTTPException(status_code=404, detail=str(ve))
            except Exception as e:
                logger.error(f"🔥 🔥 Error serving reference data: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
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
                logger.error(f"🔥 🔥 Error listing videos: {str(e)}")
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
                    logger.error(f"🔥 🔥 Frame file missing: {frame_path}")
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
                    logger.error(f"🔥 🔥 Error processing frame: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing frame: {str(e)}"
                    )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"🔥 🔥 Unexpected error in get_frame: {str(e)}")
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



    
    async def handle_ice_candidate(self, pc: RTCPeerConnection, candidate_data: dict, peer_id: str = None) -> None:
        """
        Handle incoming ICE candidates with optimized prioritization and state tracking.
        
        Args:
            pc: RTCPeerConnection instance
            candidate_data: Dictionary containing ICE candidate information
            peer_id: Optional identifier for the peer connection
        """
        try:
            candidate_str = candidate_data.get('candidate', '')
            if not candidate_str:
                logger.warning("Empty candidate string received")
                return

            # Parse the candidate string
            parts = candidate_str.split()
            if len(parts) < 8:
                logger.error(f"Invalid candidate string format: {candidate_str}")
                return

            # Extract and parse parameters
            foundation = parts[0].split(':')[1]
            component = int(parts[1])
            protocol = parts[2]
            priority = int(parts[3])
            ip = parts[4]
            port = int(parts[5])
            candidate_type = parts[7]

            # Adjust priority based on candidate type
            if candidate_type == 'host':
                priority = max(priority, 2130706431)  # Prefer host candidates
            elif candidate_type == 'srflx':
                priority = min(priority, 1677729535)  # Lower priority for STUN
            elif candidate_type == 'relay':
                priority = min(priority, 16777215)    # Lowest priority for TURN

            # Create and configure ICE candidate
            ice_candidate = RTCIceCandidate(
                component=component,
                foundation=foundation,
                ip=ip,
                port=port,
                priority=priority,
                protocol=protocol,
                type=candidate_type,
                sdpMid=candidate_data.get('sdpMid'),
                sdpMLineIndex=candidate_data.get('sdpMLineIndex')
            )
            ice_candidate.candidate = candidate_str

            logger.info(f"Adding ICE candidate type {candidate_type} with priority {priority}")
            await pc.addIceCandidate(ice_candidate)
            
            # Update connection state tracking
            if peer_id:
                self.ice_gathering_state[peer_id] = pc.iceGatheringState
                self.ice_connection_states[peer_id] = pc.iceConnectionState
                
                if pc.iceConnectionState == "completed":
                    self.connected_peers.add(peer_id)
                    logger.info(f"Peer {peer_id} connection completed")

        except Exception as e:
            logger.error(f"🔥 Error handling ICE candidate: {str(e)}", exc_info=True)



                
    async def start_video_stream(self, channel, video_id: int, peer_id: str):
        """Handle video streaming with proper data channel communication"""
        try:
            if channel.readyState != "open":
                logger.warning(f"Data channel not open for peer {peer_id}, queueing start_stream message")
                if peer_id not in self.pending_messages:
                    self.pending_messages[peer_id] = []
                self.pending_messages[peer_id].append({
                    "type": "start_stream",
                    "videoId": video_id,
                    "fps": 24
                })
                return

            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            
            # Get reference features first
            reference_data = await self.process_video_frames(video_id, 0, 0)
            reference_features = reference_data["features"]["reference_features"]
            
            # Stream frames
            for frame_idx, frame_path in enumerate(frames):
                if channel.readyState != "open":
                    logger.info(f"Data channel closed for peer {peer_id}, stopping stream")
                    break
                    
                try:
                    # Process frame
                    frame_data = await self.process_video_frames(video_id, frame_idx, 0)
                    
                    # Prepare message
                    message = {
                        "type": "frame_token",
                        "frameIndex": frame_idx,
                        "token": frame_data["features"]["current_token"],
                        "timestamp": frame_idx * (1000 / 24)  # ms timestamp at 24fps
                    }
                    
                    # Send frame token using non-async send
                    try:
                        channel.send(json.dumps(message))
                        logger.info(f"Sent frame {frame_idx} to peer {peer_id}")
                    except Exception as send_error:
                        logger.error(f"Error sending frame {frame_idx} to peer {peer_id}: {send_error}")
                        if "closed" in str(send_error).lower():
                            logger.info("Data channel appears to be closed, stopping stream")
                            break
                    
                    # Control frame rate
                    await asyncio.sleep(1/24)
                    
                except Exception as e:
                    logger.error(f"🔥 Error processing frame {frame_idx} for peer {peer_id}: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"🔥 Error in video stream for peer {peer_id}: {e}")
        finally:
            logger.info(f"Video stream {video_id} complete for peer {peer_id}")

    def setup_data_channel(self, channel, peer_id: str):
        """Set up data channel with proper event handling and error recovery"""
        
        def send_safely(msg):
            """Helper function to safely send messages on the data channel"""
            try:
                if channel.readyState == "open":
                    if isinstance(msg, dict):
                        msg = json.dumps(msg)
                    channel.send(msg)
                    return True
                else:
                    logger.warning(f"Attempted to send message on closed channel for peer {peer_id}")
                    return False
            except Exception as e:
                logger.error(f"Error sending message on data channel for peer {peer_id}: {e}")
                return False

        @channel.on("open")
        def on_open():
            logger.info(f"Data channel opened for peer {peer_id}")
            self.data_channels[peer_id] = channel
            
            # Send any pending messages
            if peer_id in self.pending_messages:
                for msg in self.pending_messages[peer_id]:
                    if not send_safely(msg):
                        logger.error(f"Failed to send pending message for peer {peer_id}")
                        break
                del self.pending_messages[peer_id]

        @channel.on("close")
        def on_close():
            logger.info(f"Data channel closed for peer {peer_id}")
            if peer_id in self.data_channels:
                del self.data_channels[peer_id]

        @channel.on("error")
        def on_error(error):
            logger.error(f"Data channel error for peer {peer_id}: {error}")

        # Store the send_safely function with the channel for use elsewhere
        channel.send_safely = send_safely

    async def handle_connection_failure(self, peer_id: str):
        """Handle connection failures with cleanup and recovery"""
        logger.error(f"Connection failed for peer {peer_id}")
        try:
            # Clean up data channel
            if peer_id in self.data_channels:
                try:
                    channel = self.data_channels[peer_id]
                    channel.close()
                except Exception as e:
                    logger.error(f"Error closing data channel for peer {peer_id}: {e}")
                finally:
                    del self.data_channels[peer_id]

            # Clean up pending messages
            if peer_id in self.pending_messages:
                del self.pending_messages[peer_id]

            # Update connection states
            if peer_id in self.ice_connection_states:
                self.ice_connection_states[peer_id] = "failed"
            
            logger.info(f"Cleaned up after connection failure for peer {peer_id}")
            
        except Exception as e:
            logger.error(f"Error during connection failure cleanup: {e}")

    async def get_video_reference_data(self, video_id: int) -> Dict:
        """Get reference features and token for a video"""
        try:
            if video_id < 0 or video_id >= len(self.dataset):
                raise ValueError(f"Invalid video_id: {video_id}")

            # Get video folder
            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            
            if not frames:
                raise ValueError(f"No frames found for video {video_id}")
            
            # Check if we have cached data
            video_cached_tokens = self.token_cache.video_tokens.get(video_id, {})
            reference_features = None
            reference_token = None
            
            if 0 in video_cached_tokens:
                ref_data = video_cached_tokens[0]
                if isinstance(ref_data, dict):
                    reference_features = ref_data.get('features')
                    reference_token = ref_data.get('tokens')

            # Generate if not cached
            if reference_features is None or reference_token is None:
                logger.info(f"Generating reference data for video {video_id}")
                
                # Load and transform reference frame
                reference_frame_path = frames[0]
                img = Image.open(reference_frame_path).convert('RGB')
                frame_tensor = self.transform(img).unsqueeze(0)
                
                # Generate features and tokens
                with torch.no_grad():
                    features = self.model.dense_feature_encoder(frame_tensor)
                    _, reference_token, _ = self.model.tokens(frame_tensor, frame_tensor)
                    
                    reference_features = [f.cpu().numpy() for f in features]
                    reference_token = reference_token.cpu().numpy()
                    
                    # Cache the data
                    self.token_cache.set_tokens(video_id, 0, {
                        'features': reference_features,
                        'tokens': reference_token
                    })
                    
                    logger.info(f"Generated and cached reference data for video {video_id}")

            # Ensure proper shape and convert to list for JSON serialization
            reference_features_list = [
                f.tolist() if isinstance(f, np.ndarray) else f 
                for f in reference_features
            ]
            reference_token_list = reference_token.tolist() if isinstance(reference_token, np.ndarray) else reference_token

            # Verify shapes before returning
            expected_shapes = [
                [1, 128, 64, 64],
                [1, 256, 32, 32],
                [1, 512, 16, 16],
                [1, 512, 8, 8]
            ]

            for feat, expected in zip(reference_features_list, expected_shapes):
                actual = np.array(feat).shape
                if actual != tuple(expected):
                    raise ValueError(f"Feature shape mismatch. Expected {expected}, got {actual}")

            return {
                "video_id": video_id,
                "reference_features": reference_features_list,
                "reference_token": reference_token_list,
                "shapes": {
                    "features": expected_shapes,
                    "token": np.array(reference_token_list).shape
                }
            }
            
        except Exception as e:
            logger.error(f"🔥 🔥 Error getting reference data for video {video_id}: {str(e)}")
            raise
        
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
                            logger.error(f"🔥 🔥 Error during initialization: {str(e)}")
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
                            logger.error(f"🔥 🔥 Invalid message payload: {str(ve)}")
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Invalid message payload: {str(ve)}"
                                })
                        except Exception as e:
                            logger.error(f"🔥 🔥 Error processing frames: {str(e)}")
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
                logger.error(f"🔥 🔥 Error in WebSocket connection: {str(e)}")
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)
                    
        except Exception as e:
            logger.error(f"🔥 🔥 Failed to establish WebSocket connection: {str(e)}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def process_video_frames(self, video_id: int, current_frame: int, reference_frame: int) -> Dict:
        """Process frames using cached tokens"""
        try:
            if video_id < 0 or video_id >= len(self.dataset):
                raise ValueError(f"Invalid video_id: {video_id}")

            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            frame_count = len(frames)
            
            if current_frame >= frame_count or reference_frame >= frame_count:
                raise ValueError(f"Frame index out of range. Video {video_id} has {frame_count} frames")

            # Get cached data
            video_cached_tokens = self.token_cache.video_tokens.get(video_id, {})
            logger.info(f"Found {len(video_cached_tokens)} cached frames for video {video_id}")

            # Process current frame
            current_tokens = None
            reference_features = None
            need_processing = False

            # Try to get reference features from frame 0
            if 0 in video_cached_tokens:
                ref_data = video_cached_tokens[0]
                if isinstance(ref_data, dict) and 'features' in ref_data:
                    reference_features = ref_data['features']

            # Try to get current frame tokens
            if current_frame in video_cached_tokens:
                curr_data = video_cached_tokens[current_frame]
                if isinstance(curr_data, dict) and 'tokens' in curr_data:
                    current_tokens = curr_data['tokens']

            # Generate tokens if needed
            if reference_features is None or current_tokens is None:
                need_processing = True
                logger.info("Generating tokens on-the-fly for requested frames")
                
                # Load and transform frames
                current_frame_tensor = self.transform(
                    Image.open(frames[current_frame]).convert('RGB')
                ).unsqueeze(0)
                
                reference_frame_tensor = self.transform(
                    Image.open(frames[0]).convert('RGB')
                ).unsqueeze(0)

                # Extract features and tokens
                with torch.no_grad():
                    f_r, t_r, t_c = self.model.tokens(
                        current_frame_tensor,
                        reference_frame_tensor
                    )

                    # Update reference features if needed
                    if reference_features is None:
                        reference_features = [f.cpu().numpy() for f in f_r]
                        self.token_cache.set_tokens(video_id, 0, {
                            'features': reference_features,
                            'tokens': t_r.cpu().numpy()
                        })

                    # Update current frame tokens
                    current_tokens = t_c.cpu().numpy()
                    self.token_cache.set_tokens(video_id, current_frame, {
                        'tokens': current_tokens
                    })

                    logger.info(f"Generated tokens for frame {current_frame}")

            # Prepare response data
            features_data = {
                'reference_features': [
                    f.tolist() if isinstance(f, np.ndarray) else f 
                    for f in reference_features
                ],
                'current_token': current_tokens.tolist() if isinstance(current_tokens, np.ndarray) else current_tokens
            }

            # Get cached ranges
            cached_frame_indices = sorted(list(video_cached_tokens.keys()))
            cached_frames_ranges = []
            if cached_frame_indices:
                start_idx = cached_frame_indices[0]
                prev_idx = start_idx
                
                for idx in cached_frame_indices[1:]:
                    if idx != prev_idx + 1:
                        cached_frames_ranges.append((start_idx, prev_idx))
                        start_idx = idx
                    prev_idx = idx
                cached_frames_ranges.append((start_idx, prev_idx))

            # Prepare response
            return {
                "type": "frame_features",
                "video_id": video_id,
                "current_frame": current_frame,
                "reference_frame": 0,  # Always use frame 0 as reference
                "features": features_data,
                "metadata": {
                    "frame_count": frame_count,
                    "cached": not need_processing,
                    "total_cached_frames": len(video_cached_tokens),
                    "cached_ranges": cached_frames_ranges,
                    "processing_progress": self.token_cache.get_generation_progress(video_id)
                }
            }

        except Exception as e:
            logger.error(f"🔥 🔥 Error in process_video_frames: {str(e)}", exc_info=True)
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
            logger.error(f"🔥 🔥 Error processing frame request: {str(e)}")
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
            logger.error(f"🔥 🔥 Failed to start server: {str(e)}")
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