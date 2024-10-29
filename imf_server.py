from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect, HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
import torch
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer

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
from VideoAudioDataset import VideoAudioDataset
from pathlib import Path
from starlette.websockets import WebSocketState
from fastapi.responses import JSONResponse
from collections import OrderedDict
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
from av import AudioFrame
import numpy as np

from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from AudioStreamTrack import AudioStreamTrack
import os 


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




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
        videos_root = "/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/processed_dataset"
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.dataset = VideoAudioDataset(
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


        logger.info(f"Loaded dataset with {len(self.dataset)} videos")
        logger.info(f"Frame rate: {self.dataset.frame_rate}")
        logger.info(f"Audio sample rate: {self.dataset.audio_sample_rate}")
        
        # Initialize connection tracking
        self.ice_gathering_state = {}
        self.ice_connection_states = {}
        self.connected_peers = set()
        self.data_channels = {}
        self.pending_messages = {}
        self.audio_queues = {}
        self.frame_timers = {}

    async def list_videos(self):
        """List all available videos with metadata"""
        try:
            videos = []
            for idx in range(len(self.dataset)):
                video_metadata = self.dataset.videos[idx]
                frame_metadata = video_metadata['frames']
                
                videos.append({
                    "id": idx,
                    "name": os.path.basename(os.path.splitext(video_metadata['video_path'])[0]),
                    "frame_count": frame_metadata['total_frames'],
                    "duration": frame_metadata['total_frames'] / self.dataset.frame_rate,
                    "frame_rate": self.dataset.frame_rate
                })
            return {"videos": videos}
        except Exception as e:
            logger.error(f"🔥 🔥 Error listing videos: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_frame(self, video_id: int, frame_id: int):
        """Get a specific frame from a video"""

        if video_id < 0 or video_id >= len(self.dataset):
            raise HTTPException(
                status_code=404,
                detail=f"Video {video_id} not found"
            )
        
        # Get video metadata
        video_metadata = self.dataset.videos[video_id]
        frame_metadata = video_metadata['frames']
        
        if frame_id < 0 or frame_id >= frame_metadata['total_frames']:
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame_id} not found. Video has {frame_metadata['total_frames']} frames"
            )
        
        # Get frame path
        frame_info = frame_metadata['frames'][frame_id]
        frame_path = os.path.join(
            self.dataset.root_dir,
            os.path.splitext(video_metadata['video_path'])[0],
            frame_info['path']
        )
        
        # Load and convert frame
        try:
            img = Image.open(frame_path).convert('RGB')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            
            return JSONResponse({
                "frame": base64.b64encode(img_bytes.getvalue()).decode('utf-8'),
                "metadata": {
                    "frame_number": frame_id,
                    "timestamp": frame_info['timestamp'],
                    "total_frames": frame_metadata['total_frames'],
                    "video_id": video_id
                }
            })
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")
            
    async def create_connection(self, peer_id: str) -> RTCPeerConnection:
        """Create and set up a new WebRTC peer connection"""
        pc = RTCPeerConnection(RTCConfiguration([
            RTCIceServer(urls=["stun:stun.l.google.com:19302"])
        ]))
        
        # Set up data channel for token streaming
        dc = pc.createDataChannel("tokens")
        dc.on("open", lambda: logger.info(f"Data channel opened for peer {peer_id}"))
        dc.on("close", lambda: logger.info(f"Data channel closed for peer {peer_id}"))
        
        # Connection state monitoring
        @pc.on("connectionstatechange")
        async def on_connection_state():
            logger.info(f"Connection state for {peer_id}: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await self.cleanup_peer(peer_id)

        self.peer_connections[peer_id] = {
            "pc": pc,
            "dc": dc
        }
        return pc

    async def cleanup_peer(self, peer_id: str):
        """Clean up all resources for a peer"""
        try:
            # Cancel streaming tasks
            if peer_id in self.stream_tasks:
                for task in self.stream_tasks[peer_id]:
                    task.cancel()
                del self.stream_tasks[peer_id]

            # Close WebRTC connection
            if peer_id in self.peer_connections:
                conn = self.peer_connections[peer_id]
                if "dc" in conn:
                    conn["dc"].close()
                if "pc" in conn:
                    await conn["pc"].close()
                del self.peer_connections[peer_id]

            # Clear audio queue
            if peer_id in self.audio_queues:
                self.audio_queues[peer_id].empty()
                del self.audio_queues[peer_id]

            logger.info(f"Cleaned up resources for peer {peer_id}")
        except Exception as e:
            logger.error(f"Error during cleanup for peer {peer_id}: {e}")

    async def stream_video(self, peer_id: str, video_id: int):
        """Stream video tokens and audio for a specific video"""
        try:
            conn = self.peer_connections.get(peer_id)
            if not conn:
                raise ValueError(f"No connection found for peer {peer_id}")

            dc = conn["dc"]
            pc = conn["pc"]

            # Get video metadata
            video_metadata = self.dataset.videos[video_id]
            frame_metadata = video_metadata['frames']
            audio_metadata = video_metadata['audio']
            total_frames = frame_metadata['total_frames']

            # Set up audio streaming
            audio_queue = asyncio.Queue()
            self.audio_queues[peer_id] = audio_queue
            audio_track = AudioStreamTrack(audio_queue)
            pc.addTrack(audio_track)

            # Create streaming tasks
            token_task = asyncio.create_task(
                self.stream_tokens(dc, video_id, total_frames)
            )
            audio_task = asyncio.create_task(
                self.stream_audio(peer_id, video_id, audio_metadata)
            )

            self.stream_tasks[peer_id] = [token_task, audio_task]
            await asyncio.gather(token_task, audio_task)

        except Exception as e:
            logger.error(f"Error in stream_video: {e}")
            await self.cleanup_peer(peer_id)

    async def stream_tokens(self, dc, video_id: int, total_frames: int):
        """Stream tokens for video frames"""
        try:
            frame_rate = self.dataset.frame_rate
            frame_interval = 1.0 / frame_rate
            
            for frame_idx in range(total_frames):
                try:
                    # Get cached or generate tokens
                    token_data = await self.get_frame_tokens(video_id, frame_idx)
                    
                    # Create frame message
                    message = {
                        "type": "frame",
                        "frameIndex": frame_idx,
                        "timestamp": frame_idx * frame_interval * 1000,  # ms
                        "token": token_data,
                    }
                    
                    # Send through data channel
                    if dc.readyState == "open":
                        dc.send(json.dumps(message))
                    else:
                        raise ConnectionError("Data channel closed")
                    
                    # Maintain timing
                    await asyncio.sleep(frame_interval)
                    
                except Exception as e:
                    logger.error(f"Error streaming frame {frame_idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in token streaming: {e}")
            raise

    async def stream_audio(self, peer_id: str, video_id: int, audio_metadata: dict):
        """Stream audio chunks for a video"""
        try:
            audio_queue = self.audio_queues.get(peer_id)
            if not audio_queue:
                raise ValueError("No audio queue found")

            for chunk_info in audio_metadata['chunks']:
                try:
                    # Load audio chunk
                    chunk_path = os.path.join(
                        self.dataset.root_dir,
                        os.path.splitext(self.dataset.videos[video_id]['video_path'])[0],
                        chunk_info['path']
                    )
                    audio_chunk = np.load(chunk_path)
                    
                    # Queue chunk for streaming
                    await audio_queue.put(audio_chunk)
                    
                    # Wait for duration of chunk
                    await asyncio.sleep(chunk_info['duration'])
                    
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    # Send silence on error
                    await audio_queue.put(np.zeros(int(48000 * chunk_info['duration']), dtype=np.int16))

        except Exception as e:
            logger.error(f"Error in audio streaming: {e}")
            raise

    async def get_frame_tokens(self, video_id: int, frame_idx: int):
        """Get or generate tokens for a frame"""
        try:
            # Check cache first
            cached_tokens = self.token_cache.get_tokens(video_id, frame_idx)
            if cached_tokens is not None:
                return cached_tokens

            # Generate tokens if not cached
            frame_data = await self.process_video_frames(video_id, frame_idx, 0)
            return frame_data["features"]["current_token"]

        except Exception as e:
            logger.error(f"Error getting frame tokens: {e}")
            raise


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
        """Handle initial connection with audio setup"""
        try:
            payload = message.get("payload", {})
            fps = payload.get("fps", self.dataset.frame_rate)
            
            # Create RTCConfiguration
            config = RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
            
            # Create peer connection
            pc = RTCPeerConnection(configuration=config)
            
            # Set up audio queue and track
            audio_queue = asyncio.Queue()
            self.audio_queues[peer_id] = audio_queue
            
            # Create audio track
            audio_track = AudioStreamTrack(frames_queue=audio_queue)
            pc.addTrack(audio_track)
            
            # Setup data channel and event handlers
            @pc.on("datachannel")
            def on_datachannel(channel):
                logger.info(f"Data channel established for peer {peer_id}")
                self.setup_data_channel(channel, peer_id)

            @pc.on("connectionstatechange")
            async def on_connection_state_change():
                logger.info(f"Connection state changed for peer {peer_id}: {pc.connectionState}")
                logger.info(f"ICE gathering state: {pc.iceGatheringState}")
                logger.info(f"ICE connection state: {pc.iceConnectionState}")
                logger.info(f"Signaling state: {pc.signalingState}")
                
                if pc.connectionState == "failed":
                    logger.error(f"👺 Connection failed for peer {peer_id}")
                    logger.error(f"Last successful candidate pair: {pc.currentLocalDescription}")
                    await self.handle_connection_failure(peer_id)
                elif pc.connectionState == "disconnected":
                    logger.warning(f"👺 Connection disconnected for peer {peer_id}")
                elif pc.connectionState == "connected":
                    if peer_id not in self.data_channels:
                        data_channel = pc.createDataChannel("frames")
                        self.setup_data_channel(data_channel, peer_id)

            # Send init response
            await websocket.send_json({
                "type": "init_response",
                "status": "success",
                "payload": {
                    "rtcConfig": {
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    },
                    "fps": fps,
                    "sampleRate": self.dataset.audio_sample_rate,
                    "maxFrames": 300
                }
            })

            return {
                "peer_connection": pc,
                "audio_queue": audio_queue,
                "fps": fps
            }

        except Exception as e:
            logger.error(f"Error in init connection: {str(e)}")
            raise

    async def handle_connection_failure(self, peer_id: str):
        """Clean up on connection failure"""
        try:
            # Clean up audio resources
            if peer_id in self.audio_queues:
                self.audio_queues[peer_id].empty()
                del self.audio_queues[peer_id]
                
            if peer_id in self.frame_timers:
                del self.frame_timers[peer_id]
                
            await super().handle_connection_failure(peer_id)
            
        except Exception as e:
            logger.error(f"Error during connection failure cleanup: {e}")

    def cleanup(self, peer_id: str):
        """Clean up resources for a peer"""
        try:
            # Clean up audio resources
            if peer_id in self.audio_queues:
                self.audio_queues[peer_id].empty()
                del self.audio_queues[peer_id]
                
            if peer_id in self.frame_timers:
                del self.frame_timers[peer_id]
            
            super().cleanup(peer_id)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


    async def create_peer_connection(self, peer_id: str) -> RTCPeerConnection:
        """Create and configure a new peer connection"""
        pc = RTCPeerConnection(configuration=self.rtc_configuration)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state [{peer_id}]: {pc.connectionState}")
            if pc.connectionState == "failed":
                await self.cleanup_peer_connection(peer_id)
            elif pc.connectionState == "closed":
                await self.cleanup_peer_connection(peer_id)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE connection state [{peer_id}]: {pc.iceConnectionState}")

        self.peer_connections[peer_id] = pc
        return pc

    async def cleanup_peer_connection(self, peer_id: str):
        """Clean up resources for a peer connection"""
        try:
            if peer_id in self.peer_connections:
                pc = self.peer_connections[peer_id]
                await pc.close()
                del self.peer_connections[peer_id]

            if peer_id in self.audio_queues:
                self.audio_queues[peer_id].empty()
                del self.audio_queues[peer_id]

            if peer_id in self.active_streams:
                del self.active_streams[peer_id]

            logger.info(f"Cleaned up resources for peer {peer_id}")

        except Exception as e:
            logger.error(f"Error cleaning up peer {peer_id}: {e}")

    async def start_audio_stream(self, peer_id: str, video_id: int):
        """Start streaming audio for a video"""
        try:
            # Get video metadata
            video_metadata = self.dataset.videos[video_id]
            audio_metadata = video_metadata['audio']
            
            # Create audio queue if it doesn't exist
            if peer_id not in self.audio_queues:
                self.audio_queues[peer_id] = asyncio.Queue()
            
            # Get the peer connection
            pc = self.peer_connections.get(peer_id)
            if not pc:
                raise ValueError(f"No peer connection found for {peer_id}")

            # Create and add audio track
            audio_track = AudioStreamTrack(self.audio_queues[peer_id])
            pc.addTrack(audio_track)

            # Start audio streaming task
            self.active_streams[peer_id] = {
                'video_id': video_id,
                'current_chunk': 0,
                'total_chunks': len(audio_metadata['chunks'])
            }

            asyncio.create_task(self.stream_audio_chunks(peer_id, video_id))

        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            await self.cleanup_peer_connection(peer_id)
            raise

    async def stream_audio_chunks(self, peer_id: str, video_id: int):
        """Stream audio chunks for a video"""
        try:
            stream_info = self.active_streams.get(peer_id)
            if not stream_info:
                return

            video_metadata = self.dataset.videos[video_id]
            audio_metadata = video_metadata['audio']
            audio_queue = self.audio_queues[peer_id]

            for chunk_info in audio_metadata['chunks']:
                if peer_id not in self.active_streams:
                    break

                # Load audio chunk
                chunk_path = os.path.join(
                    self.dataset.root_dir,
                    os.path.splitext(video_metadata['video_path'])[0],
                    chunk_info['path']
                )
                
                try:
                    audio_chunk = np.load(chunk_path)
                    await audio_queue.put(audio_chunk)
                    
                    # Wait for proper timing based on chunk duration
                    await asyncio.sleep(chunk_info['duration'])
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in audio streaming: {e}")
        finally:
            if peer_id in self.active_streams:
                await self.cleanup_peer_connection(peer_id)

    def setup_routes(self):

        @self.app.websocket("/rtc/{peer_id}")
        async def websocket_rtc(websocket: WebSocket, peer_id: str):
            try:
                await websocket.accept()
                logger.info(f"WebRTC WebSocket connection accepted for peer {peer_id}")

                while True:
                    try:
                        message = await websocket.receive_json()
                        message_type = message.get("type")

                        if message_type == "offer":
                            # Create peer connection if it doesn't exist
                            if peer_id not in self.peer_connections:
                                pc = await self.create_peer_connection(peer_id)
                            else:
                                pc = self.peer_connections[peer_id]

                            # Set remote description
                            offer = RTCSessionDescription(
                                sdp=message["sdp"]["sdp"],
                                type=message["sdp"]["type"]
                            )
                            await pc.setRemoteDescription(offer)

                            # Create and send answer
                            answer = await pc.createAnswer()
                            await pc.setLocalDescription(answer)
                            
                            await websocket.send_json({
                                "type": "answer",
                                "sdp": {
                                    "type": answer.type,
                                    "sdp": answer.sdp
                                }
                            })

                        elif message_type == "ice-candidate":
                            if peer_id in self.peer_connections:
                                pc = self.peer_connections[peer_id]
                                candidate = RTCIceCandidate(
                                    sdpMid=message["candidate"]["sdpMid"],
                                    sdpMLineIndex=message["candidate"]["sdpMLineIndex"],
                                    candidate=message["candidate"]["candidate"]
                                )
                                await pc.addIceCandidate(candidate)

                        elif message_type == "start-stream":
                            video_id = message.get("videoId")
                            if video_id is not None:
                                await self.start_audio_stream(peer_id, video_id)

                    except WebSocketDisconnect:
                        logger.info(f"WebSocket disconnected for peer {peer_id}")
                        break
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                        break

            except Exception as e:
                logger.error(f"Error in WebSocket connection: {e}")
            finally:
                await self.cleanup_peer_connection(peer_id)

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



        

        @self.app.get("/videos/{video_id}/tokens")
        async def get_bulk_tokens(
            video_id: int,
            start: int = Query(default=0, description="Start frame index"),
            end: int = Query(default=99, description="End frame index")
        ):
            """Get tokens for a range of frames from a video using the JSON dataset structure"""
            try:
                # Validate video_id
                if video_id < 0 or video_id >= len(self.dataset):
                    raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

                # Get video metadata from dataset
                video_metadata = self.dataset.videos[video_id]
                frame_metadata = video_metadata['frames']
                frame_count = frame_metadata['total_frames']

                # Validate frame range
                if start < 0 or end >= frame_count:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid frame range. Video has {frame_count} frames"
                    )

                # Get cached tokens
                tokens = {}
                video_cached_tokens = self.token_cache.video_tokens.get(video_id, {})

                # Get base video folder path
                video_folder = os.path.join(
                    self.dataset.root_dir,
                    os.path.splitext(video_metadata['video_path'])[0]
                )

                # Process frames in batches for better performance
                batch_size = 16  # Adjust based on memory constraints
                for batch_start in range(start, end + 1, batch_size):
                    batch_end = min(batch_start + batch_size, end + 1)
                    batch_frames = []
                    batch_indices = []

                    # Collect frames that need processing
                    for frame_idx in range(batch_start, batch_end):
                        # Check if we have cached tokens
                        if frame_idx in video_cached_tokens:
                            token_data = video_cached_tokens[frame_idx]
                            if isinstance(token_data, dict) and 'tokens' in token_data:
                                tokens[frame_idx] = token_data['tokens'].tolist() if isinstance(token_data['tokens'], np.ndarray) else token_data['tokens']
                            continue

                        # Get frame path from metadata
                        frame_info = frame_metadata['frames'][frame_idx]
                        # Use frame_path instead of path
                        frame_path = os.path.join(video_folder, 'frames', frame_info['frame_path'])

                        # Load and transform frame
                        try:
                            img = Image.open(frame_path).convert('RGB')
                            frame_tensor = self.transform(img).unsqueeze(0)
                            batch_frames.append(frame_tensor)
                            batch_indices.append(frame_idx)
                        except Exception as e:
                            logger.error(f"Error loading frame {frame_idx} from {frame_path}: {e}")
                            continue

                    # Process batch if there are uncached frames
                    if batch_frames:
                        try:
                            # Stack frames into a single batch tensor
                            batch_tensor = torch.cat(batch_frames, dim=0)

                            # Generate tokens using the latent token encoder
                            with torch.no_grad():
                                batch_tokens = self.model.latent_token_encoder(batch_tensor)
                                
                                # Convert to numpy and process each token
                                batch_tokens_np = batch_tokens.cpu().numpy()
                                
                                for idx, frame_idx in enumerate(batch_indices):
                                    # Extract token for this frame
                                    frame_token = batch_tokens_np[idx]
                                    # Convert to list for JSON serialization
                                    token_list = frame_token.tolist()
                                    tokens[frame_idx] = token_list
                                    
                                    # Cache the generated token
                                    self.token_cache.set_tokens(video_id, frame_idx, {
                                        'tokens': frame_token  # Store numpy array in cache
                                    })

                        except Exception as e:
                            logger.error(f"Error processing batch: {str(e)}", exc_info=True)
                            raise

                return {
                    "videoId": video_id,
                    "tokens": tokens,
                    "metadata": {
                        "totalFrames": frame_count,
                        "processedFrames": len(video_cached_tokens),
                        "requestedRange": {
                            "start": start,
                            "end": end
                        },
                        "frameRate": self.dataset.frame_rate,
                        "videoPath": video_metadata['video_path']
                    }
                }

            except Exception as e:
                logger.error(f"Error in bulk token fetch: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
            
            
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
            """List all available videos with metadata"""
            try:
                videos = []
                # Get total videos in dataset
                total_videos = len(self.dataset)
                logger.info(f"Found {total_videos} total videos in dataset")

                # Iterate through available videos
                for idx, video_metadata in enumerate(self.dataset.videos):
                    try:
                        frame_metadata = video_metadata['frames']
                        video_info = {
                            "id": idx,
                            "name": os.path.basename(os.path.splitext(video_metadata['video_path'])[0]),
                            "frame_count": frame_metadata['total_frames'],
                            "duration": frame_metadata['total_frames'] / self.dataset.frame_rate,
                            "frame_rate": self.dataset.frame_rate
                        }
                        videos.append(video_info)
                        
                        logger.debug(f"Added video: {video_info['name']} with {video_info['frame_count']} frames")
                        
                    except KeyError as ke:
                        logger.error(f"Missing key in video metadata for index {idx}: {ke}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing video at index {idx}: {e}")
                        continue

                if not videos:
                    logger.warning("No valid videos found in dataset")
                    return JSONResponse({
                        "videos": [],
                        "message": "No valid videos found"
                    })

                logger.info(f"Successfully listed {len(videos)} videos")
                return JSONResponse({
                    "videos": videos,
                    "total": len(videos)
                })
                
            except Exception as e:
                logger.error(f"🔥 🔥 Error listing videos: {str(e)}")
                # Return empty list instead of error for better frontend handling
                return JSONResponse({
                    "videos": [],
                    "error": str(e)
                })


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
        """Handle video and audio streaming with proper synchronization"""
        try:
            if channel.readyState != "open":
                logger.warning(f"Data channel not open for peer {peer_id}")
                return

            # Get video metadata
            video_metadata = self.dataset.videos[video_id]
            frame_metadata = video_metadata['frames']
            audio_metadata = video_metadata['audio']
            
            total_frames = frame_metadata['total_frames']
            frame_duration = 1.0 / self.dataset.frame_rate
            
            # Get reference data first
            reference_data = await self.process_video_frames(video_id, 0, 0)
            reference_features = reference_data["features"]["reference_features"]

            # Get audio queue for this peer
            audio_queue = self.audio_queues.get(peer_id)
            if not audio_queue:
                logger.error(f"No audio queue found for peer {peer_id}")
                return

            # Initialize frame timer
            self.frame_timers[peer_id] = {
                'start_time': time.time(),
                'frame_count': 0,
                'last_audio_ts': 0
            }

            # Stream frames with synchronized audio
            for frame_idx in range(total_frames):
                if channel.readyState != "open":
                    logger.info(f"Data channel closed for peer {peer_id}")
                    break

                try:
                    # Calculate timing
                    target_time = frame_idx * frame_duration
                    current_time = time.time() - self.frame_timers[peer_id]['start_time']
                    
                    # Wait if we're ahead of schedule
                    if current_time < target_time:
                        await asyncio.sleep(target_time - current_time)

                    # Process frame
                    frame_data = await self.process_video_frames(video_id, frame_idx, 0)

                    # Get corresponding audio chunk
                    chunk_info = audio_metadata['chunks'][frame_idx]
                    chunk_path = os.path.join(
                        self.dataset.root_dir,
                        os.path.splitext(video_metadata['video_path'])[0],
                        chunk_info['path']
                    )
                    audio_chunk = self.dataset.load_audio_chunk(chunk_path)

                    # Put audio chunk in queue
                    await audio_queue.put(audio_chunk)

                    # Send frame token with timing info
                    message = {
                        "type": "frame_token",
                        "frameIndex": frame_idx,
                        "token": frame_data["features"]["current_token"],
                        "timestamp": target_time * 1000,  # Convert to ms
                        "audioTimestamp": chunk_info['start_time'] * 1000
                    }

                    try:
                        channel.send(json.dumps(message))
                        logger.info(f"Sent frame {frame_idx} to peer {peer_id}")
                    except Exception as send_error:
                        logger.error(f"Error sending frame {frame_idx}: {send_error}")
                        if "closed" in str(send_error).lower():
                            break

                    # Update timing info
                    self.frame_timers[peer_id]['frame_count'] += 1
                    self.frame_timers[peer_id]['last_audio_ts'] = chunk_info['start_time']

                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in video stream: {e}")
        finally:
            # Cleanup
            if peer_id in self.frame_timers:
                del self.frame_timers[peer_id]
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

            # Get video metadata
            video_metadata = self.dataset.videos[video_id]
            frame_metadata = video_metadata['frames']
            
            # Log metadata structure for debugging
            logger.info(f"Processing video: {video_metadata['video_path']}")
            
            # Get base video folder
            video_folder = os.path.join(
                self.dataset.root_dir,
                os.path.splitext(video_metadata['video_path'])[0]
            )

            # Check cached data
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
                
                # Get first frame info from metadata
                first_frame_info = frame_metadata['frames'][0]
                
                # Construct full frame path including frames subdirectory
                frame_path = os.path.join(video_folder, 'frames', first_frame_info['frame_path'])
                
                logger.info(f"Loading reference frame from: {frame_path}")
                
                if not os.path.exists(frame_path):
                    raise ValueError(f"Reference frame not found: {frame_path}")
                
                # Load and transform reference frame
                img = Image.open(frame_path).convert('RGB')
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

            # Convert to lists for JSON serialization
            reference_features_list = [
                f.tolist() if isinstance(f, np.ndarray) else f 
                for f in reference_features
            ]
            reference_token_list = reference_token.tolist() if isinstance(reference_token, np.ndarray) else reference_token

            return {
                "video_id": video_id,
                "reference_features": reference_features_list,
                "reference_token": reference_token_list,
                "metadata": {
                    "frame_count": frame_metadata['total_frames'],
                    "frame_rate": self.dataset.frame_rate,
                    "token_shape": video_metadata.get('token_shape', [1, 32])
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

            # Get video metadata
            video_metadata = self.dataset.videos[video_id]
            frame_metadata = video_metadata['frames']
            total_frames = frame_metadata['total_frames']
            
            # Get base video folder
            video_folder = os.path.join(
                self.dataset.root_dir,
                os.path.splitext(video_metadata['video_path'])[0]
            )
            
            if current_frame >= total_frames or reference_frame >= total_frames:
                raise ValueError(f"Frame index out of range. Video has {total_frames} frames")

            # Get cached tokens
            video_cached_tokens = self.token_cache.video_tokens.get(video_id, {})
            logger.info(f"Found {len(video_cached_tokens)} cached frames for video {video_id}")

            # Get reference features and current tokens
            reference_features = None
            current_tokens = None

            if 0 in video_cached_tokens:
                ref_data = video_cached_tokens[0]
                reference_features = ref_data.get('features')

            if current_frame in video_cached_tokens:
                current_tokens = video_cached_tokens[current_frame].get('tokens')

            if reference_features is None or current_tokens is None:
                logger.info("Generating tokens on-the-fly")
                
                # Get frame paths from metadata
                current_frame_info = frame_metadata['frames'][current_frame]
                reference_frame_info = frame_metadata['frames'][0]  # Always use frame 0
                
                # Construct full frame paths including frames subdirectory
                current_frame_path = os.path.join(video_folder, 'frames', current_frame_info['frame_path'])
                reference_frame_path = os.path.join(video_folder, 'frames', reference_frame_info['frame_path'])
                
                logger.info(f"Loading frames from: {current_frame_path} and {reference_frame_path}")
                
                # Load and transform frames
                current_frame_tensor = self.transform(
                    Image.open(current_frame_path).convert('RGB')
                ).unsqueeze(0)
                
                reference_frame_tensor = self.transform(
                    Image.open(reference_frame_path).convert('RGB')
                ).unsqueeze(0)

                # Generate features and tokens
                with torch.no_grad():
                    f_r, t_r, t_c = self.model.tokens(
                        current_frame_tensor,
                        reference_frame_tensor
                    )

                    if reference_features is None:
                        reference_features = [f.cpu().numpy() for f in f_r]
                        self.token_cache.set_tokens(video_id, 0, {
                            'features': reference_features,
                            'tokens': t_r.cpu().numpy()
                        })

                    current_tokens = t_c.cpu().numpy()
                    self.token_cache.set_tokens(video_id, current_frame, {
                        'tokens': current_tokens
                    })

            # Prepare response
            features_data = {
                'reference_features': [f.tolist() if isinstance(f, np.ndarray) else f for f in reference_features],
                'current_token': current_tokens.tolist() if isinstance(current_tokens, np.ndarray) else current_tokens
            }

            return {
                "type": "frame_features",
                "video_id": video_id,
                "current_frame": current_frame,
                "reference_frame": reference_frame,
                "features": features_data,
                "metadata": {
                    "frame_count": total_frames,
                    "cached": bool(current_tokens is not None),
                    "total_cached_frames": len(video_cached_tokens),
                    "token_shape": video_metadata.get('token_shape', [1, 32]),
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
        

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Start the server without SSL
        
        Args:
            host (str): Host address to bind to
            port (int): Port number to listen on
        """
        logger.info(f"Starting server on {host}:{port}")
        
        try:
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
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
    async def main():
        server = IMFServer()
        config = uvicorn.Config(
            app=server.app,
            host="0.0.0.0",
            port=8000,
            log_level="debug",
            ws_ping_interval=30.0,
            ws_ping_timeout=10.0,
            timeout_keep_alive=30,
        )
        server_instance = uvicorn.Server(config)
        await server_instance.serve()

    # Run the async main function
    asyncio.run(main())
