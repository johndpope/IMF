from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from typing import Dict, Optional, List
import json
import torch
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import os
from dotenv import load_dotenv
from livekit.api import LiveKitAPI, VideoGrants, AccessToken
import torchvision.transforms as transforms
from model import IMFModel
from VideoAudioDataset import VideoAudioDataset
from imf_server_cache import TokenCache
import uvicorn
import aiohttp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration management class"""
    def __init__(self):
        # Server configuration
        self.HOST = os.getenv("SERVER_HOST", "0.0.0.0")
        self.PORT = int(os.getenv("SERVER_PORT", "8000"))
        self.SSL_CERT = os.getenv("SSL_CERT_FILE", "192.168.1.108.pem")
        self.SSL_KEY = os.getenv("SSL_KEY_FILE", "192.168.1.108-key.pem")
        
        # LiveKit configuration
        self.LIVEKIT_URL = os.getenv("LIVEKIT_URL", "http://localhost:7880")
        self.LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
        self.LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
        
        # Model configuration
        self.CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "./checkpoints/checkpoint.pth")
        self.CACHE_DIR = os.getenv("TOKEN_CACHE_DIR", "./token_cache")
        
        # Dataset configuration
        self.DATASET_ROOT = os.getenv("DATASET_ROOT", "/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/processed_dataset")
        
        # CORS configuration
        self.ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://192.168.1.108:3001,wss://192.168.1.108:8000").split(",")
        
        # Room configuration
        self.MAX_PARTICIPANTS = int(os.getenv("MAX_PARTICIPANTS", "10"))
        self.ROOM_TIMEOUT = int(os.getenv("ROOM_TIMEOUT", "300"))
        
        self.validate()
    
    def validate(self):
        """Validate required configuration"""
        if not self.LIVEKIT_API_KEY or not self.LIVEKIT_API_SECRET:
            raise ValueError("LiveKit API key and secret must be set in environment variables")
        
        if not os.path.exists(self.CHECKPOINT_PATH):
            raise ValueError(f"Model checkpoint not found at {self.CHECKPOINT_PATH}")
        
        if not os.path.exists(self.DATASET_ROOT):
            raise ValueError(f"Dataset root directory not found at {self.DATASET_ROOT}")

class LiveKitIMFServer:
    def __init__(self):
        # Load configuration
        self.config = Config()
        
        # Initialize FastAPI
        self.app = FastAPI()
        self.setup_cors()
        self.setup_routes()
        
        # Load model
        self.load_model()
        
        # Initialize cache
        self.token_cache = TokenCache(
            max_size=1000, 
            cache_dir=self.config.CACHE_DIR
        )
        
        # Initialize dataset
        self.setup_dataset()
        
        # Track active sessions
        self.active_rooms: Dict[str, Dict] = {}
        self.stream_tasks: Dict[str, List[asyncio.Task]] = {}
        
        # LiveKit client will be initialized in startup event
        self.livekit: Optional[LiveKitAPI] = None
        
        logger.info("Server initialized successfully")

    def setup_cors(self):
        """Configure CORS middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def load_model(self):
        """Load the ML model"""
        try:
            self.model = IMFModel()
            self.model.eval()
            checkpoint = torch.load(
                self.config.CHECKPOINT_PATH, 
                map_location='cpu',
                weights_only=True  # Add this to address the warning
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def setup_dataset(self):
        """Initialize the dataset"""
        try:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            
            self.dataset = VideoAudioDataset(
                root_dir=self.config.DATASET_ROOT,
                transform=self.transform
            )
            
            self.videos_root = Path(self.config.DATASET_ROOT)
            
            logger.info(f"Loaded dataset with {len(self.dataset)} videos")
            logger.info(f"Frame rate: {self.dataset.frame_rate}")
            logger.info(f"Audio sample rate: {self.dataset.audio_sample_rate}")
        except Exception as e:
            logger.error(f"Failed to setup dataset: {e}")
            raise

    async def initialize_livekit(self):
        """Initialize LiveKit API client"""
        self.livekit = LiveKitAPI(
            url=self.config.LIVEKIT_URL,
            api_key=self.config.LIVEKIT_API_KEY,
            api_secret=self.config.LIVEKIT_API_SECRET
        )
        logger.info("LiveKit client initialized")

    def setup_routes(self):
        @self.app.on_event("startup")
        async def startup():
            """Initialize async components on startup"""
            await self.initialize_livekit()

        @self.app.on_event("shutdown")
        async def shutdown():
            """Cleanup async components on shutdown"""
            if self.livekit:
                await self.livekit.aclose()

        @self.app.post("/rooms/{room_name}/join")
        async def join_room(room_name: str, participant_identity: str):
            """Join or create a room"""
            try:
                # Create room if it doesn't exist
                if room_name not in self.active_rooms:
                    room = await self.livekit.room.create_room({
                        "name": room_name,
                        "empty_timeout": self.config.ROOM_TIMEOUT,
                        "max_participants": self.config.MAX_PARTICIPANTS
                    })
                    self.active_rooms[room_name] = {
                        "info": room,
                        "participants": {}
                    }

                # Generate access token
                token = AccessToken(
                    api_key=self.config.LIVEKIT_API_KEY,
                    api_secret=self.config.LIVEKIT_API_SECRET
                )
                
                grant = VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True
                )
                
                token.with_identity(participant_identity)
                token.with_grants(grant)
                
                return {
                    "room": room_name,
                    "token": token.to_jwt(),
                    "participant": participant_identity,
                    "config": {
                        "frameRate": self.dataset.frame_rate,
                        "audioSampleRate": self.dataset.audio_sample_rate
                    }
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/videos")
        async def list_videos():
            """List available videos"""
            try:
                videos = []
                for idx, video_metadata in enumerate(self.dataset.videos):
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
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "version": "1.0",
                "dataset_size": len(self.dataset),
                "active_rooms": len(self.active_rooms)
            }

    def run(self):
        """Run the server"""
        uvicorn.run(
            self.app,
            host=self.config.HOST,
            port=self.config.PORT,
            ssl_certfile=self.config.SSL_CERT,
            ssl_keyfile=self.config.SSL_KEY
        )

if __name__ == "__main__":
    server = LiveKitIMFServer()
    server.run()