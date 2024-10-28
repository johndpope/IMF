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
from livekit_server import LiveKitIMFServer
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
        
        # LiveKit configuration
        self.LIVEKIT_URL = os.getenv("LIVEKIT_URL", "http://localhost:7880")
        self.LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
        self.LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
        
        # Model configuration
        self.CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "./checkpoints/checkpoint.pth")
        self.CACHE_DIR = os.getenv("TOKEN_CACHE_DIR", "./token_cache")
        
        # Dataset configuration
        self.DATASET_ROOT = os.getenv("DATASET_ROOT", "/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/processed_dataset")
        
        # CORS configuration - Updated for Cloudflare tunnel
        self.ALLOWED_ORIGINS = [
            "https://chat.covershot.ai",
            "https://ws.covershot.ai",
            "http://localhost:3001",
            "http://localhost:8000"
        ]
        
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

# Rest of your code remains the same until the run() method

    def run(self):
        """Run the server without SSL"""
        uvicorn.run(
            self.app,
            host=self.config.HOST,
            port=self.config.PORT
        )

if __name__ == "__main__":
    server = LiveKitIMFServer()
    server.run()