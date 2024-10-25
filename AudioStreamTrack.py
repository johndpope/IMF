from fastapi import FastAPI, WebSocket, UploadFile, File, WebSocketDisconnect, HTTPException,Query
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
from VideoAudioDataset import VideoAudioDataset
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
import fractions


class AudioStreamTrack(MediaStreamTrack):
    """Custom audio track for streaming audio samples aligned with video frames"""
    kind = "audio"
    
    def __init__(self, frames_queue):
        super().__init__()
        self.frames_queue = frames_queue
        self._timestamp = 0
        self._timebase = fractions.Fraction(1, 48000)  # WebRTC standard sample rate
        
    async def recv(self):
        try:
            # Get audio chunk from queue
            frame_data = await self.frames_queue.get()
            
            # Ensure frame_data is the right shape and type
            if isinstance(frame_data, np.ndarray):
                # Convert to mono if stereo
                if len(frame_data.shape) > 1 and frame_data.shape[1] > 1:
                    frame_data = np.mean(frame_data, axis=1)
                
                # Ensure the data is the right type
                frame_data = frame_data.astype(np.int16)
                
                # Create audio frame
                frame = AudioFrame.from_ndarray(
                    frame_data,
                    format='s16',
                    layout='mono'
                )
                
                frame.pts = self._timestamp
                frame.time_base = self._timebase
                self._timestamp += frame.samples
                
                return frame
            else:
                raise ValueError("Invalid frame data type")
        except Exception as e:
            logger.error(f"Error in AudioStreamTrack.recv: {e}")
            # Return silent frame on error
            silent_frame = np.zeros(1024, dtype=np.int16)  # Default chunk size
            frame = AudioFrame.from_ndarray(
                silent_frame,
                format='s16',
                layout='mono'
            )
            frame.pts = self._timestamp
            frame.time_base = self._timebase
            self._timestamp += frame.samples
            return frame