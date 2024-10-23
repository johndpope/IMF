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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IMFServer:
    def __init__(self, checkpoint_path: str = "./checkpoints/checkpoint.pth"):
        self.app = FastAPI()
        self.setup_cors()
        self.setup_routes()
        self.load_model(checkpoint_path)
        self.active_connections: List[WebSocket] = []



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

    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def load_model(self, checkpoint_path: str):
        # Initialize model
        self.model = IMFModel()
        self.model.eval()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def setup_routes(self):
        @self.app.get("/videos")
        async def list_videos():
            """List all available videos"""
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
    
        @self.app.get("/videos/{video_id}/frames/{frame_id}")
        async def get_frame(video_id: int, frame_id: int):
            """Get a specific frame from a video"""
            try:
                video_folder = self.dataset.video_folders[video_id]
                frames = sorted([f for f in Path(video_folder).glob("*.png")])
                if frame_id >= len(frames):
                    raise HTTPException(status_code=404, message="Frame not found")
                
                frame_path = frames[frame_id]
                img = Image.open(frame_path)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                return {
                    "frame": base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                }
            except IndexError:
                raise HTTPException(status_code=404, detail="Video not found")
            
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
                        data = await websocket.receive_json()
                        logger.info(f"Received message: {data}")
                        
                        if data["type"] == "process_frames":
                            response = await self.process_video_frames(
                                video_id=data["video_id"],
                                current_frame=data["current_frame"],
                                reference_frame=data["reference_frame"]
                            )
                            await websocket.send_json(response)
                        
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected")
                    self.active_connections.remove(websocket)
                except Exception as e:
                    logger.error(f"Error in WebSocket connection: {str(e)}")
                    if websocket in self.active_connections:
                        self.active_connections.remove(websocket)
            except Exception as e:
                logger.error(f"Failed to establish WebSocket connection: {str(e)}")

    async def process_video_frames(self, video_id: int, current_frame: int, reference_frame: int) -> Dict:
        """Process a pair of frames from a video"""
        try:
            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            
            # Load frames
            current = cv2.imread(str(frames[current_frame]))
            reference = cv2.imread(str(frames[reference_frame]))
            
            # Extract features
            features_data = self.extract_features(current, reference)
            
            return {
                "type": "frame_features",
                "video_id": video_id,
                "current_frame": current_frame,
                "reference_frame": reference_frame,
                "features": features_data
            }
        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}")
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
        # Preprocess frames
        def preprocess_frame(frame):
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to 256x256
            frame = cv2.resize(frame, (256, 256))
            # Convert to torch tensor and normalize
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            frame = frame.unsqueeze(0)
            return frame

        x_current = preprocess_frame(current_frame)
        x_reference = preprocess_frame(reference_frame)

        # Extract features and tokens using the encoder parts of the model
        f_r, t_r, t_c = self.model.tokens(x_current, x_reference)

        # Convert to serializable format
        features_data = {
            "reference_features": [f.cpu().numpy().tolist() for f in f_r],
            "reference_token": t_r.cpu().numpy().tolist(),
            "current_token": t_c.cpu().numpy().tolist()
        }

        return features_data

    async def process_frame_request(self, data: Dict[str, Any]):
        frame_idx = data["frame_index"]
        ref_frame_idx = data.get("reference_frame_index", 0)

        # Get frames
        current_frame = self.stored_frames[frame_idx]
        reference_frame = self.stored_frames[ref_frame_idx]

        # Extract features
        features_data = self.extract_features(current_frame, reference_frame)

        return {
            "type": "frame_features",
            "frame_index": frame_idx,
            "reference_frame_index": ref_frame_idx,
            "features": features_data
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
                log_level="debug",  # Enable debug logging
                ws_ping_interval=30.0,  # Send ping every 30 seconds
                ws_ping_timeout=10.0,   # Wait 10 seconds for pong
                timeout_keep_alive=30,   # Keep-alive timeout
            )
            
            server = uvicorn.Server(config)
            server.run()
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            raise

if __name__ == "__main__":
    server = IMFServer()

    # Use the same certificates as your Next.js server
    server.run(
        ssl_certfile="192.168.1.108.pem",
        ssl_keyfile="192.168.1.108-key.pem"
    )