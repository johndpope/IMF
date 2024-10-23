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
        """Process a pair of frames from a video"""
        logger.info(f"Processing frames - video: {video_id}, current: {current_frame}, reference: {reference_frame}")
        
        try:
            # Validate input parameters
            if video_id < 0 or video_id >= len(self.dataset):
                raise ValueError(f"Invalid video_id: {video_id}")

            video_folder = self.dataset.video_folders[video_id]
            frames = sorted([f for f in Path(video_folder).glob("*.png")])
            frame_count = len(frames)
            
            # Validate frame indices
            if current_frame < 0 or current_frame >= frame_count:
                raise ValueError(f"Invalid current_frame: {current_frame}. Valid range: 0-{frame_count-1}")
            if reference_frame < 0 or reference_frame >= frame_count:
                raise ValueError(f"Invalid reference_frame: {reference_frame}. Valid range: 0-{frame_count-1}")
            
            # Load frames
            try:
                current = cv2.imread(str(frames[current_frame]))
                reference = cv2.imread(str(frames[reference_frame]))
                
                if current is None or reference is None:
                    raise ValueError("Failed to load frames")
                
                # Extract features
                features_data = self.extract_features(current, reference)
                
                return {
                    "type": "frame_features",
                    "video_id": video_id,
                    "current_frame": current_frame,
                    "reference_frame": reference_frame,
                    "features": features_data,
                    "metadata": {
                        "frame_count": frame_count
                    }
                }
            except Exception as e:
                logger.error(f"Error processing frames: {str(e)}")
                raise ValueError(f"Frame processing error: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in process_video_frames: {str(e)}")
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