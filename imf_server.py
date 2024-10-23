from fastapi import FastAPI, WebSocket, UploadFile, File
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


class IMFServer:
    def __init__(self, checkpoint_path: str = "./checkpoints/checkpoint.pth"):
        self.app = FastAPI()
        self.setup_cors()
        self.setup_routes()
        self.load_model(checkpoint_path)
        self.active_connections: List[WebSocket] = []

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
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket_connection(websocket)

        @self.app.post("/upload-video")
        async def upload_video(file: UploadFile = File(...)):
            return await self.handle_video_upload(file)

    async def handle_websocket_connection(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        try:
            while True:
                # Receive message
                data = await websocket.receive_json()
                
                if data["type"] == "request_frame":
                    frame_data = await self.process_frame_request(data)
                    await websocket.send_json(frame_data)
                
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.active_connections.remove(websocket)
            await websocket.close()

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
        
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            ssl_version=ssl.PROTOCOL_TLS,
            ssl_cert_reqs=ssl.CERT_NONE
        )
        
        server = uvicorn.Server(config)
        server.run()

if __name__ == "__main__":
    server = IMFServer()

    # Use the same certificates as your Next.js server
    server.run(
        ssl_certfile="192.168.1.108.pem",
        ssl_keyfile="192.168.1.108-key.pem"
    )