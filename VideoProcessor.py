import os
from tinydb import TinyDB, Query
import datetime
from pathlib import Path
import hashlib
from tqdm import tqdm
import random
import numpy as np
import subprocess
from pydub import AudioSegment
import json
from typing import Dict, Tuple, List
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import IMFModel
import cv2

class VideoProcessor:
    def __init__(self, input_folder: str, output_base_folder: str, checkpoint_path: str):
        self.input_folder = input_folder
        self.output_base_folder = output_base_folder
        self.frame_rate = 24
        self.audio_sample_rate = 48000
        self.chunk_duration = 1/24.0

        # Initialize model
        self.model = IMFModel()
        self.model.eval()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("model loaded...")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # Initialize database
        db_path = os.path.join(output_base_folder, 'processing_state.json')
        self.db = TinyDB(db_path)
        self.processed_table = self.db.table('processed_videos')
        self.failed_table = self.db.table('failed_videos')
        self.progress_table = self.db.table('progress')

    def get_video_hash(self, video_path: str) -> str:
        """Generate a unique hash for a video based on path and modification time"""
        stat = os.stat(video_path)
        hash_string = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def is_video_processed(self, video_path: str) -> bool:
        """Check if a video has been successfully processed"""
        video_hash = self.get_video_hash(video_path)
        Video = Query()
        return bool(self.processed_table.search(Video.hash == video_hash))

    def is_video_failed(self, video_path: str) -> bool:
        """Check if a video has failed processing"""
        video_hash = self.get_video_hash(video_path)
        Video = Query()
        return bool(self.failed_table.search(Video.hash == video_hash))

    def mark_video_processed(self, video_path: str, metadata: Dict):
        """Mark a video as successfully processed"""
        video_hash = self.get_video_hash(video_path)
        self.processed_table.upsert({
            'hash': video_hash,
            'path': video_path,
            'processed_at': str(datetime.datetime.now()),
            'metadata': metadata
        }, Query().hash == video_hash)

    def mark_video_failed(self, video_path: str, error: str):
        """Mark a video as failed"""
        video_hash = self.get_video_hash(video_path)
        self.failed_table.upsert({
            'hash': video_hash,
            'path': video_path,
            'error': str(error),
            'failed_at': str(datetime.datetime.now())
        }, Query().hash == video_hash)

    def save_progress(self, total_videos: int, processed: int, failed: int):
        """Save current processing progress"""
        self.progress_table.upsert({
            'timestamp': str(datetime.datetime.now()),
            'total_videos': total_videos,
            'processed': processed,
            'failed': failed
        }, Query().total_videos == total_videos)

    def generate_frame_tokens(self, image: np.ndarray) -> np.ndarray:
        """Generate tokens for a single frame"""
        try:
            # Convert BGR to RGB and to PIL Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transform and add batch dimension
            frame_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Generate token
            with torch.no_grad():
                token = self.model.latent_token_encoder(frame_tensor)
                
            return token.cpu().numpy()
            
        except Exception as e:
            print(f"Error generating token: {str(e)}")
            raise

    def extract_audio(self, video_path: str, output_folder: str) -> Dict:
        """Extract audio and save in chunks aligned with video frames"""
        try:
            # Create audio folder
            audio_folder = os.path.join(output_folder, 'audio')
            os.makedirs(audio_folder, exist_ok=True)

            # Extract audio using ffmpeg
            audio_path = os.path.join(audio_folder, 'audio.wav')
            subprocess.run([
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', str(self.audio_sample_rate),  # Sample rate
                '-ac', '1',  # Mono audio
                '-y',  # Overwrite output
                audio_path
            ], check=True)

            # Load audio file
            audio = AudioSegment.from_wav(audio_path)
            
            # Get audio metadata
            audio_metadata = {
                'sample_rate': self.audio_sample_rate,
                'channels': 1,
                'duration': len(audio) / 1000.0,  # Convert to seconds
                'chunk_duration': self.chunk_duration,
                'chunks': []
            }

            # Split audio into frame-aligned chunks
            chunk_samples = int(self.chunk_duration * self.audio_sample_rate)
            num_chunks = int(np.ceil(len(audio) / (self.chunk_duration * 1000)))

            print(f"Extracting {num_chunks} audio chunks...")
            for i in tqdm(range(num_chunks), desc="Processing audio chunks"):
                start_ms = i * self.chunk_duration * 1000
                end_ms = start_ms + (self.chunk_duration * 1000)
                chunk = audio[start_ms:end_ms]

                # Convert to numpy array and ensure correct size
                samples = np.array(chunk.get_array_of_samples())
                if len(samples) < chunk_samples:
                    # Pad with zeros if needed
                    samples = np.pad(samples, (0, chunk_samples - len(samples)))
                elif len(samples) > chunk_samples:
                    # Trim if too long
                    samples = samples[:chunk_samples]

                # Save chunk
                chunk_path = os.path.join(audio_folder, f'chunk_{i:06d}.npy')
                np.save(chunk_path, samples)
                
                # Add chunk info to metadata
                audio_metadata['chunks'].append({
                    'index': i,
                    'start_time': start_ms / 1000.0,
                    'duration': self.chunk_duration,
                    'samples': len(samples),
                    'path': os.path.relpath(chunk_path, output_folder)
                })

                # Save partial progress to database
                Progress = Query()
                self.progress_table.upsert({
                    'video_path': video_path,
                    'audio_chunks_processed': i + 1,
                    'total_chunks': num_chunks,
                    'last_updated': str(datetime.datetime.now())
                }, Progress.video_path == video_path)

            # Clean up WAV file if requested
            if os.path.exists(audio_path):
                os.remove(audio_path)

            # Save audio metadata
            audio_metadata_path = os.path.join(audio_folder, 'metadata.json')
            with open(audio_metadata_path, 'w') as f:
                json.dump(audio_metadata, f, indent=2)

            return audio_metadata

        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error processing {video_path}: {str(e)}"
            print(f"⚠️ {error_msg}")
            self.mark_video_failed(video_path, error_msg)
            raise

        except Exception as e:
            error_msg = f"Error extracting audio from {video_path}: {str(e)}"
            print(f"⚠️ {error_msg}")
            self.mark_video_failed(video_path, error_msg)
            
            # Clean up partial audio folder
            if os.path.exists(audio_folder):
                import shutil
                shutil.rmtree(audio_folder)
            
            raise


    def extract_frames(self, video_path: str, output_folder: str, 
                        frame_skip: int = 0, max_frames: int = None) -> Dict:
            """Extract frames, generate tokens, and return metadata with persistence"""
            try:
                # Create necessary folders
                os.makedirs(output_folder, exist_ok=True)
                tokens_folder = os.path.join(output_folder, 'tokens')
                os.makedirs(tokens_folder, exist_ok=True)
                
                # Check for existing progress
                video_hash = self.get_video_hash(video_path)
                Progress = Query()
                progress = self.progress_table.get(
                    (Progress.video_path == video_path) & 
                    (Progress.hash == video_hash)
                )
                
                # Initialize or load frame metadata
                metadata_path = os.path.join(output_folder, 'frames_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        frame_metadata = json.load(f)
                    last_processed_frame = max([f['index'] for f in frame_metadata['frames']]) if frame_metadata['frames'] else -1
                else:
                    # Open video and get properties
                    video = cv2.VideoCapture(video_path)
                    original_fps = video.get(cv2.CAP_PROP_FPS)
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_duration = 1.0 / self.frame_rate
                    
                    frame_metadata = {
                        'original_fps': float(original_fps),
                        'target_fps': self.frame_rate,
                        'frame_duration': frame_duration,
                        'total_frames': 0,
                        'frames': []
                    }
                    last_processed_frame = -1
                    video.release()

                # Process frames in batches
                batch_size = 32  # Adjust based on available memory
                video = cv2.VideoCapture(video_path)
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Skip to last processed frame
                if last_processed_frame >= 0:
                    video.set(cv2.CAP_PROP_POS_FRAMES, last_processed_frame + 1)
                
                success = True
                count = last_processed_frame + 1
                saved_count = len(frame_metadata['frames'])
                batch_frames = []
                batch_indices = []
                
                with tqdm(total=frame_count, initial=count,
                        desc=f"Processing {os.path.basename(video_path)}") as pbar:
                    
                    while success:
                        success, image = video.read()
                        if not success:
                            break
                            
                        if count % (frame_skip + 1) == 0:
                            frame_name = f"frame_{saved_count:06d}.png"
                            token_name = f"token_{saved_count:06d}.npy"
                            frame_path = os.path.join(output_folder, frame_name)
                            token_path = os.path.join(tokens_folder, token_name)
                            
                            # Add to batch
                            batch_frames.append(image)
                            batch_indices.append({
                                'index': saved_count,
                                'frame_path': frame_path,
                                'token_path': token_path
                            })
                            
                            # Process batch if full
                            if len(batch_frames) >= batch_size:
                                self._process_frame_batch(
                                    batch_frames, 
                                    batch_indices, 
                                    frame_metadata,
                                    frame_duration
                                )
                                
                                # Save progress
                                self._save_frame_progress(
                                    video_path, 
                                    video_hash,
                                    frame_metadata, 
                                    saved_count, 
                                    frame_count
                                )
                                
                                batch_frames = []
                                batch_indices = []
                            
                            saved_count += 1
                            if max_frames and saved_count >= max_frames:
                                break
                        
                        count += 1
                        pbar.update(1)
                    
                    # Process remaining batch
                    if batch_frames:
                        self._process_frame_batch(
                            batch_frames, 
                            batch_indices, 
                            frame_metadata,
                            frame_duration
                        )

                video.release()
                frame_metadata['total_frames'] = saved_count

                # Save final metadata
                with open(metadata_path, 'w') as f:
                    json.dump(frame_metadata, f, indent=2)

                # Update progress
                self.progress_table.upsert({
                    'video_path': video_path,
                    'hash': video_hash,
                    'status': 'frames_completed',
                    'frames_completed_at': str(datetime.datetime.now()),
                    'total_frames': saved_count
                }, (Progress.video_path == video_path) & (Progress.hash == video_hash))

                return frame_metadata

            except Exception as e:
                error_msg = f"Error extracting frames: {str(e)}"
                print(f"⚠️ {error_msg}")
                self.mark_video_failed(video_path, error_msg)
                raise

    def _process_frame_batch(self, batch_frames, batch_indices, frame_metadata, frame_duration):
        """Process a batch of frames and their tokens"""
        try:
            # Process frames in parallel
            for idx, (frame, info) in enumerate(zip(batch_frames, batch_indices)):
                # Save frame
                cv2.imwrite(info['frame_path'], frame)
                
                # Generate and save token
                token = self.generate_frame_tokens(frame)
                np.save(info['token_path'], token)
                
                # Add metadata
                frame_metadata['frames'].append({
                    'index': info['index'],
                    'timestamp': info['index'] * frame_duration,
                    'frame_path': os.path.relpath(info['frame_path'], os.path.dirname(info['frame_path'])),
                    'token_path': os.path.relpath(info['token_path'], os.path.dirname(info['frame_path'])),
                    'token_shape': token.shape
                })
                
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            raise

    def _save_frame_progress(self, video_path, video_hash, frame_metadata, current_frame, total_frames):
        """Save frame processing progress"""
        Progress = Query()
        self.progress_table.upsert({
            'video_path': video_path,
            'hash': video_hash,
            'status': 'frames_in_progress',
            'current_frame': current_frame,
            'total_frames': total_frames,
            'last_updated': str(datetime.datetime.now())
        }, (Progress.video_path == video_path) & (Progress.hash == video_hash))
            
    def get_processed_videos(self) -> set:
            """Get set of already processed video paths from output folder"""
            processed_videos = set()
            try:
                # Get list of folders in output directory
                for entry in os.scandir(self.output_base_folder):
                    if entry.is_dir():
                        metadata_path = os.path.join(entry.path, 'metadata.json')
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path, 'r') as f:
                                    metadata = json.load(f)
                                if metadata.get('video_path'):
                                    full_path = os.path.join(self.input_folder, metadata['video_path'])
                                    processed_videos.add(os.path.normpath(full_path))
                            except Exception as e:
                                print(f"Error reading metadata for {entry.path}: {e}")
                                continue
            except Exception as e:
                print(f"Error scanning output directory: {e}")
            
            print(f"Found {len(processed_videos)} already processed videos")
            return processed_videos

    def process_videos(self, max_videos: int = None, frame_skip: int = 0, 
                      max_frames: int = None) -> List[Dict]:
        """Process videos with persistence and progress tracking"""
        # First get list of already processed videos
        processed_videos = self.get_processed_videos()
        
        # Get video files that need processing
        print("Scanning for unprocessed videos...")
        video_files = []
        for root, _, files in os.walk(self.input_folder):
            for file in files:
                if file.endswith('.mp4'):
                    full_path = os.path.normpath(os.path.join(root, file))
                    # Only add if not already processed
                    if full_path not in processed_videos:
                        video_files.append(full_path)
                    if len(video_files) >= (max_videos or float('inf')):
                        break
            if len(video_files) >= (max_videos or float('inf')):
                break

        print(f"Found {len(video_files)} unprocessed videos")

        # Shuffle and limit if needed
        if max_videos and len(video_files) > max_videos:
            random.shuffle(video_files)
            video_files = video_files[:max_videos]

        # Process videos
        metadata_list = []
        total_videos = len(video_files)
        processed_count = 0
        failed_count = 0

        try:
            for video_path in tqdm(video_files, desc="Processing videos"):
                try:
                    relative_path = os.path.relpath(video_path, self.input_folder)
                    video_name = os.path.splitext(relative_path)[0]
                    output_folder = os.path.join(self.output_base_folder, video_name)

                    metadata = self.process_video(video_path, output_folder, frame_skip, max_frames)
                    
                    if metadata is not None:
                        metadata_list.append(metadata)
                        processed_count += 1
                    else:
                        failed_count += 1

                    # Save progress periodically
                    if (processed_count + failed_count) % 5 == 0:
                        self.save_progress(total_videos, processed_count, failed_count)

                except Exception as e:
                    print(f"Error processing {video_path}: {str(e)}")
                    failed_count += 1
                    continue

        finally:
                # Save final progress
                self.save_progress(total_videos, processed_count, failed_count)

                # Print summary
                print("\nProcessing Summary:")
                print(f"Total videos: {total_videos}")
                print(f"Successfully processed: {processed_count}")
                print(f"Failed: {failed_count}")

                # Update dataset metadata
                if metadata_list:
                    dataset_metadata = {
                        'videos': metadata_list,
                        'frame_rate': self.frame_rate,
                        'audio_sample_rate': self.audio_sample_rate,
                        'token_info': {
                            'model': 'IMFModel',
                            'shape': metadata_list[0]['token_shape'] if metadata_list else None
                        },
                        'processing_summary': {
                            'total_videos': total_videos,
                            'processed': processed_count,
                            'failed': failed_count,
                            'completed_at': str(datetime.datetime.now())
                        }
                    }

                    with open(os.path.join(self.output_base_folder, 'dataset.json'), 'w') as f:
                        json.dump(dataset_metadata, f, indent=2)

                return metadata_list
                
    def process_video(self, video_path: str, output_folder: str, 
                     frame_skip: int = 0, max_frames: int = None) -> Dict:
        """Process a single video with persistence"""
        video_hash = self.get_video_hash(video_path)
        
        # Check if already processed
        if self.is_video_processed(video_path):
            print(f"Skipping already processed video: {video_path}")
            Video = Query()
            result = self.processed_table.search(Video.hash == video_hash)
            return result[0]['metadata'] if result else None

        # Check if previously failed
        if self.is_video_failed(video_path):
            print(f"Skipping previously failed video: {video_path}")
            return None

        try:
            # Create output folders
            os.makedirs(output_folder, exist_ok=True)
            
            # Record start of processing
            self.progress_table.upsert({
                'video_path': video_path,
                'hash': video_hash,
                'status': 'started',
                'started_at': str(datetime.datetime.now())
            }, Query().hash == video_hash)
            
            # Try audio extraction first
            try:
                print(f"\nExtracting audio from {os.path.basename(video_path)}...")
                audio_metadata = self.extract_audio(video_path, output_folder)
                print("Audio extraction completed successfully")
                
                # Update progress
                self.progress_table.upsert({
                    'video_path': video_path,
                    'hash': video_hash,
                    'status': 'audio_completed',
                    'audio_completed_at': str(datetime.datetime.now())
                }, Query().hash == video_hash)
                
            except Exception as e:
                error_msg = f"Audio extraction failed: {str(e)}"
                self.mark_video_failed(video_path, error_msg)
                print(f"⚠️ {error_msg}")
                print(f"Skipping further processing for this video...")
                return None

            # If audio succeeded, proceed with frames and tokens
            try:
                print("\nProcessing frames and generating tokens...")
                frames_folder = os.path.join(output_folder, 'frames')
                os.makedirs(frames_folder, exist_ok=True)

                frame_metadata = self.extract_frames(video_path, frames_folder, frame_skip, max_frames)
                
                # Update progress
                self.progress_table.upsert({
                    'video_path': video_path,
                    'hash': video_hash,
                    'status': 'frames_completed',
                    'frames_completed_at': str(datetime.datetime.now())
                }, Query().hash == video_hash)

                # Save combined metadata
                metadata = {
                    'video_path': os.path.relpath(video_path, self.input_folder),
                    'frames': frame_metadata,
                    'audio': audio_metadata,
                    'token_shape': frame_metadata['frames'][0]['token_shape'] if frame_metadata['frames'] else None,
                    'processed_at': str(datetime.datetime.now())
                }

                metadata_path = os.path.join(output_folder, 'metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Mark as successfully processed
                self.mark_video_processed(video_path, metadata)
                print(f"\nSuccessfully processed {os.path.basename(video_path)}")
                return metadata

            except Exception as e:
                error_msg = f"Frame processing failed: {str(e)}"
                self.mark_video_failed(video_path, error_msg)
                print(f"Error processing frames: {str(e)}")
                
                # Clean up output folder
                if os.path.exists(output_folder):
                    import shutil
                    shutil.rmtree(output_folder)
                return None

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.mark_video_failed(video_path, error_msg)
            print(f"Error processing video {video_path}: {str(e)}")
            
            # Clean up output folder
            if os.path.exists(output_folder):
                import shutil
                shutil.rmtree(output_folder)
            return None
        

# Example usage
if __name__ == "__main__":
    processor = VideoProcessor(
        input_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/",
        output_base_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/processed_dataset",
        checkpoint_path="./checkpoints/checkpoint.pth"
    )
    
    metadata = processor.process_videos(
        max_videos=30,
        frame_skip=0,
        max_frames=1000
    )


# import concurrent.futures
# from queue import Queue
# import threading

# class VideoProcessor:
#     def __init__(self, input_folder: str, output_base_folder: str, checkpoint_path: str, 
#                  num_workers: int = 4):
#         # ... (existing initialization code) ...
        
#         self.num_workers = num_workers
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = self.model.to(self.device)
        
#         # Add thread-safe queues and locks
#         self.frame_queue = Queue(maxsize=100)
#         self.token_queue = Queue(maxsize=100)
#         self.metadata_lock = threading.Lock()
        
#     def _process_frame_worker(self, frame_metadata, frame_duration):
#         """Worker function to process frames and generate tokens"""
#         while True:
#             try:
#                 # Get frame data from queue
#                 frame_data = self.frame_queue.get()
#                 if frame_data is None:  # Poison pill
#                     break
                    
#                 frame, info = frame_data
                
#                 try:
#                     # Save frame
#                     cv2.imwrite(info['frame_path'], frame)
                    
#                     # Generate and save token
#                     token = self.generate_frame_tokens(frame)
#                     np.save(info['token_path'], token)
                    
#                     # Add metadata (thread-safe)
#                     with self.metadata_lock:
#                         frame_metadata['frames'].append({
#                             'index': info['index'],
#                             'timestamp': info['index'] * frame_duration,
#                             'frame_path': os.path.relpath(info['frame_path'], 
#                                                         os.path.dirname(info['frame_path'])),
#                             'token_path': os.path.relpath(info['token_path'], 
#                                                         os.path.dirname(info['frame_path'])),
#                             'token_shape': token.shape
#                         })
                        
#                 except Exception as e:
#                     print(f"Error processing frame {info['index']}: {str(e)}")
                    
#                 finally:
#                     self.frame_queue.task_done()
                    
#             except Exception as e:
#                 print(f"Worker error: {str(e)}")
#                 continue

#     def _process_frame_batch(self, batch_frames, batch_indices, frame_metadata, frame_duration):
#         """Process a batch of frames using thread pool"""
#         try:
#             # Start worker threads
#             workers = []
#             for _ in range(self.num_workers):
#                 thread = threading.Thread(
#                     target=self._process_frame_worker,
#                     args=(frame_metadata, frame_duration)
#                 )
#                 thread.daemon = True
#                 thread.start()
#                 workers.append(thread)
            
#             # Add frames to queue
#             for frame, info in zip(batch_frames, batch_indices):
#                 self.frame_queue.put((frame, info))
            
#             # Add poison pills
#             for _ in range(self.num_workers):
#                 self.frame_queue.put(None)
            
#             # Wait for all frames to be processed
#             self.frame_queue.join()
            
#             # Wait for workers to finish
#             for worker in workers:
#                 worker.join()
                
#         except Exception as e:
#             print(f"Error processing batch: {str(e)}")
#             raise

#     def process_videos(self, max_videos: int = None, frame_skip: int = 0, 
#                       max_frames: int = None) -> List[Dict]:
#         """Process multiple videos in parallel"""
#         video_files = []
#         for root, _, files in os.walk(self.input_folder):
#             for file in files:
#                 if file.endswith('.mp4'):
#                     video_files.append(os.path.join(root, file))

#         # Filter unprocessed videos
#         unprocessed_videos = [v for v in video_files if not self.is_video_processed(v)]
        
#         if max_videos:
#             random.shuffle(unprocessed_videos)
#             unprocessed_videos = unprocessed_videos[:max_videos]

#         total_videos = len(unprocessed_videos)
#         metadata_list = []
#         processed_count = 0
#         failed_count = 0

#         # Process videos with ThreadPoolExecutor
#         with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#             # Submit all video processing tasks
#             future_to_video = {
#                 executor.submit(
#                     self.process_video,
#                     video_path,
#                     os.path.join(
#                         self.output_base_folder,
#                         os.path.splitext(os.path.relpath(video_path, self.input_folder))[0]
#                     ),
#                     frame_skip,
#                     max_frames
#                 ): video_path for video_path in unprocessed_videos
#             }

#             # Process completed tasks as they finish
#             for future in tqdm(
#                 concurrent.futures.as_completed(future_to_video),
#                 total=len(unprocessed_videos),
#                 desc="Processing videos"
#             ):
#                 video_path = future_to_video[future]
#                 try:
#                     metadata = future.result()
#                     if metadata is not None:
#                         metadata_list.append(metadata)
#                         processed_count += 1
#                     else:
#                         failed_count += 1

#                     # Save progress periodically
#                     if (processed_count + failed_count) % 5 == 0:
#                         self.save_progress(total_videos, processed_count, failed_count)

#                 except Exception as e:
#                     print(f"Error processing {video_path}: {str(e)}")
#                     failed_count += 1

#         # Save final progress and dataset metadata
#         self.save_progress(total_videos, processed_count, failed_count)
        
#         if metadata_list:
#             dataset_metadata = {
#                 'videos': metadata_list,
#                 'frame_rate': self.frame_rate,
#                 'audio_sample_rate': self.audio_sample_rate,
#                 'token_info': {
#                     'model': 'IMFModel',
#                     'shape': metadata_list[0]['token_shape'] if metadata_list else None
#                 },
#                 'processing_summary': {
#                     'total_videos': total_videos,
#                     'processed': processed_count,
#                     'failed': failed_count,
#                     'completed_at': str(datetime.datetime.now())
#                 }
#             }
            
#             with open(os.path.join(self.output_base_folder, 'dataset.json'), 'w') as f:
#                 json.dump(dataset_metadata, f, indent=2)

#         return metadata_list

# # Usage example with specified number of workers
# if __name__ == "__main__":
#     processor = VideoProcessor(
#         input_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/",
#         output_base_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/processed_dataset",
#         checkpoint_path="./checkpoints/checkpoint.pth",
#         num_workers=4  # Adjust based on your CPU cores
#     )
    
#     metadata = processor.process_videos(
#         max_videos=100,
#         frame_skip=0,
#         max_frames=1000
#     )