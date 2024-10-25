import os
import cv2
from tqdm import tqdm
import random
import numpy as np
import subprocess
from pydub import AudioSegment
import json
from typing import Dict, Tuple, List

class VideoProcessor:
    def __init__(self, input_folder: str, output_base_folder: str):
        self.input_folder = input_folder
        self.output_base_folder = output_base_folder
        self.frame_rate = 24  # Target frame rate for consistency
        self.audio_sample_rate = 48000  # WebRTC standard
        self.chunk_duration = 1/24.0  # Duration of each audio chunk in seconds (matching frame rate)

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

            for i in range(num_chunks):
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

            # Save metadata
            with open(os.path.join(audio_folder, 'metadata.json'), 'w') as f:
                json.dump(audio_metadata, f, indent=2)

            return audio_metadata

        except Exception as e:
            print(f"Error extracting audio from {video_path}: {str(e)}")
            raise

    def extract_frames(self, video_path: str, output_folder: str, 
                      frame_skip: int = 0, max_frames: int = None) -> Dict:
        """Extract frames and return metadata"""
    
        os.makedirs(output_folder, exist_ok=True)
        
        # Open video
        video = cv2.VideoCapture(video_path)
        original_fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame timing
        frame_duration = 1.0 / self.frame_rate
        
        frame_metadata = {
            'original_fps': float(original_fps),
            'target_fps': self.frame_rate,
            'frame_duration': frame_duration,
            'total_frames': 0,
            'frames': []
        }

        success, image = video.read()
        count = 0
        saved_count = 0
        
        with tqdm(total=frame_count, desc=f"Processing {os.path.basename(video_path)}") as pbar:
            while success:
                if count % (frame_skip + 1) == 0:
                    frame_name = f"frame_{saved_count:06d}.png"
                    frame_path = os.path.join(output_folder, frame_name)
                    
                    # Save frame
                    cv2.imwrite(frame_path, image)
                    
                    # Add frame info to metadata
                    frame_metadata['frames'].append({
                        'index': saved_count,
                        'timestamp': saved_count * frame_duration,
                        'path': os.path.relpath(frame_path, output_folder)
                    })
                    
                    saved_count += 1
                    if max_frames and saved_count >= max_frames:
                        break
                
                count += 1
                success, image = video.read()
                pbar.update(1)

        video.release()
        frame_metadata['total_frames'] = saved_count

        # Save metadata
        with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
            json.dump(frame_metadata, f, indent=2)

        return frame_metadata

    def process_video(self, video_path: str, output_folder: str, 
                     frame_skip: int = 0, max_frames: int = None) -> Dict:
        """Process a single video, extracting both frames and audio"""
    
        # Create output folders
        frames_folder = os.path.join(output_folder, 'frames')
        os.makedirs(frames_folder, exist_ok=True)

        # Extract frames
        frame_metadata = self.extract_frames(
            video_path, frames_folder, 
            frame_skip, max_frames
        )

        # Extract audio
        audio_metadata = self.extract_audio(video_path, output_folder)

        # Save combined metadata
        metadata = {
            'video_path': os.path.relpath(video_path, self.input_folder),
            'frames': frame_metadata,
            'audio': audio_metadata
        }

        with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def process_videos(self, max_videos: int = None, frame_skip: int = 0, 
                      max_frames: int = None) -> List[Dict]:
        """Process all videos in the input folder"""
        # Get video files
        video_files = []
        for root, _, files in os.walk(self.input_folder):
            for file in files:
                if file.endswith('.mp4'):
                    video_files.append(os.path.join(root, file))

        # Shuffle and limit videos
        if max_videos:
            random.shuffle(video_files)
            video_files = video_files[:max_videos]

        # Process each video
        metadata_list = []
        for video_path in video_files:
            try:
                relative_path = os.path.relpath(video_path, self.input_folder)
                video_name = os.path.splitext(relative_path)[0]
                output_folder = os.path.join(self.output_base_folder, video_name)

                metadata = self.process_video(
                    video_path, output_folder,
                    frame_skip, max_frames
                )
                metadata_list.append(metadata)

            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                continue

        # Save dataset metadata
        dataset_metadata = {
            'videos': metadata_list,
            'frame_rate': self.frame_rate,
            'audio_sample_rate': self.audio_sample_rate
        }

        with open(os.path.join(self.output_base_folder, 'dataset.json'), 'w') as f:
            json.dump(dataset_metadata, f, indent=2)

        return metadata_list

# Usage example
if __name__ == "__main__":
    processor = VideoProcessor(
        input_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/",
        output_base_folder="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/processed_dataset"
    )
    
    metadata = processor.process_videos(
        max_videos=100,      # Process up to 100 videos
        frame_skip=0,        # Don't skip any frames
        max_frames=1000      # Up to 1000 frames per video
    )