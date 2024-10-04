import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from model import IMFModel
from omegaconf import OmegaConf
import numpy as np
from decord import VideoReader
from decord import cpu, gpu
import cv2

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_output(tensor, filename):
    save_image(tensor, filename, normalize=True)

def process_video(model, video_path, output_dir, transform, device, frame_skip=0):
    ctx = gpu(0) if torch.cuda.is_available() else cpu(0)
    vr = VideoReader(video_path, ctx=ctx)
    
    # Create output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    # Process reference frame
    reference_frame = vr[0].asnumpy()
    reference_frame = Image.fromarray(reference_frame)
    reference_frame = transform(reference_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        f_r = model.dense_feature_encoder(reference_frame)
        t_r = model.latent_token_encoder(reference_frame)

    total_frames = len(vr)
    for i in range(1, total_frames):
        if i % (frame_skip + 1) != 0:
            continue

        current_frame = vr[i].asnumpy()
        current_frame = Image.fromarray(current_frame)
        current_frame = transform(current_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            t_c = model.latent_token_encoder(current_frame)
            reconstructed_frame = model.decode_latent_tokens(f_r, t_r, t_c)

        # Convert the reconstructed frame to a PIL Image
        reconstructed_frame = reconstructed_frame.squeeze().cpu()
        reconstructed_frame = transforms.ToPILImage()(reconstructed_frame)
        
        # Save the reconstructed frame as an image
        output_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        reconstructed_frame.save(output_path)

    print(f"Processed {total_frames} frames. Output saved in {output_dir}")

def main():
    # Load configuration
    config = OmegaConf.load('./configs/inference.yaml')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = IMFModel(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
        use_resnet_feature=config.model.use_resnet_feature
    ).to(device)

    # Load the checkpoint
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    process_video(model, config.input.video_path, config.output.directory, transform, device, config.input.frame_skip)


if __name__ == "__main__":
    main()