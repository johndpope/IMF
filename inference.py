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

def process_video(model, video_path, output_path, transform, device, frame_skip=0):
    ctx = gpu(0) if torch.cuda.is_available() else cpu(0)
    vr = VideoReader(video_path, ctx=ctx)
    
    fps = vr.get_avg_fps()
    width, height = vr[0].shape[1], vr[0].shape[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

        reconstructed_frame = reconstructed_frame.squeeze().cpu().numpy().transpose(1, 2, 0)
        reconstructed_frame = (reconstructed_frame * 255).astype(np.uint8)
        reconstructed_frame = cv2.cvtColor(reconstructed_frame, cv2.COLOR_RGB2BGR)

        out.write(reconstructed_frame)

    out.release()

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

    process_video(model, config.input.video_path, config.output.path, transform, device, config.input.frame_skip)


if __name__ == "__main__":
    main()