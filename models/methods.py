import math
import sys
import pickle
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.psp.encoders import psp_encoders
from models.psp.stylegan2.model import Generator
from models.hyperinverter.stylegan2_ada import Discriminator 
from utils.class_registry import ClassRegistry
from utils.common_utils import get_keys
from utils.model_utils import toogle_grad
from configs.paths import DefaultPaths
from argparse import Namespace
from training.loggers import BaseTimer


sys.path.append("./utils")
methods_registry = ClassRegistry()


# @methods_registry.add_to_registry("fse_full", stop_args=("self", "checkpoint_path"))
class FSEFull(nn.Module):
    def __init__(self,
                 device="cuda:0",
                 paths=DefaultPaths,
                 checkpoint_path=None,
                 inverter_pth=None):
        super(FSEFull, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)

        self.device = device
        self.inverter_pth = inverter_pth

        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.latent_avg = None
        self.load_disc()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()


    def load_disc(self):
        # We used the hyperinverter discriminator since it has a cars checkpoint
        print("Loading default Discriminator from ", self.opts.stylegan_weights_pkl)
        with open(self.opts.stylegan_weights_pkl, "rb") as f:
            ckpt = pickle.load(f)

        D_original = ckpt["D"]
        D_original = D_original.float()

        self.discriminator = Discriminator(**D_original.init_kwargs)
        self.discriminator.load_state_dict(D_original.state_dict())
        self.discriminator.to(self.device)

    def load_disc_from_ckpt(self, ckpt):
        unique_keys = set(key.split(".")[0] for key in ckpt["state_dict"].keys())
        if "discriminator" in unique_keys:
            self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)
        else:
            print("Can not find Discriminator weights in checkpoint, leave default weights.")

    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print(f"Loading from checkpoint: {self.opts.checkpoint_path}")
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
            self.inverter.load_state_dict(get_keys(ckpt, "inverter"), strict=True)
        else:
            print(f"Loading Discriminator and Inverter from Inverter checkpoint: {self.inverter_pth}")
            ckpt = torch.load(self.inverter_pth, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            self.inverter.load_state_dict(get_keys(ckpt, "encoder"), strict=True)

        self.inverter = self.inverter.eval().to(self.device)
        toogle_grad(self.inverter, False)

        print("Loading Decoder from", self.opts.stylegan_weights)
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)
        self.decoder = self.decoder.eval().to(self.device)
        toogle_grad(self.decoder, False)

        print("Loading E4E from", self.opts.e4e_path)
        ckpt = torch.load(self.opts.e4e_path, map_location="cpu")
        self.e4e_encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)
        self.e4e_encoder = self.e4e_encoder.eval().to(self.device)
        toogle_grad(self.e4e_encoder, False)


    def set_encoder(self):
        self.inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18) 
        self.e4e_encoder = psp_encoders.Encoder4Editing(50, "ir_se", self.opts)
        feat_editor = psp_encoders.ContentLayerDeepFast(6, 1024, 512)
        return feat_editor  # trainable part
    
    def forward(self, x, return_latents=False, n_iter=1e5):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        with torch.no_grad():
            w_recon, predicted_feat = self.inverter.fs_backbone(x)
            w_recon = w_recon + self.latent_avg
                    
            _, w_feats = self.decoder(
                [w_recon],
                input_is_latent=True,
                return_features=True,
                is_stylespace=False,
                randomize_noise=False,
                early_stop=64
            )

            w_feat = w_feats[9]  # bs x 512 x 64 x 64 
            
            fused_feat = self.inverter.fuser(torch.cat([predicted_feat, w_feat], dim=1))
            delta = torch.zeros_like(fused_feat)  # inversion case

        edited_feat = self.encoder(torch.cat([fused_feat, delta], dim=1))
        feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)

        images, _ = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
            randomize_noise=False
        )

        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                predicted_feat = predicted_feat.cpu()
            return images, w_recon, fused_feat, predicted_feat
        return images


# @methods_registry.add_to_registry("fse_inverter", stop_args=("self", "checkpoint_path"))
class FSEInverter(nn.Module):
    def __init__(self,
                 device="cuda:0",
                 paths=DefaultPaths,
                 checkpoint_path=None):
        super(FSEInverter, self).__init__()
        self.opts = {
            "device": device,
            "checkpoint_path": checkpoint_path,
            "stylegan_size": 1024
        }
        self.opts.update(paths)
        self.opts = Namespace(**self.opts)

        self.device = device
        self.encoder = self.set_encoder()

        self.decoder = Generator(self.opts.stylegan_size, 512, 8)
        self.latent_avg = None
        self.load_disc()

        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()


    def load_disc(self):
        print("Loading default Discriminator from ", self.opts.stylegan_weights_pkl)
        # We used the hyperinverter discriminator since it has a cars checkpoint
        with open(self.opts.stylegan_weights_pkl, "rb") as f:
            ckpt = pickle.load(f)

        D_original = ckpt["D"]
        D_original = D_original.float()

        self.discriminator = Discriminator(**D_original.init_kwargs)
        self.discriminator.load_state_dict(D_original.state_dict())
        self.discriminator.to(self.device)

    def load_disc_from_ckpt(self, ckpt):
        unique_keys = set(key.split(".")[0] for key in ckpt["state_dict"].keys())
        if "discriminator" in unique_keys:
            self.discriminator.load_state_dict(get_keys(ckpt, "discriminator"), strict=True)
        else:
            print("Can not find Discriminator weights in checkpoint, leave default weights.")

    def load_weights(self):
        if self.opts.checkpoint_path != "":
            print("Loading  from checkpoint: {}".format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")
            self.load_disc_from_ckpt(ckpt)
            self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=True)

        print("Loading decoder from", self.opts.stylegan_weights)
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt["g_ema"], strict=False)
        self.latent_avg = ckpt['latent_avg'].to(self.device)

    def set_encoder(self):
        inverter = psp_encoders.Inverter(opts=self.opts, n_styles=18)
        return inverter  # trainable part
    
    def forward(self, x, return_latents=False, n_iter=1e5):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        w_recon, predicted_feat = self.encoder.fs_backbone(x)
        w_recon = w_recon + self.latent_avg
                
        _, w_feats = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            is_stylespace=False,
            randomize_noise=False,
            early_stop=64
        )

        w_feat = w_feats[9]  # bs x 512 x 64 x 64 
        fused_feat = self.encoder.fuser(torch.cat([predicted_feat, w_feat], dim=1))
        feats = [None] * 9 + [fused_feat] + [None] * (17 - 9)

        images, _ = self.decoder(
            [w_recon],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=min(1.0, 0.0001 * n_iter),
            is_stylespace=False,
            randomize_noise=False
        )
        
        if return_latents:
            if not self.encoder.training:
                fused_feat = fused_feat.cpu()
                w_feat = w_feat.cpu()
            return images, w_recon, fused_feat, w_feat
        return images
