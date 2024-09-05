import facer
from facer.face_parsing import FaRLFaceParser
from facer.face_detection import RetinaFaceDetector
from facer.face_detection.retinaface import RetinaFace
from configs.paths import DefaultPaths
import torch.backends.cudnn as cudnn
import numpy as np

import torch
from torch import nn

import torchvision.transforms as transforms

import torch.nn.functional as F

def my_fp_init(self, model_path=DefaultPaths.farl_path):
    super(FaRLFaceParser, self).__init__()
    self.conf_name = 'lapa/448'
    self.net = torch.jit.load(model_path)
    self.eval()

FaRLFaceParser.__init__ = my_fp_init

def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True

def load_model(model, pretrained_path, load_to_cpu, network: str):
    if load_to_cpu:
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage
            )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def load_net(model_path):
    cfg = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": True,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
    }
    # net and model
    net = RetinaFace(cfg=cfg, phase="test").cuda()
    net = load_model(net, model_path, True, network="mobilenet")
    net.eval()
    cudnn.benchmark = True
    # net = net.to(device)
    return net

def my_fd_init(self, model_path=DefaultPaths.mobile_net_pth, trash=0.8):
    super(RetinaFaceDetector, self).__init__()
    self.conf_name = 'mobilenet'
    self.threshold=trash
    self.net = load_net(model_path)
    self.eval()

RetinaFaceDetector.__init__ = my_fd_init

class TargetMask(nn.Module):
    def __init__(self, tfm=True):
        super().__init__()
        self.face_detector = RetinaFaceDetector(trash=0.8).cuda().eval()
        self.face_parser = FaRLFaceParser().cuda().eval()
        self.to_farl = transforms.Compose(
            [
                transforms.Normalize([0., 0., 0.], [2., 2., 2.]),
                transforms.Normalize([-0.5, -0.5, -0.5], [1., 1., 1.]),
            ]
        )
        self.tfm = tfm
        self.sigm = torch.nn.Sigmoid()

    def get_u_idxs(self, all_indexes):
            res = []
            for i in range(all_indexes[-1] + 1):
                res.append(((all_indexes == i).nonzero(as_tuple=True)[0][0]))
            return torch.tensor(res)

    def get_mask(self, y, threshold=0.5):
        #print(y.type(), y.shape, y.max(), y.min())
        y = y.long()
        faces = self.face_detector(y)
        #print(len(faces['image_ids']))
        faces = self.face_parser(y, faces)
        
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)

        uniq_idx = self.get_u_idxs(faces['image_ids'])

        chroma_mask = (seg_probs[uniq_idx, 0, :, :] >= threshold).to(y.dtype).unsqueeze(1)
        return chroma_mask

    def forward(self, x, y):
        if self.tfm:
            mask_y = self.get_mask(255. * self.to_farl(y))
        else:
            mask_y = self.get_mask(255. * y)
        return (1 - mask_y) * x + mask_y * y
        
    def forward2(self, x, y):
        batch = (255. * self.to_farl(y)).long()
        try:
            faces = self.face_detector(batch)
            assert len(faces['image_ids']) != 0
        except:
            for trash in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                try:
                    new_masker = Masker(trash=trash)
                    faces = new_masker.face_detector(batch)
                    assert len(faces['image_ids']) != 0
                    break
                except:
                    pass
        assert len(faces['image_ids']) != 0
        faces = self.face_parser(batch, faces)
        farl_mask = self.sigm(faces['seg']['logits'][:, 0])
        farl_mask = (farl_mask >= 0.995).float()[0]

        return (1 - farl_mask) * x + farl_mask * y


class Masker(nn.Module):
    def __init__(self, trash=0.8):
        super().__init__()
        self.face_detector = RetinaFaceDetector(trash=trash).cuda().eval()
        self.face_parser = FaRLFaceParser().cuda().eval()

    def get_u_idxs(self, all_indexes):
            res = []
            for i in range(all_indexes[-1] + 1):
                res.append(((all_indexes == i).nonzero(as_tuple=True)[0][0]))
            return torch.tensor(res)

    def get_mask(self, y, threshold=0.5):
        faces = self.face_detector(y)
        faces = self.face_parser(y, faces)
        
        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)

        uniq_idx = self.get_u_idxs(faces['image_ids'])

        chroma_mask = (seg_probs[uniq_idx, 0, :, :] >= threshold).to(y.dtype).unsqueeze(1)
        return chroma_mask

    def forward(self, x):
        return self.get_mask(255. * x).repeat(1, 3, 1, 1)
