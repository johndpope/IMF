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


import numpy as np
import torch
import torch.nn.functional as F
from models.hyperinverter.encoders.helpers import (
    bottleneck_IR,
    bottleneck_IR_SE,
    get_blocks,
)
from models.hyperinverter.stylegan2.model import EqualLinear
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential
from torchvision.models import resnet34
from time import time


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class E2Part(nn.Module):
    def __init__(self):
        super().__init__()
        self.styles = nn.ModuleList()
        self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = StyleBlock(512, 512, 1)
            elif i < self.middle_ind:
                style = StyleBlock(512, 512, 2)
            else:
                style = StyleBlock(512, 512, 3)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, c1, c2, c3):
        latents = []

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


# =============================================================================== #
# Official pSp backbone encoder


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        ]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


# =============================================================================== #


class StyleBlock(Module):
    def __init__(self, in_c, out_c, num_pools=1):
        super(StyleBlock, self).__init__()
        self.out_c = out_c
        modules = []
        modules += [
            Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        ]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        self.convs = nn.Sequential(*modules)

    def forward(self, x):
        x = self.convs(x)
        return x


class LayerWiseEncoder(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(LayerWiseEncoder, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = StyleBlock(512, 512, 1)
            elif i < self.middle_ind:
                style = StyleBlock(512, 512, 2)
            else:
                style = StyleBlock(512, 512, 3)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class LayerWiseEncoderWPlus(LayerWiseEncoder):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(LayerWiseEncoderWPlus, self).__init__(num_layers, mode, opts)
        self.heads = []
        for i in range(18):
            self.heads.append(nn.Linear(512 * 8 * 8, 512))
        self.heads = nn.ModuleList(self.heads)

    def body_forward(self, x):
        return super().forward(x)

    def forward(self, x):
        body_out = self.body_forward(x)
        final_latents = []
        for i in range(18):
            final_latents.append(self.heads[i](body_out[:, i]))
        out = torch.stack(final_latents, dim=1)
        return out


class ResNetLayerWiseEncoder(Module):
    """
    Original encoder architecture from pixel2style2pixel.
    This classes uses an FPN-based architecture applied over
    an ResNet34 backbone.
    """

    def __init__(self, opts=None):
        super(ResNetLayerWiseEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4,
        ]

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)

        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = StyleBlock(512, 512, 1)
            elif i < self.middle_ind:
                style = StyleBlock(512, 512, 2)
            else:
                style = StyleBlock(512, 512, 3)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 12:
                c2 = x
            elif i == 15:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class ResNetEncoderUsingLastLayerIntoW(Module):
    def __init__(self):
        super(ResNetEncoderUsingLastLayerIntoW, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4,
        ]

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)

        self.body = Sequential(*modules)

        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


from collections import namedtuple

import torch
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    MaxPool2d,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)


"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    else:
        raise ValueError(
            "Invalid number of layers: {}. Must be one of [50, 100, 152]".format(
                num_layers
            )
        )
    return blocks


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

from models.hyperinverter.encoders.helpers import (
    Flatten,
    bottleneck_IR,
    bottleneck_IR_SE,
    get_blocks,
    l2_norm,
)
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    Module,
    PReLU,
    Sequential,
)


"""
Modified Backbone implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode="ir", drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        if input_size == 112:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(drop_ratio),
                Flatten(),
                Linear(512 * 7 * 7, 512),
                BatchNorm1d(512, affine=affine),
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(drop_ratio),
                Flatten(),
                Linear(512 * 14 * 14, 512),
                BatchNorm1d(512, affine=affine),
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, num_layers=50, mode="ir", drop_ratio=0.4, affine=False)
    return model


def IR_101(input_size):
    """Constructs a ir-101 model."""
    model = Backbone(
        input_size, num_layers=100, mode="ir", drop_ratio=0.4, affine=False
    )
    return model


def IR_152(input_size):
    """Constructs a ir-152 model."""
    model = Backbone(
        input_size, num_layers=152, mode="ir", drop_ratio=0.4, affine=False
    )
    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(
        input_size, num_layers=50, mode="ir_se", drop_ratio=0.4, affine=False
    )
    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(
        input_size, num_layers=100, mode="ir_se", drop_ratio=0.4, affine=False
    )
    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(
        input_size, num_layers=152, mode="ir_se", drop_ratio=0.4, affine=False
    )
    return model

from collections import OrderedDict

import torch
import torch.nn as nn


def shape_to_num_params(shapes):
    return torch.sum(torch.tensor([torch.prod(s) for s in shapes])).int().item()


class WeightRegressor(nn.Module):
    """Regressing features to convolution weight kernel"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3, out_channels=16, in_channels=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels

        # Feature Transformer
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * self.input_dim, self.input_dim, kernel_size=1, padding=0, stride=1, bias=True),
            nn.InstanceNorm2d(self.input_dim),
            nn.ReLU(),
        )
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, self.hidden_dim, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        # Linear Mapper
        self.w1 = nn.Parameter(torch.randn((self.hidden_dim, self.in_channels * self.hidden_dim)))
        self.b1 = nn.Parameter(torch.randn((self.in_channels * self.hidden_dim)))
        self.w2 = nn.Parameter(torch.randn((self.hidden_dim, self.out_channels * self.kernel_size * self.kernel_size)))
        self.b2 = nn.Parameter(torch.randn((self.out_channels * self.kernel_size * self.kernel_size)))

        self.weight_init()

    def weight_init(self):
        nn.init.kaiming_normal_(self.w1)
        nn.init.zeros_(self.b1)
        nn.init.kaiming_normal_(self.w2)
        nn.init.zeros_(self.b2)

    def forward(self, w_image_codes, w_bar_codes):
        bs = w_image_codes.size(0)

        # Feature Transformation
        out = self.fusion(torch.cat((w_image_codes, w_bar_codes), 1))
        out = self.feature_extractor(out)
        out = out.view(bs, -1)

        # Linear map to weights
        out = torch.matmul(out, self.w1) + self.b1
        out = out.view(bs, self.in_channels, self.hidden_dim)
        out = torch.matmul(out, self.w2) + self.b2
        kernel = out.view(bs, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        return kernel


class Hypernetwork(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=64, target_shape=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.target_shape = target_shape

        num_predicted_weights = 0
        weight_regressors = OrderedDict()
        for layer_name in target_shape:
            new_layer_name = "_".join(layer_name.split("."))
            shape = target_shape[layer_name]["shape"]

            # without consider bias
            if len(shape) == 4:
                out_channels, in_channels, kernel_size = shape[:3]
            else:
                out_channels, in_channels = shape
                kernel_size = 1

            num_predicted_weights += shape_to_num_params([torch.tensor(list(shape))])
            weight_regressors[new_layer_name] = WeightRegressor(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                kernel_size=kernel_size,
                out_channels=out_channels,
                in_channels=in_channels,
            )
        self.weight_regressors = nn.ModuleDict(weight_regressors)
        self.num_predicted_weights = num_predicted_weights

    def forward(self, w_image_codes, w_bar_codes):
        bs = w_image_codes.size(0)
        out_weights = {}

        for layer_name in self.weight_regressors:
            ori_layer_name = ".".join(layer_name.split("_"))
            w_idx = self.target_shape[ori_layer_name]["w_idx"]
            weights = self.weight_regressors[layer_name](
                w_image_codes[:, w_idx, :, :, :], w_bar_codes[:, w_idx, :, :, :]
            )
            out_weights[ori_layer_name] = weights.view(bs, *list(self.target_shape[ori_layer_name]["shape"]))

        return out_weights
import torch


STYLEGAN2_ADA_CONV_WEIGHT_WITHOUT_BIAS_SHAPES = {
    "b4.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 0},
    "b4.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 1},
    "b8.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 2},
    "b8.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 3},
    "b8.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 3},
    "b16.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 4},
    "b16.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 5},
    "b16.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 5},
    "b32.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 6},
    "b32.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 7},
    "b32.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 7},
    "b64.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 8},
    "b64.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 9},
    "b64.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 9},
    "b128.conv0.weight": {"shape": torch.Size([256, 512, 3, 3]), "w_idx": 10},
    "b128.conv1.weight": {"shape": torch.Size([256, 256, 3, 3]), "w_idx": 11},
    "b128.torgb.weight": {"shape": torch.Size([3, 256, 1, 1]), "w_idx": 11},
    "b256.conv0.weight": {"shape": torch.Size([128, 256, 3, 3]), "w_idx": 12},
    "b256.conv1.weight": {"shape": torch.Size([128, 128, 3, 3]), "w_idx": 13},
    "b256.torgb.weight": {"shape": torch.Size([3, 128, 1, 1]), "w_idx": 13},
    "b512.conv0.weight": {"shape": torch.Size([64, 128, 3, 3]), "w_idx": 14},
    "b512.conv1.weight": {"shape": torch.Size([64, 64, 3, 3]), "w_idx": 15},
    "b512.torgb.weight": {"shape": torch.Size([3, 64, 1, 1]), "w_idx": 15},
    "b1024.conv0.weight": {"shape": torch.Size([32, 64, 3, 3]), "w_idx": 16},
    "b1024.conv1.weight": {"shape": torch.Size([32, 32, 3, 3]), "w_idx": 17},
    "b1024.torgb.weight": {"shape": torch.Size([3, 32, 1, 1]), "w_idx": 17},
}

STYLEGAN2_ADA_CONV_WEIGHT_WITHOUT_BIAS_WITHOUT_TO_RGB_SHAPES = {
    "b4.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 0},
    "b8.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 2},
    "b8.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 3},
    "b16.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 4},
    "b16.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 5},
    "b32.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 6},
    "b32.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 7},
    "b64.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 8},
    "b64.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 9},
    "b128.conv0.weight": {"shape": torch.Size([256, 512, 3, 3]), "w_idx": 10},
    "b128.conv1.weight": {"shape": torch.Size([256, 256, 3, 3]), "w_idx": 11},
    "b256.conv0.weight": {"shape": torch.Size([128, 256, 3, 3]), "w_idx": 12},
    "b256.conv1.weight": {"shape": torch.Size([128, 128, 3, 3]), "w_idx": 13},
    "b512.conv0.weight": {"shape": torch.Size([64, 128, 3, 3]), "w_idx": 14},
    "b512.conv1.weight": {"shape": torch.Size([64, 64, 3, 3]), "w_idx": 15},
    "b1024.conv0.weight": {"shape": torch.Size([32, 64, 3, 3]), "w_idx": 16},
    "b1024.conv1.weight": {"shape": torch.Size([32, 32, 3, 3]), "w_idx": 17},
}

STYLEGAN2_ADA_ALL_WEIGHT_WITHOUT_BIAS_SHAPES = {
    "b4.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 0},
    "b4.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 1},
    "b8.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 2},
    "b8.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 3},
    "b8.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 3},
    "b16.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 4},
    "b16.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 5},
    "b16.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 5},
    "b32.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 6},
    "b32.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 7},
    "b32.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 7},
    "b64.conv0.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 8},
    "b64.conv1.weight": {"shape": torch.Size([512, 512, 3, 3]), "w_idx": 9},
    "b64.torgb.weight": {"shape": torch.Size([3, 512, 1, 1]), "w_idx": 9},
    "b128.conv0.weight": {"shape": torch.Size([256, 512, 3, 3]), "w_idx": 10},
    "b128.conv1.weight": {"shape": torch.Size([256, 256, 3, 3]), "w_idx": 11},
    "b128.torgb.weight": {"shape": torch.Size([3, 256, 1, 1]), "w_idx": 11},
    "b256.conv0.weight": {"shape": torch.Size([128, 256, 3, 3]), "w_idx": 12},
    "b256.conv1.weight": {"shape": torch.Size([128, 128, 3, 3]), "w_idx": 13},
    "b256.torgb.weight": {"shape": torch.Size([3, 128, 1, 1]), "w_idx": 13},
    "b512.conv0.weight": {"shape": torch.Size([64, 128, 3, 3]), "w_idx": 14},
    "b512.conv1.weight": {"shape": torch.Size([64, 64, 3, 3]), "w_idx": 15},
    "b512.torgb.weight": {"shape": torch.Size([3, 64, 1, 1]), "w_idx": 15},
    "b1024.conv0.weight": {"shape": torch.Size([32, 64, 3, 3]), "w_idx": 16},
    "b1024.conv1.weight": {"shape": torch.Size([32, 32, 3, 3]), "w_idx": 17},
    "b1024.torgb.weight": {"shape": torch.Size([3, 32, 1, 1]), "w_idx": 17},
    # affine
    "b4.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 0},
    "b4.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 1},
    "b8.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 2},
    "b8.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 3},
    "b8.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 3},
    "b16.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 4},
    "b16.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 5},
    "b16.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 5},
    "b32.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 6},
    "b32.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 7},
    "b32.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 7},
    "b64.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 8},
    "b64.conv1.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 9},
    "b64.torgb.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 9},
    "b128.conv0.affine.weight": {"shape": torch.Size([512, 512]), "w_idx": 10},
    "b128.conv1.affine.weight": {"shape": torch.Size([256, 512]), "w_idx": 11},
    "b128.torgb.affine.weight": {"shape": torch.Size([256, 512]), "w_idx": 11},
    "b256.conv0.affine.weight": {"shape": torch.Size([256, 512]), "w_idx": 12},
    "b256.conv1.affine.weight": {"shape": torch.Size([128, 512]), "w_idx": 13},
    "b256.torgb.affine.weight": {"shape": torch.Size([128, 512]), "w_idx": 13},
    "b512.conv0.affine.weight": {"shape": torch.Size([128, 512]), "w_idx": 14},
    "b512.conv1.affine.weight": {"shape": torch.Size([64, 512]), "w_idx": 15},
    "b512.torgb.affine.weight": {"shape": torch.Size([64, 512]), "w_idx": 15},
    "b1024.conv0.affine.weight": {"shape": torch.Size([64, 512]), "w_idx": 16},
    "b1024.conv1.affine.weight": {"shape": torch.Size([32, 512]), "w_idx": 17},
    "b1024.torgb.affine.weight": {"shape": torch.Size([32, 512]), "w_idx": 17},
}

from .fused_act import FusedLeakyReLU, fused_leaky_relu
from .upfirdn2d import upfirdn2d


__all__ = ["FusedLeakyReLU", "fused_leaky_relu", "upfirdn2d"]

import os

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
fused = load(
    "fused",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused_bias_act_kernel.cu"),
    ],
)


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        (out,) = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)

import os

import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
upfirdn2d_op = load(
    "upfirdn2d",
    sources=[
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ],
)


class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        (kernel,) = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = UpFirDn2d.apply(
        input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
    )

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)

    return out[:, ::down_y, ::down_x, :]

import math
import random

import torch
from models.hyperinverter.stylegan2.op import (
    FusedLeakyReLU,
    fused_leaky_relu,
    upfirdn2d,
)
from torch import nn
from torch.nn import functional as F


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2**2, 2**2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        return_features=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_features:
            return image, out
        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


""" Modify StyleGAN2-Ada generator to allow forward pass receiving predicted weights """

import numpy as np
import torch
from utils.torch_utils import misc, persistence
from utils.torch_utils.ops import bias_act, conv2d_resample, fma, upfirdn2d


# ----------------------------------------------------------------------------


def get_added_weights(added_weights_dict, key):
    if (added_weights_dict is None) or (key not in added_weights_dict):
        return None
    else:
        return added_weights_dict[key]


@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------


@misc.profiled_function
def modulated_conv2d(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,  # Modulation coefficients of shape [batch_size, in_channels].
    noise=None,  # Optional noise tensor to add to the output activations.
    up=1,  # Integer upsampling factor.
    down=1,  # Integer downsampling factor.
    padding=0,  # Padding with respect to the upsampled image.
    resample_filter=None,  # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate=True,  # Apply weight demodulation?
    flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (
            1
            / np.sqrt(in_channels * kh * kw)
            / weight.norm(float("inf"), dim=[1, 2, 3], keepdim=True)
        )  # max_Ikk
        styles = styles / styles.norm(float("inf"), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=weight.to(x.dtype),
            f=resample_filter,
            up=up,
            down=down,
            padding=padding,
            flip_weight=flip_weight,
        )
        if demodulate and noise is not None:
            x = fma.fma(
                x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype)
            )
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(
        x=x,
        w=w.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=batch_size,
        flip_weight=flip_weight,
    )
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,  # Learning rate multiplier.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x, added_weights=None):
        aws = get_added_weights(added_weights, "weight")
        w = (
            self.weight.to(x.dtype) + aws.to(x.dtype)
            if aws is not None
            else self.weight.to(x.dtype)
        ) * self.weight_gain
        aws = get_added_weights(added_weights, "bias")
        b = self.bias + aws if aws is not None else self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
        channels_last=False,  # Expect the input to have memory_format=channels_last?
        trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
            memory_format=memory_format
        )
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = self.up == 1  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = (
            [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False
    ):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(
                    x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
                )

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi
                    )
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this layer.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
        use_noise=True,  # Enable noise input?
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last=False,  # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
                memory_format=memory_format
            )
        )
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(
        self, x, w, added_weights=None, noise_mode="random", fused_modconv=True, gain=1, is_stylespace=False
    ):
        #print("SYNT LAYER")
        #print(w.shape)
        assert noise_mode in ["random", "const", "none"]
        in_resolution = self.resolution // self.up
        if is_stylespace:
            styles = w
            #print("STYLES", styles.shape)
        else:
            misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
            styles = self.affine(w, get_added_weights(added_weights, "affine"))
        #t(styles.shape)

        noise = None
        aws = get_added_weights(added_weights, "noise_strength")
        assert aws is None
        if self.use_noise and noise_mode == "random":
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution], device=x.device
            ) * (self.noise_strength + aws if aws is not None else self.noise_strength)
        if self.use_noise and noise_mode == "const":
            noise = self.noise_const * (
                self.noise_strength + aws if aws is not None else self.noise_strength
            )

        flip_weight = self.up == 1  # slightly faster
        aws = get_added_weights(added_weights, "weight")
        conv_weight = self.weight
        if aws is not None:
            if isinstance(aws, tuple):
                conv_weight = self.weight * (aws[0].to(x.dtype) + 1) + aws[1].to(x.dtype)
            else:
                conv_weight = self.weight + aws.to(x.dtype)

        x = modulated_conv2d(
            x=x,
            weight=conv_weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        aws = get_added_weights(added_weights, "bias")
        conv_bias = self.bias.to(x.dtype)
        if aws is not None:
            if isinstance(aws, tuple):
                conv_bias = self.bias.to(x.dtype) * (aws[0].to(x.dtype) + 1) + aws[1].to(x.dtype)
            else:
                conv_bias = self.bias.to(x.dtype) + aws.to(x.dtype)
        x = bias_act.bias_act(
            x,
            conv_bias,
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
        )
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        kernel_size=1,
        conv_clamp=None,
        channels_last=False,
    ):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
                memory_format=memory_format
            )
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w, added_weights=None, fused_modconv=True, is_stylespace=False):
        if is_stylespace:
            styles = w * self.weight_gain
        else:
            styles = (
                self.affine(w, get_added_weights(added_weights, "affine"))
                * self.weight_gain
            )
        aws = get_added_weights(added_weights, "weight")
        conv_weight = self.weight
        if aws is not None:
            if isinstance(aws, tuple):
                conv_weight = self.weight * (aws[0].to(x.dtype) + 1) + aws[1].to(x.dtype)
            else:
                conv_weight = self.weight + aws.to(x.dtype)
        x = modulated_conv2d(
            x=x,
            weight=conv_weight,
            styles=styles,
            demodulate=False,
            fused_modconv=fused_modconv,
        )
        aws = get_added_weights(added_weights, "bias")
        conv_bias = self.bias.to(x.dtype)
        if aws is not None:
            if isinstance(aws, tuple):
                conv_bias = self.bias.to(x.dtype) * (aws[0].to(x.dtype) + 1) + aws[1].to(x.dtype)
            else:
                conv_bias = self.bias.to(x.dtype) + aws.to(x.dtype)

        x = bias_act.bias_act(
            x,
            conv_bias,
            clamp=self.conv_clamp,
        )
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this block.
        img_channels,  # Number of output color channels.
        is_last,  # Is this the last block?
        architecture="skip",  # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(
                torch.randn([out_channels, resolution, resolution])
            )

        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                up=2,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                **layer_kwargs,
            )
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            **layer_kwargs,
        )
        self.num_conv += 1

        if is_last or architecture == "skip":
            self.torgb = ToRGBLayer(
                out_channels,
                img_channels,
                w_dim=w_dim,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )
            self.num_torgb += 1

        if in_channels != 0 and architecture == "resnet":
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=2,
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(
        self,
        x,
        img,
        ws,
        added_weights=None,
        force_fp32=False,
        fused_modconv=None,
        is_stylespace=False,
        **layer_kwargs,
    ):
        if is_stylespace:
            w_iter = iter(ws)
        else:
            misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
            w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )
        if fused_modconv is None:
            with misc.suppress_tracer_warnings():  # this value will be treated as a constant
                fused_modconv = (not self.training) and (
                    dtype == torch.float32 or int(x.shape[0]) == 1
                )

        # Input.
        if is_stylespace:
            bs = ws[0].shape[0]
        else:
            bs = ws.shape[0]
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([bs, 1, 1, 1])
        else:
            misc.assert_shape(
                x, [None, self.in_channels, self.resolution // 2, self.resolution // 2]
            )
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(
                x,
                next(w_iter),
                added_weights=get_added_weights(added_weights, "conv1"),
                fused_modconv=fused_modconv,
                is_stylespace=is_stylespace,
                **layer_kwargs,
            )
        elif self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(
                x,
                next(w_iter),
                added_weights=get_added_weights(added_weights, "conv0"),
                fused_modconv=fused_modconv,
                is_stylespace=is_stylespace,
                **layer_kwargs,
            )
            x = self.conv1(
                x,
                next(w_iter),
                added_weights=get_added_weights(added_weights, "conv1"),
                fused_modconv=fused_modconv,
                gain=np.sqrt(0.5),
                is_stylespace=is_stylespace,
                **layer_kwargs,
            )
            x = y.add_(x)
        else:
            x = self.conv0(
                x,
                next(w_iter),
                added_weights=get_added_weights(added_weights, "conv0"),
                fused_modconv=fused_modconv,
                is_stylespace=is_stylespace,
                **layer_kwargs,
            )
            x = self.conv1(
                x,
                next(w_iter),
                added_weights=get_added_weights(added_weights, "conv1"),
                fused_modconv=fused_modconv,
                is_stylespace=is_stylespace,
                **layer_kwargs,
            )

        # ToRGB.
        if img is not None:
            misc.assert_shape(
                img,
                [None, self.img_channels, self.resolution // 2, self.resolution // 2],
            )
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == "skip":
            y = self.torgb(
                x,
                next(w_iter),
                added_weights=get_added_weights(added_weights, "torgb"),
                fused_modconv=fused_modconv,
                is_stylespace=is_stylespace,
            )
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output image resolution.
        img_channels,  # Number of color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max) for res in self.block_resolutions
        }
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            is_last = res == self.img_resolution
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f"b{res}", block)

    def forward(self, ws, added_weights=None, is_stylespace=False, early_stop=None,  **block_kwargs):
        block_ws = []
        if is_stylespace:
            styles, rgb = ws
            block_ws = []
            i = 0
            for j in range(len(rgb)):
                block_ws.append([styles[i]])
                i += 1
                if j != 0:
                    block_ws[-1].append(styles[i])
                    i += 1
                block_ws[-1].append(rgb[j])

            ######
            # for block in block_ws:
            #     print(" ")
            #     for tmp in block:
            #         print(tmp.shape)
            #
            # print('----------------')
        else:
            with torch.autograd.profiler.record_function("split_ws"):
                misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in self.block_resolutions:
                    block = getattr(self, f"b{res}")
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f"b{res}")
            aws = get_added_weights(added_weights, f"b{res}")
            x, img = block(x, img, cur_ws, added_weights=aws, is_stylespace=is_stylespace, **block_kwargs)
            if early_stop and early_stop == res:
                return img
        return img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(
            z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs
        )

    def forward(
        self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs
    ):
        ws = self.mapping(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
        )
        img = self.synthesis(ws, **synthesis_kwargs)
        return img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        tmp_channels,  # Number of intermediate channels.
        out_channels,  # Number of output channels.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        first_layer_idx,  # Index of the first layer.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        freeze_layers=0,  # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = layer_idx >= freeze_layers
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels,
                tmp_channels,
                kernel_size=1,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )

        self.conv0 = Conv2dLayer(
            tmp_channels,
            tmp_channels,
            kernel_size=3,
            activation=activation,
            trainable=next(trainable_iter),
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = Conv2dLayer(
            tmp_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
            trainable=next(trainable_iter),
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if architecture == "resnet":
            self.skip = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )

        # Input.
        if x is not None:
            misc.assert_shape(
                x, [None, self.in_channels, self.resolution, self.resolution]
            )
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == "skip":
            misc.assert_shape(
                img, [None, self.img_channels, self.resolution, self.resolution]
            )
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = (
                upfirdn2d.downsample2d(img, self.resample_filter)
                if self.architecture == "skip"
                else None
            )

        # Main layers.
        if self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = (
                torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
                if self.group_size is not None
                else N
            )
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels, in_channels, kernel_size=1, activation=activation
            )
        self.mbstd = (
            MinibatchStdLayer(
                group_size=mbstd_group_size, num_channels=mbstd_num_channels
            )
            if mbstd_num_channels > 0
            else None
        )
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels,
            in_channels,
            kernel_size=3,
            activation=activation,
            conv_clamp=conv_clamp,
        )
        self.fc = FullyConnectedLayer(
            in_channels * (resolution**2), in_channels, activation=activation
        )
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(
            x, [None, self.in_channels, self.resolution, self.resolution]
        )  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == "skip":
            misc.assert_shape(
                img, [None, self.img_channels, self.resolution, self.resolution]
            )
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        block_kwargs={},  # Arguments for DiscriminatorBlock.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            2**i for i in range(self.img_resolution_log2, 2, -1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions + [4]
        }
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(
            img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp
        )
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = res >= fp16_resolution
            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                **block_kwargs,
                **common_kwargs,
            )
            setattr(self, f"b{res}", block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=0,
                c_dim=c_dim,
                w_dim=cmap_dim,
                num_ws=None,
                w_avg_beta=None,
                **mapping_kwargs,
            )
        self.b4 = DiscriminatorEpilogue(
            channels_dict[4],
            cmap_dim=cmap_dim,
            resolution=4,
            **epilogue_kwargs,
            **common_kwargs,
        )

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

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


@methods_registry.add_to_registry("fse_full", stop_args=("self", "checkpoint_path"))
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


@methods_registry.add_to_registry("fse_inverter", stop_args=("self", "checkpoint_path"))
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

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:43:29 2017
@author: zhaoy
"""
import numpy as np
import cv2

# from scipy.linalg import lstsq
# from scipy.ndimage import geometric_transform  # , map_coordinates

from models.mtcnn.mtcnn_pytorch.src.matlab_cp2tform import (
    get_similarity_transform_for_cv2,
)

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156],
]

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):
    def __str__(self):
        return "In File {}:{}".format(__file__, super.__str__(self))


def get_reference_facial_points(
    output_size=None,
    inner_padding_factor=0.0,
    outer_padding=(0, 0),
    default_square=False,
):
    """
    Function:
    ----------
        get reference 5 key points according to crop settings:
        0. Set default crop_size:
            if default_square:
                crop_size = (112, 112)
            else:
                crop_size = (96, 112)
        1. Pad the crop_size by inner_padding_factor in each side;
        2. Resize crop_size into (output_size - outer_padding*2),
            pad into output_size with outer_padding;
        3. Output reference_5point;
    Parameters:
    ----------
        @output_size: (w, h) or None
            size of aligned face image
        @inner_padding_factor: (w_factor, h_factor)
            padding factor for inner (w, h)
        @outer_padding: (w_pad, h_pad)
            each row is a pair of coordinates (x, y)
        @default_square: True or False
            if True:
                default crop_size = (112, 112)
            else:
                default crop_size = (96, 112);
        !!! make sure, if output_size is not None:
                (output_size - outer_padding)
                = some_scale * (default crop_size * (1.0 + inner_padding_factor))
    Returns:
    ----------
        @reference_5point: 5x2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    # print('\n===> get_reference_facial_points():')

    # print('---> Params:')
    # print('            output_size: ', output_size)
    # print('            inner_padding_factor: ', inner_padding_factor)
    # print('            outer_padding:', outer_padding)
    # print('            default_square: ', default_square)

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # 0) make the inner region a square
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    # print('---> default:')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    if (
        output_size
        and output_size[0] == tmp_crop_size[0]
        and output_size[1] == tmp_crop_size[1]
    ):
        # print('output_size == DEFAULT_CROP_SIZE {}: return default reference points'.format(tmp_crop_size))
        return tmp_5pts

    if inner_padding_factor == 0 and outer_padding == (0, 0):
        if output_size is None:
            # print('No paddings to do: return default reference points')
            return tmp_5pts
        else:
            raise FaceWarpException(
                "No paddings to do, output_size must be None or {}".format(
                    tmp_crop_size
                )
            )

    # check output size
    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException("Not (0 <= inner_padding_factor <= 1.0)")

    if (
        inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0
    ) and output_size is None:
        output_size = tmp_crop_size * (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)
        # print('              deduced from paddings, output_size = ', output_size)

    if not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1]):
        raise FaceWarpException(
            "Not (outer_padding[0] < output_size[0]"
            "and outer_padding[1] < output_size[1])"
        )

    # 1) pad the inner region according inner_padding_factor
    # print('---> STEP1: pad the inner region according inner_padding_factor')
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 2) resize the padded inner region
    # print('---> STEP2: resize the padded inner region')
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    # print('              crop_size = ', tmp_crop_size)
    # print('              size_bf_outer_pad = ', size_bf_outer_pad)

    if (
        size_bf_outer_pad[0] * tmp_crop_size[1]
        != size_bf_outer_pad[1] * tmp_crop_size[0]
    ):
        raise FaceWarpException(
            "Must have (output_size - outer_padding)"
            "= some_scale * (crop_size * (1.0 + inner_padding_factor)"
        )

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    # print('              resize scale_factor = ', scale_factor)
    tmp_5pts = tmp_5pts * scale_factor
    #    size_diff = tmp_crop_size * (scale_factor - min(scale_factor))
    #    tmp_5pts = tmp_5pts + size_diff / 2
    tmp_crop_size = size_bf_outer_pad
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # 3) add outer_padding to make output_size
    reference_5point = tmp_5pts + np.array(outer_padding)
    tmp_crop_size = output_size
    # print('---> STEP3: add outer_padding to make output_size')
    # print('              crop_size = ', tmp_crop_size)
    # print('              reference_5pts = ', tmp_5pts)

    # print('===> end get_reference_facial_points\n')

    return reference_5point


def get_affine_transform_matrix(src_pts, dst_pts):
    """
    Function:
    ----------
        get affine transform matrix 'tfm' from src_pts to dst_pts
    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points matrix, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points matrix, each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @tfm: 2x3 np.array
            transform matrix from src_pts to dst_pts
    """

    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    #    #print(('src_pts_:\n' + str(src_pts_))
    #    #print(('dst_pts_:\n' + str(dst_pts_))

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    #    #print(('np.linalg.lstsq return A: \n' + str(A))
    #    #print(('np.linalg.lstsq return res: \n' + str(res))
    #    #print(('np.linalg.lstsq return rank: \n' + str(rank))
    #    #print(('np.linalg.lstsq return s: \n' + str(s))

    if rank == 3:
        tfm = np.float32([[A[0, 0], A[1, 0], A[2, 0]], [A[0, 1], A[1, 1], A[2, 1]]])
    elif rank == 2:
        tfm = np.float32([[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0]])

    return tfm


def warp_and_crop_face(
    src_img, facial_pts, reference_pts=None, crop_size=(96, 112), align_type="smilarity"
):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type: transform type, could be one of
            1) 'similarity': use similarity transform
            2) 'cv2_affine': use the first 3 points to do affine transform,
                    by calling cv2.getAffineTransform()
            3) 'affine': use all points to do affine transform
    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """

    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        else:
            default_square = False
            inner_padding_factor = 0
            outer_padding = (0, 0)
            output_size = crop_size

            reference_pts = get_reference_facial_points(
                output_size, inner_padding_factor, outer_padding, default_square
            )

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException("reference_pts.shape must be (K,2) or (2,K) and K>2")

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException("facial_pts.shape must be (K,2) or (2,K) and K>2")

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    #    #print('--->src_pts:\n', src_pts
    #    #print('--->ref_pts\n', ref_pts

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException("facial_pts and reference_pts must have the same shape")

    if align_type is "cv2_affine":
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    #        #print(('cv2.getAffineTransform() returns tfm=\n' + str(tfm))
    elif align_type is "affine":
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    #        #print(('get_affine_transform_matrix() returns tfm=\n' + str(tfm))
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
    #        #print(('get_similarity_transform_for_cv2() returns tfm=\n' + str(tfm))

    #    #print('--->Transform matrix: '
    #    #print(('type(tfm):' + str(type(tfm)))
    #    #print(('tfm.dtype:' + str(tfm.dtype))
    #    #print( tfm

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img, tfm

import numpy as np
from PIL import Image


def nms(boxes, overlap_threshold=0.5, mode="union"):
    """Non-maximum suppression.

    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.

    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == "min":
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == "union":
            # intersection over union (IoU)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids, np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].

    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
    """

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes


def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].

    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # are offsets always such that
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h]) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes


def get_image_boxes(bounding_boxes, img, size=24):
    """Cut out boxes from the image.

    Arguments:
        bounding_boxes: a float numpy array of shape [n, 5].
        img: an instance of PIL.Image.
        size: an integer, size of cutouts.

    Returns:
        a float numpy array of shape [n, 3, size, size].
    """

    num_boxes = len(bounding_boxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(
        bounding_boxes, width, height
    )
    img_boxes = np.zeros((num_boxes, 3, size, size), "float32")

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), "uint8")

        img_array = np.asarray(img, "uint8")
        img_box[dy[i] : (edy[i] + 1), dx[i] : (edx[i] + 1), :] = img_array[
            y[i] : (ey[i] + 1), x[i] : (ex[i] + 1), :
        ]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, "float32")

        img_boxes[i, :, :, :] = _preprocess(img_box)

    return img_boxes


def correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts.

    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.

    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.

        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype("int32") for i in return_list]

    return return_list


def _preprocess(img):
    """Preprocessing step before feeding the network.

    Arguments:
        img: a float numpy array of shape [h, w, c].

    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from configs.paths import DefaultPaths
from pathlib import Path
import numpy as np

PNET_PATH = Path(DefaultPaths.mtcnn) / "pnet.npy"
ONET_PATH =  Path(DefaultPaths.mtcnn) / "onet.npy"
RNET_PATH =  Path(DefaultPaths.mtcnn) / "rnet.npy"


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class PNet(nn.Module):
    def __init__(self):
        super().__init__()

        # suppose we have input with size HxW, then
        # after first layer: H - 2,
        # after pool: ceil((H - 2)/2),
        # after second conv: ceil((H - 2)/2) - 2,
        # after last conv: ceil((H - 2)/2) - 4,
        # and the same for W

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 10, 3, 1)),
                    ("prelu1", nn.PReLU(10)),
                    ("pool1", nn.MaxPool2d(2, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(10, 16, 3, 1)),
                    ("prelu2", nn.PReLU(16)),
                    ("conv3", nn.Conv2d(16, 32, 3, 1)),
                    ("prelu3", nn.PReLU(32)),
                ]
            )
        )

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        weights = np.load(PNET_PATH, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a, dim=-1)
        return b, a


class RNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 28, 3, 1)),
                    ("prelu1", nn.PReLU(28)),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(28, 48, 3, 1)),
                    ("prelu2", nn.PReLU(48)),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv3", nn.Conv2d(48, 64, 2, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("flatten", Flatten()),
                    ("conv4", nn.Linear(576, 128)),
                    ("prelu4", nn.PReLU(128)),
                ]
            )
        )

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load(RNET_PATH, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a, dim=-1)
        return b, a


class ONet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 3, 1)),
                    ("prelu1", nn.PReLU(32)),
                    ("pool1", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv2", nn.Conv2d(32, 64, 3, 1)),
                    ("prelu2", nn.PReLU(64)),
                    ("pool2", nn.MaxPool2d(3, 2, ceil_mode=True)),
                    ("conv3", nn.Conv2d(64, 64, 3, 1)),
                    ("prelu3", nn.PReLU(64)),
                    ("pool3", nn.MaxPool2d(2, 2, ceil_mode=True)),
                    ("conv4", nn.Conv2d(64, 128, 2, 1)),
                    ("prelu4", nn.PReLU(128)),
                    ("flatten", Flatten()),
                    ("conv5", nn.Linear(1152, 256)),
                    ("drop5", nn.Dropout(0.25)),
                    ("prelu5", nn.PReLU(256)),
                ]
            )
        )

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load(ONET_PATH, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, dim=-1)
        return c, b, a

import numpy as np
import torch
from torch.autograd import Variable
from .get_nets import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from .first_stage import run_first_stage


def detect_faces(
    image,
    min_face_size=20.0,
    thresholds=[0.6, 0.7, 0.8],
    nms_thresholds=[0.7, 0.7, 0.7],
):
    """
    Arguments:
        image: an instance of PIL.Image.
        min_face_size: a float number.
        thresholds: a list of length 3.
        nms_thresholds: a list of length 3.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """

    # LOAD MODELS
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    # BUILD AN IMAGE PYRAMID
    width, height = image.size
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size / min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    with torch.no_grad():
        # run P-Net on different scales
        for s in scales:
            boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = torch.FloatTensor(img_boxes)

        output = rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = torch.FloatTensor(img_boxes)
        output = onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = (
            np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        )
        landmarks[:, 5:10] = (
            np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
        )

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode="min")
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

    return bounding_boxes, landmarks

from PIL import ImageDraw


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """

    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="white")

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse(
                [(p[i] - 1.0, p[i + 5] - 1.0), (p[i] + 1.0, p[i + 5] + 1.0)],
                outline="blue",
            )

    return img_copy

from .visualization_utils import show_bboxes
from .detector import detect_faces

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 06:54:28 2017

@author: zhaoyafei
"""

import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank


class MatlabCp2tormException(Exception):
    def __str__(self):
        return "In File {}:{}".format(__file__, super.__str__(self))


def tformfwd(trans, uv):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)

    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of transformed coordinates (x, y)
    """
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, 0:-1]
    return xy


def tforminv(trans, uv):
    """
    Function:
    ----------
        apply the inverse of affine transform 'trans' to uv

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix
        @uv: Kx2 np.array
            each row is a pair of coordinates (x, y)

    Returns:
    ----------
        @xy: Kx2 np.array
            each row is a pair of inverse-transformed coordinates (x, y)
    """
    Tinv = inv(trans)
    xy = tformfwd(Tinv, uv)
    return xy


def findNonreflectiveSimilarity(uv, xy, options=None):
    options = {"K": 2}

    K = options["K"]
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    # print('--->x, y:\n', x, y

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))
    # print('--->X.shape: ', X.shape
    # print('X:\n', X

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))
    # print('--->U.shape: ', U.shape
    # print('U:\n', U

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=None)  # Make sure this is what I want
        r = np.squeeze(r)
    else:
        raise Exception("cp2tform:twoUniquePointsReq")

    # print('--->r:\n', r

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([[sc, -ss, 0], [ss, sc, 0], [tx, ty, 1]])

    # print('--->Tinv:\n', Tinv

    T = inv(Tinv)
    # print('--->T:\n', T

    T[:, 2] = np.array([0, 0, 1])

    return T, Tinv


def findSimilarity(uv, xy, options=None):
    options = {"K": 2}

    #    uv = np.array(uv)
    #    xy = np.array(xy)

    # Solve for trans1
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    # Solve for trans2

    # manually reflect the xy data across the Y-axis
    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    trans2 = np.dot(trans2r, TreflectY)

    # Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'trans':
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y, 1] = [u, v, 1] * trans

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        @reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform

    Returns:
    ----------
       @trans: 3x3 np.array
            transform matrix from uv to xy
        trans_inv: 3x3 np.array
            inverse of trans, transform matrix from xy to uv
    """

    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)

    return trans, trans_inv


def cvt_tform_mat_for_cv2(trans):
    """
    Function:
    ----------
        Convert Transform Matrix 'trans' into 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
    ----------
        @trans: 3x3 np.array
            transform matrix from uv to xy

    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    cv2_trans = trans[:, 0:2].T

    return cv2_trans


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    """
    Function:
    ----------
        Find Similarity Transform Matrix 'cv2_trans' which could be
        directly used by cv2.warpAffine():
            u = src_pts[:, 0]
            v = src_pts[:, 1]
            x = dst_pts[:, 0]
            y = dst_pts[:, 1]
            [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points, each row is a pair of transformed
            coordinates (x, y)
        reflective: True or False
            if True:
                use reflective similarity transform
            else:
                use non-reflective similarity transform

    Returns:
    ----------
        @cv2_trans: 2x3 np.array
            transform matrix from src_pts to dst_pts, could be directly used
            for cv2.warpAffine()
    """
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    cv2_trans = cvt_tform_mat_for_cv2(trans)

    return cv2_trans


if __name__ == "__main__":
    """
    u = [0, 6, -2]
    v = [0, 3, 5]
    x = [-1, 0, 4]
    y = [-1, -10, 4]

    # In Matlab, run:
    #
    #   uv = [u'; v'];
    #   xy = [x'; y'];
    #   tform_sim=cp2tform(uv,xy,'similarity');
    #
    #   trans = tform_sim.tdata.T
    #   ans =
    #       -0.0764   -1.6190         0
    #        1.6190   -0.0764         0
    #       -3.2156    0.0290    1.0000
    #   trans_inv = tform_sim.tdata.Tinv
    #    ans =
    #
    #       -0.0291    0.6163         0
    #       -0.6163   -0.0291         0
    #       -0.0756    1.9826    1.0000
    #    xy_m=tformfwd(tform_sim, u,v)
    #
    #    xy_m =
    #
    #       -3.2156    0.0290
    #        1.1833   -9.9143
    #        5.0323    2.8853
    #    uv_m=tforminv(tform_sim, x,y)
    #
    #    uv_m =
    #
    #        0.5698    1.3953
    #        6.0872    2.2733
    #       -2.6570    4.3314
    """
    u = [0, 6, -2]
    v = [0, 3, 5]
    x = [-1, 0, 4]
    y = [-1, -10, 4]

    uv = np.array((u, v)).T
    xy = np.array((x, y)).T

    print("\n--->uv:")
    print(uv)
    print("\n--->xy:")
    print(xy)

    trans, trans_inv = get_similarity_transform(uv, xy)

    print("\n--->trans matrix:")
    print(trans)

    print("\n--->trans_inv matrix:")
    print(trans_inv)

    print("\n---> apply transform to uv")
    print("\nxy_m = uv_augmented * trans")
    uv_aug = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy_m = np.dot(uv_aug, trans)
    print(xy_m)

    print("\nxy_m = tformfwd(trans, uv)")
    xy_m = tformfwd(trans, uv)
    print(xy_m)

    print("\n---> apply inverse transform to xy")
    print("\nuv_m = xy_augmented * trans_inv")
    xy_aug = np.hstack((xy, np.ones((xy.shape[0], 1))))
    uv_m = np.dot(xy_aug, trans_inv)
    print(uv_m)

    print("\nuv_m = tformfwd(trans_inv, xy)")
    uv_m = tformfwd(trans_inv, xy)
    print(uv_m)

    uv_m = tforminv(trans, xy)
    print("\nuv_m = tforminv(trans, xy)")
    print(uv_m)

import torch
from torch.autograd import Variable
import math
from PIL import Image
import numpy as np
from .box_utils import nms, _preprocess

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda:0"


def run_first_stage(image, net, scale, threshold):
    """Run P-Net, generate bounding boxes, and do NMS.

    Arguments:
        image: an instance of PIL.Image.
        net: an instance of pytorch's nn.Module, P-Net.
        scale: a float number,
            scale width and height of the image by this number.
        threshold: a float number,
            threshold on the probability of a face when generating
            bounding boxes from predictions of the net.

    Returns:
        a float numpy array of shape [n_boxes, 9],
            bounding boxes with scores and offsets (4 + 1 + 4).
    """

    # scale the image and convert it to a float array
    width, height = image.size
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, "float32")

    img = torch.FloatTensor(_preprocess(img)).to(device)
    with torch.no_grad():
        output = net(img)
        probs = output[1].cpu().data.numpy()[0, 1, :, :]
        offsets = output[0].cpu().data.numpy()
        # probs: probability of a face at each sliding window
        # offsets: transformations to true bounding boxes

        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.

    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.

    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = np.vstack(
        [
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score,
            offsets,
        ]
    )
    # why one is added?

    return bounding_boxes.T


import numpy as np
import torch
from PIL import Image
from models.mtcnn.mtcnn_pytorch.src.get_nets import PNet, RNet, ONet
from models.mtcnn.mtcnn_pytorch.src.box_utils import (
    nms,
    calibrate_box,
    get_image_boxes,
    convert_to_square,
)
from models.mtcnn.mtcnn_pytorch.src.first_stage import run_first_stage
from models.mtcnn.mtcnn_pytorch.src.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)

device = "cuda:0"


class MTCNN:
    def __init__(self):
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.refrence = get_reference_facial_points(default_square=True)

    def align(self, img):
        _, landmarks = self.detect_faces(img)
        if len(landmarks) == 0:
            return None, None
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face, tfm = warp_and_crop_face(
            np.array(img), facial5points, self.refrence, crop_size=(112, 112)
        )
        return Image.fromarray(warped_face), tfm

    def align_multi(self, img, limit=None, min_face_size=30.0):
        boxes, landmarks = self.detect_faces(img, min_face_size)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        tfms = []
        for landmark in landmarks:
            facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
            warped_face, tfm = warp_and_crop_face(
                np.array(img), facial5points, self.refrence, crop_size=(112, 112)
            )
            faces.append(Image.fromarray(warped_face))
            tfms.append(tfm)
        return boxes, faces, tfms

    def detect_faces(
        self,
        image,
        min_face_size=20.0,
        thresholds=[0.15, 0.25, 0.35],
        nms_thresholds=[0.7, 0.7, 0.7],
    ):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(
                    image, self.pnet, scale=s, threshold=thresholds[0]
                )
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            bounding_boxes = np.vstack(bounding_boxes)

            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(
                bounding_boxes[:, 0:5], bounding_boxes[:, 5:]
            )
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 2

            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            img_boxes = torch.FloatTensor(img_boxes).to(device)

            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 3

            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.FloatTensor(img_boxes).to(device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = (
                np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            )
            landmarks[:, 5:10] = (
                np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
            )

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode="min")
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

        return bounding_boxes, landmarks



import torch
import torch.nn.functional as F
from collections import namedtuple
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    PReLU,
    ReLU,
    Sigmoid,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Sequential,
    Module,
)

"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)
    ]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    else:
        raise ValueError(
            "Invalid number of layers: {}. Must be one of [50, 100, 152]".format(
                num_layers
            )
        )
    return blocks


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def _upsample_add(x, y):
    """Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    """
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

from torch.nn import (
    Linear,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    PReLU,
    Dropout,
    Sequential,
    Module,
)
from models.psp.encoders.helpers import (
    get_blocks,
    Flatten,
    bottleneck_IR,
    bottleneck_IR_SE,
    l2_norm,
)

"""
Modified Backbone implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode="ir", drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        if input_size == 112:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(drop_ratio),
                Flatten(),
                Linear(512 * 7 * 7, 512),
                BatchNorm1d(512, affine=affine),
            )
        else:
            self.output_layer = Sequential(
                BatchNorm2d(512),
                Dropout(drop_ratio),
                Flatten(),
                Linear(512 * 14 * 14, 512),
                BatchNorm1d(512, affine=affine),
            )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


def IR_50(input_size):
    """Constructs a ir-50 model."""
    model = Backbone(input_size, num_layers=50, mode="ir", drop_ratio=0.4, affine=False)
    return model


def IR_101(input_size):
    """Constructs a ir-101 model."""
    model = Backbone(
        input_size, num_layers=100, mode="ir", drop_ratio=0.4, affine=False
    )
    return model


def IR_152(input_size):
    """Constructs a ir-152 model."""
    model = Backbone(
        input_size, num_layers=152, mode="ir", drop_ratio=0.4, affine=False
    )
    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(
        input_size, num_layers=50, mode="ir_se", drop_ratio=0.4, affine=False
    )
    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(
        input_size, num_layers=100, mode="ir_se", drop_ratio=0.4, affine=False
    )
    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(
        input_size, num_layers=152, mode="ir_se", drop_ratio=0.4, affine=False
    )
    return model

import torch
from torch import nn

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, )
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05, )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = []
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            out.append(x)
            x = self.layer2(x)
            out.append(x)
            x = self.layer3(x)
            out.append(x)
            x = self.layer4(x)
            out.append(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)

        if return_features:
            out.append(x)
            return out
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)
import numpy as np
import math
import torch
import torch.nn.functional as F
from enum import Enum
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from models.psp.encoders.feature_resnet import *

from models.psp.encoders.helpers import (
    get_blocks,
    Flatten,
    bottleneck_IR,
    bottleneck_IR_SE,
    _upsample_add
)
from models.psp.stylegan2.model import EqualLinear


class Inverter(nn.Module):
    def __init__(self, n_styles=18, opts=None):
        super(Inverter, self).__init__()

        self.fs_backbone = FSLikeBackbone(opts=opts, n_styles=n_styles)
        self.fuser = ContentLayerDeepFast(6, 1024, 512)


class FSLikeBackbone(nn.Module):
    def __init__(self, n_styles=18, opts=None):
        super(FSLikeBackbone, self).__init__()

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(opts.arcface_model_path))

        self.conv = nn.Sequential(*list(resnet50.children())[:3])

        # define layers
        self.block_1 = list(resnet50.children())[3]  # 15-18
        self.block_2 = list(resnet50.children())[4]  # 10-14
        self.block_3 = list(resnet50.children())[5]  # 5-9
        self.block_4 = list(resnet50.children())[6]  # 1-4

        # replace stride in conv to increase predicted dimensionality
        state = self.block_3.state_dict()
        self.block_3[0].conv2 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.block_3[0].downsample[0] = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.block_3.load_state_dict(state, strict=True)

        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))

        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))


    def apply_head(self, x):
        latents = []
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        out = torch.stack(latents, dim=1)
        return out


    def forward(self, x):
        features = []

        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))
        x = self.block_2(x)
        features.append(self.avg_pool(x))
        x = self.block_3(x)
        features.append(self.avg_pool(x))

        content = self.content_layer(x)
        x = self.block_4(x)
        features.append(self.avg_pool(x))

        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)

        return self.apply_head(x), content 


class ContentLayerDeepFast(nn.Module):
    def __init__(self, len=4, inp=1024, out=512):
        super().__init__()
        self.body = nn.ModuleList([])
        self.conv = nn.Conv2d(inp, out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        for i in range(len - 1):
            self.body.append(FeatureEncoderBlock(out, out))

    def forward(self, x):
        x = self.conv(x)
        for i, block in enumerate(self.body):
            x = x + block(x)
        return x


class FeatureEncoderBlock(nn.Module):
    def __init__(self, inp=1024, out=512):
        super().__init__()
        self.body = nn.Sequential(
                nn.BatchNorm2d(inp, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Conv2d(inp, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.PReLU(num_parameters=out),
                nn.Conv2d(out, out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

    def forward(self, x):
        return self.body(x)


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial, norm=False):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        self.norm = norm
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        ]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def body_forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print("Using BackboneEncoderUsingLastLayerIntoW")
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print("Using BackboneEncoderUsingLastLayerIntoWPlus")
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(
            Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        self.output_layer_2 = Sequential(
            BatchNorm2d(512),
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            Linear(512 * 7 * 7, 512),
        )
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x


class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18


class Encoder4Editing(Module):
    def __init__(self, num_layers, mode="ir", opts=None):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100, or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == "ir":
            unit_module = bottleneck_IR
        elif mode == "ir_se":
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel, bottleneck.depth, bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        """Get a list of the initial dimension of every delta from which it is applied"""
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print("Changed progressive stage to: ", new_stage)

    def forward(self, x, return_feat=False):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        if return_feat:
            return w, features
        return w


from .fused_act import FusedLeakyReLU, fused_leaky_relu
from .upfirdn2d import upfirdn2d

import os

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load
module_path = os.path.dirname(__file__)
fused = load(
    "fused",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused_bias_act_kernel.cu"),
    ],
)

class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        (out,) = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)

import os

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
upfirdn2d_op = load(
    "upfirdn2d",
    sources=[
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ],
)


class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):
        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        (kernel,) = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = UpFirDn2d.apply(
        input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
    )

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)

    return out[:, ::down_y, ::down_x, :]

import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from models.psp.stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from PIL import Image


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style, is_stylespace=False):
        batch, in_channel, height, width = input.shape
        
        weight = self.weight

        if not is_stylespace:
            style = self.modulation(style)
        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            padding = self.padding
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None, is_stylespace=False):
        out = self.conv(input, style, is_stylespace)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None, is_stylespace=False):
        out = self.conv(input, style, is_stylespace)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2**2, 2**2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        return_features=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        is_stylespace=False,
        new_features=None,
        feature_scale=1.0,
        early_stop=None,
    ):

        if is_stylespace:
            styles, to_rgb_stylespace = styles
        if not input_is_latent and not is_stylespace:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        def insert_feature(x, layer_idx):
            if new_features is not None and new_features[layer_idx] is not None:
                x = (1 - feature_scale) * x + feature_scale * new_features[layer_idx].type_as(x)
            return x

        outs = []
        out = self.input(latent)
        outs.append(out)

        out = self.conv1(out, styles[0].float() if is_stylespace else latent[:, 0], noise=noise[0], is_stylespace=is_stylespace)
        outs.append(out)

        skip = self.to_rgb1(out, to_rgb_stylespace[0].float() if is_stylespace else latent[:, 1], is_stylespace=is_stylespace)
        

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = insert_feature(out, i)
            out = conv1(out, styles[i].float() if is_stylespace else latent[:, i], noise=noise1, is_stylespace=is_stylespace)
            outs.append(out)

            out = insert_feature(out, i + 1)
            out = conv2(out, styles[i + 1].float() if is_stylespace else latent[:, i + 1], noise=noise2, is_stylespace=is_stylespace)
            outs.append(out)

            skip = to_rgb(out, to_rgb_stylespace[i // 2 + 1].float() if is_stylespace else latent[:, i + 2], skip, is_stylespace=is_stylespace)
            
            if early_stop is not None and skip.size(-1) == early_stop:
                break

            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_features:
            return image, outs
        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


