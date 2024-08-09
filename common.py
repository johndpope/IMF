import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# https://raw.githubusercontent.com/ml-lab/iSketchNFill/135a7a6d7daa54f12fe4b1c291973d96fbd8bc1e/models/common_net.py
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
    
def get_norm(planes,norm_type='batch',num_groups=4):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d(planes, affine=True)
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d(planes, affine=False)
    elif norm_type == 'group':
        norm_layer = nn.GroupNorm(num_groups,planes)
    elif norm_type == 'adain':
        norm_layer = AdaptiveInstanceNorm2d(planes)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def upsampleLayer(inplanes, outplanes, upsample='basic', use_sn=True):
    # padding_type = 'zero'
    if upsample == 'basic' and not use_sn:
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=3, stride=2,padding=1, output_padding=1)]
    elif upsample == 'bilinear' and not use_sn:
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif upsample == 'nearest' and not use_sn:
        upconv = [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif upsample == 'subpixel' and not use_sn:
        upconv = [ nn.Conv2d(inplanes,outplanes*4,kernel_size=3 , stride=1 , padding=1),
                   nn.PixelShuffle(2)]
    elif upsample == 'basic' and use_sn :
        upconv = [spectral_norm(nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=3, stride=2,padding=1, output_padding=1))]
    elif upsample == 'bilinear' and use_sn :
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif upsample == 'nearest' and use_sn :
        upconv = [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif upsample == 'subpixel' and use_sn:
        upconv = [ spectral_norm(nn.Conv2d(inplanes,outplanes*4,kernel_size=3 , stride=1 , padding=1)),
                   nn.PixelShuffle(2)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv

def downsampleLayer(inplanes, outplanes, downsample='basic', use_sn=True):
    # padding_type = 'zero'
    if downsample == 'basic' and not use_sn:
        downconv = [nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=2, padding=1)]
    elif downsample == 'avgpool' and not use_sn:
        downconv = [nn.AvgPool2d(2, stride=2),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    elif downsample == 'maxpool' and not use_sn:
        downconv = [nn.MaxPool2d(2, stride=2),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]

    elif downsample == 'basic' and use_sn :
        downconv = [spectral_norm(nn.Conv2d(
            inplanes, outplanes, kernel_size=3, stride=2, padding=1))]
    elif downsample == 'avgpool' and use_sn :
        downconv = [nn.AvgPool2d(2, stride=2),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif downsample == 'maxpool' and use_sn :
        downconv = [nn.MaxPool2d(2, stride=2),
                  nn.ReflectionPad2d(1),
                  spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]

    else:
        raise NotImplementedError(
            'downsample layer [%s] not implemented' % downsample)
    return downconv


class DownConvResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1, use_sn=True):
        if use_sn:
            return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
        else:
            return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

    def __init__(self, inplanes, planes, stride=1, dropout=0.0, use_sn=False, norm_layer='batch', num_groups=8):
        super(DownConvResBlock, self).__init__()
        model = []
        model += downsampleLayer(inplanes, planes, downsample='avgpool', use_sn=use_sn)
        if norm_layer != 'none':
            model += [get_norm(planes, norm_layer, num_groups)]
        model += [nn.ReLU(inplace=True)]
        model += [self.conv3x3(planes, planes, stride, use_sn)]
        if norm_layer != 'none':
            model += [get_norm(planes, norm_layer, num_groups)]
        model += [nn.ReLU(inplace=True)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

        residual_block = []
        residual_block += downsampleLayer(inplanes, planes, downsample='avgpool', use_sn=use_sn)
        self.residual_block = nn.Sequential(*residual_block)

    def forward(self, x):
        residual = self.residual_block(x)
        out = self.model(x)
        out += residual
        return out
    
class UpConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8):
    super(UpConvResBlock, self).__init__()
    model = []
    model += upsampleLayer(inplanes , planes , upsample='nearest' , use_sn=use_sn)
    if norm_layer != 'none':
        model += [get_norm(planes,norm_layer,num_groups)]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    if norm_layer != 'none':
        model += [get_norm(planes,norm_layer,num_groups)] #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

    residual_block = []
    residual_block +=  upsampleLayer(inplanes , planes , upsample='bilinear' , use_sn=use_sn)
    self.residual_block=nn.Sequential(*residual_block)

  def forward(self, x):
    residual = self.residual_block(x)
    out = self.model(x)
    out += residual
    return out
