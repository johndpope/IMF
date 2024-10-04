from typing import Tuple, Union

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F

from torch import Tensor

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}


def rgb_to_ycbcr420(rgb):
    '''
    input is 3xhxw RGB float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    '''
    c, h, w = rgb.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    r, g, b = np.split(rgb, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    # to 420
    cb = np.mean(np.reshape(cb, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    cr = np.mean(np.reshape(cr, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    uv = np.concatenate((cb, cr), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv


def ycbcr420_to_rgb(y, uv, order=1):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    cb = uv[0:1, :, :]
    cr = uv[1:2, :, :]
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=0)
    rgb = np.clip(rgb, 0., 1.)
    return rgb


def ycbcr420_to_444(y, uv, order=1):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw YCbCr float numpy array, in the range of [0, 1]
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def ycbcr444_to_420(yuv):
    '''
    input is 3xhxw YUV float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    '''
    c, h, w = yuv.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    y, u, v = np.split(yuv, 3, axis=0)

    # to 420
    u = np.mean(np.reshape(u, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    v = np.mean(np.reshape(v, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    uv = np.concatenate((u, v), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv


def rgb_to_ycbcr(rgb):
    '''
    input is 3xhxw RGB float numpy array, in the range of [0, 1]
    output is yuv: 3xhxw, in the range of [0, 1]
    '''
    c, h, w = rgb.shape
    assert c == 3
    r, g, b = np.split(rgb, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    yuv = np.concatenate((y, cb, cr), axis=0)
    yuv = np.clip(yuv, 0., 1.)

    return yuv


def ycbcr_to_rgb(yuv):
    '''
    yuv is 3xhxw YCbCr float numpy array, in the range of [0, 1]
    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    '''
    y, cb, cr = np.split(yuv, 3, axis=0)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = np.concatenate((r, g, b), axis=0)
    rgb = np.clip(rgb, 0., 1.)
    return rgb


def _check_input_tensor(tensor: Tensor) -> None:
    if (
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or not len(tensor.size()) in (3, 4)
        or not tensor.size(-3) == 3
    ):
        raise ValueError(
            "Expected a 3D or 4D tensor with shape (Nx3xHxW) or (3xHxW) as input"
        )


def rgb2ycbcr(rgb: Tensor) -> Tensor:
    """RGB to YCbCr conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        rgb (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        ycbcr (torch.Tensor): converted tensor
    """
    _check_input_tensor(rgb)

    r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    return ycbcr


def ycbcr2rgb(ycbcr: Tensor) -> Tensor:
    """YCbCr to RGB conversion for torch Tensor.
    Using ITU-R BT.709 coefficients.

    Args:
        ycbcr (torch.Tensor): 3D or 4D floating point RGB tensor

    Returns:
        rgb (torch.Tensor): converted tensor
    """
    _check_input_tensor(ycbcr)

    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = torch.cat((r, g, b), dim=-3)
    return rgb


def yuv_444_to_420(
    yuv: Union[Tensor, Tuple[Tensor, Tensor, Tensor]],
    mode: str = "avg_pool",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert a 444 tensor to a 420 representation.

    Args:
        yuv (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): 444
            input to be downsampled. Takes either a (Nx3xHxW) tensor or a tuple
            of 3 (Nx1xHxW) tensors.
        mode (str): algorithm used for downsampling: ``'avg_pool'``. Default
            ``'avg_pool'``

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): Converted 420
    """
    if mode not in ("avg_pool",):
        raise ValueError(f'Invalid downsampling mode "{mode}".')

    if mode == "avg_pool":

        def _downsample(tensor):
            return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    if isinstance(yuv, torch.Tensor):
        y, u, v = yuv.chunk(3, 1)
    else:
        y, u, v = yuv

    return (y, _downsample(u), _downsample(v))


def yuv_420_to_444(
    yuv: Tuple[Tensor, Tensor, Tensor],
    mode: str = "bilinear",
    return_tuple: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """Convert a 420 input to a 444 representation.

    Args:
        yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
            (Nx1xHxW) format
        mode (str): algorithm used for upsampling: ``'bilinear'`` |
            ``'nearest'`` Default ``'bilinear'``
        return_tuple (bool): return input as tuple of tensors instead of a
            concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
            tensor (default: False)

    Returns:
        (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
            444
    """
    if len(yuv) != 3 or any(not isinstance(c, torch.Tensor) for c in yuv):
        raise ValueError("Expected a tuple of 3 torch tensors")

    if mode not in ("bilinear", "nearest"):
        raise ValueError(f'Invalid upsampling mode "{mode}".')

    if mode in ("bilinear", "nearest"):

        def _upsample(tensor):
            return F.interpolate(tensor, scale_factor=2, mode=mode, align_corners=False)

    y, u, v = yuv
    u, v = _upsample(u), _upsample(v)
    if return_tuple:
        return y, u, v
    return torch.cat((y, u, v), dim=1)

# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
from pathlib import Path

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_ushorts(fd, n, fmt=">{:d}H"):
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def encode_i(height, width, q_in_ckpt, q_index, bit_stream, output):
    with Path(output).open("wb") as f:
        stream_length = len(bit_stream)

        write_uints(f, (height, width))
        write_uchars(f, ((q_in_ckpt << 7) + (q_index << 1),))  # 1-bit flag and 6-bit index
        write_uints(f, (stream_length,))
        write_bytes(f, bit_stream)


def decode_i(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 2)
        height = header[0]
        width = header[1]
        flag = read_uchars(f, 1)[0]
        q_in_ckpt = (flag >> 7) > 0
        q_index = ((flag & 0x7f) >> 1)
        stream_length = read_uints(f, 1)[0]

        bit_stream = read_bytes(f, stream_length)

    return height, width, q_in_ckpt, q_index, bit_stream


def encode_p(string, q_in_ckpt, q_index, frame_idx, output):
    with Path(output).open("wb") as f:
        string_length = len(string)
        write_uchars(f, ((q_in_ckpt << 7) + (q_index << 1),))
        write_uchars(f, (frame_idx,))
        write_uints(f, (string_length,))
        write_bytes(f, string)


def decode_p(inputpath):
    with Path(inputpath).open("rb") as f:
        flag = read_uchars(f, 1)[0]
        q_in_ckpt = (flag >> 7) > 0
        q_index = ((flag & 0x7f) >> 1)
        frame_idx = read_uchars(f, 1)[0]

        header = read_uints(f, 1)
        string_length = header[0]
        string = read_bytes(f, string_length)

    return q_in_ckpt, q_index, frame_idx, string

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from unittest.mock import patch

import numpy as np


def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def scale_list_to_str(scales):
    s = ''
    for scale in scales:
        s += f'{scale:.2f} '

    return s


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode  # pylint: disable=W0212

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, '.%df' % float_digits)
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def generate_log_json(frame_num, frame_pixel_num, test_time, frame_types, bits, psnrs, ssims,
                      psnrs_y=None, psnrs_u=None, psnrs_v=None,
                      ssims_y=None, ssims_u=None, ssims_v=None, verbose=False):
    include_yuv = psnrs_y is not None
    if include_yuv:
        assert psnrs_u is not None
        assert psnrs_v is not None
        assert ssims_y is not None
        assert ssims_u is not None
        assert ssims_v is not None
    i_bits = 0
    i_psnr = 0
    i_psnr_y = 0
    i_psnr_u = 0
    i_psnr_v = 0
    i_ssim = 0
    i_ssim_y = 0
    i_ssim_u = 0
    i_ssim_v = 0
    p_bits = 0
    p_psnr = 0
    p_psnr_y = 0
    p_psnr_u = 0
    p_psnr_v = 0
    p_ssim = 0
    p_ssim_y = 0
    p_ssim_u = 0
    p_ssim_v = 0
    i_num = 0
    p_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            i_bits += bits[idx]
            i_psnr += psnrs[idx]
            i_ssim += ssims[idx]
            i_num += 1
            if include_yuv:
                i_psnr_y += psnrs_y[idx]
                i_psnr_u += psnrs_u[idx]
                i_psnr_v += psnrs_v[idx]
                i_ssim_y += ssims_y[idx]
                i_ssim_u += ssims_u[idx]
                i_ssim_v += ssims_v[idx]
        else:
            p_bits += bits[idx]
            p_psnr += psnrs[idx]
            p_ssim += ssims[idx]
            p_num += 1
            if include_yuv:
                p_psnr_y += psnrs_y[idx]
                p_psnr_u += psnrs_u[idx]
                p_psnr_v += psnrs_v[idx]
                p_ssim_y += ssims_y[idx]
                p_ssim_u += ssims_u[idx]
                p_ssim_v += ssims_v[idx]

    log_result = {}
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = i_num
    log_result['p_frame_num'] = p_num
    log_result['ave_i_frame_bpp'] = i_bits / i_num / frame_pixel_num
    log_result['ave_i_frame_psnr'] = i_psnr / i_num
    log_result['ave_i_frame_msssim'] = i_ssim / i_num
    if include_yuv:
        log_result['ave_i_frame_psnr_y'] = i_psnr_y / i_num
        log_result['ave_i_frame_psnr_u'] = i_psnr_u / i_num
        log_result['ave_i_frame_psnr_v'] = i_psnr_v / i_num
        log_result['ave_i_frame_msssim_y'] = i_ssim_y / i_num
        log_result['ave_i_frame_msssim_u'] = i_ssim_u / i_num
        log_result['ave_i_frame_msssim_v'] = i_ssim_v / i_num
    if verbose:
        log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
        log_result['frame_psnr'] = psnrs
        log_result['frame_msssim'] = ssims
        log_result['frame_type'] = frame_types
        if include_yuv:
            log_result['frame_psnr_y'] = psnrs_y
            log_result['frame_psnr_u'] = psnrs_u
            log_result['frame_psnr_v'] = psnrs_v
            log_result['frame_msssim_y'] = ssims_y
            log_result['frame_msssim_u'] = ssims_u
            log_result['frame_msssim_v'] = ssims_v
    log_result['test_time'] = test_time
    if p_num > 0:
        total_p_pixel_num = p_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = p_bits / total_p_pixel_num
        log_result['ave_p_frame_psnr'] = p_psnr / p_num
        log_result['ave_p_frame_msssim'] = p_ssim / p_num
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = p_psnr_y / p_num
            log_result['ave_p_frame_psnr_u'] = p_psnr_u / p_num
            log_result['ave_p_frame_psnr_v'] = p_psnr_v / p_num
            log_result['ave_p_frame_msssim_y'] = p_ssim_y / p_num
            log_result['ave_p_frame_msssim_u'] = p_ssim_u / p_num
            log_result['ave_p_frame_msssim_v'] = p_ssim_v / p_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_psnr'] = 0
        log_result['ave_p_frame_msssim'] = 0
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = 0
            log_result['ave_p_frame_psnr_u'] = 0
            log_result['ave_p_frame_psnr_v'] = 0
            log_result['ave_p_frame_msssim_y'] = 0
            log_result['ave_p_frame_msssim_u'] = 0
            log_result['ave_p_frame_msssim_v'] = 0
    log_result['ave_all_frame_bpp'] = (i_bits + p_bits) / (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (i_psnr + p_psnr) / frame_num
    log_result['ave_all_frame_msssim'] = (i_ssim + p_ssim) / frame_num
    if include_yuv:
        log_result['ave_all_frame_psnr_y'] = (i_psnr_y + p_psnr_y) / frame_num
        log_result['ave_all_frame_psnr_u'] = (i_psnr_u + p_psnr_u) / frame_num
        log_result['ave_all_frame_psnr_v'] = (i_psnr_v + p_psnr_v) / frame_num
        log_result['ave_all_frame_msssim_y'] = (i_ssim_y + p_ssim_y) / frame_num
        log_result['ave_all_frame_msssim_u'] = (i_ssim_u + p_ssim_u) / frame_num
        log_result['ave_all_frame_msssim_v'] = (i_ssim_v + p_ssim_v) / frame_num

    return log_result

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from scipy import signal
from scipy import ndimage


def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def calc_ssim(img1, img2, data_range=255):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2

    return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                         (sigma1_sq + sigma2_sq + C2)),
            (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))


def calc_msssim(img1, img2, data_range=255):
    '''
    img1 and img2 are 2D arrays
    '''
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    height, width = img1.shape
    if height < 176 or width < 176:
        # according to HM implementation
        level = 4
        weight = np.array([0.0517, 0.3295, 0.3462, 0.2726])
    if height < 88 or width < 88:
        assert False
    downsample_filter = np.ones((2, 2)) / 4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(level):
        ssim_map, cs_map = calc_ssim(im1, im2, data_range=data_range)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter,
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter,
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level - 1]**weight[0:level - 1]) *
            (mssim[level - 1]**weight[level - 1]))


def calc_msssim_rgb(img1, img2, data_range=255):
    '''
    img1 and img2 are arrays with 3xHxW
    '''
    msssim = 0
    for i in range(3):
        msssim += calc_msssim(img1[i, :, :], img2[i, :, :], data_range)
    return msssim / 3


def calc_psnr(img1, img2, data_range=255):
    '''
    img1 and img2 are arrays with same shape
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean(np.square(img1 - img2))
    if mse > 1e-10:
        psnr = 10 * np.log10(data_range * data_range / mse)
    else:
        psnr = 999.9
    return psnr

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image
from ..transforms.functional import rgb_to_ycbcr420, ycbcr420_to_rgb


class VideoReader():
    def __init__(self, src_path, width, height):
        self.src_path = src_path
        self.width = width
        self.height = height
        self.eof = False

    def read_one_frame(self, dst_format='rgb'):
        '''
        y is 1xhxw Y float numpy array, in the range of [0, 1]
        uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
        rgb is 3xhxw float numpy array, in the range of [0, 1]
        '''
        raise NotImplementedError

    @staticmethod
    def _none_exist_frame(dst_format):
        if dst_format == "420":
            return None, None
        assert dst_format == "rgb"
        return None

    @staticmethod
    def _get_dst_format(rgb=None, y=None, uv=None, src_format='rgb', dst_format='rgb'):
        if dst_format == 'rgb':
            if rgb is None:
                rgb = ycbcr420_to_rgb(y, uv, order=1)
            return rgb
        assert dst_format == '420'
        if y is None:
            y, uv = rgb_to_ycbcr420(rgb)
        return y, uv


class PNGReader(VideoReader):
    def __init__(self, src_path, width, height, start_num=1):
        super().__init__(src_path, width, height)

        pngs = os.listdir(self.src_path)
        if 'im1.png' in pngs:
            self.padding = 1
        elif 'im00001.png' in pngs:
            self.padding = 5
        else:
            raise ValueError('unknown image naming convention; please specify')
        self.current_frame_index = start_num

    def read_one_frame(self, dst_format="rgb"):
        if self.eof:
            return self._none_exist_frame(dst_format)

        png_path = os.path.join(self.src_path,
                                f"im{str(self.current_frame_index).zfill(self.padding)}.png"
                                )
        if not os.path.exists(png_path):
            self.eof = True
            return self._none_exist_frame(dst_format)

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        _, height, width = rgb.shape
        assert height == self.height
        assert width == self.width

        self.current_frame_index += 1
        return self._get_dst_format(rgb=rgb, src_format='rgb', dst_format=dst_format)

    def close(self):
        self.current_frame_index = 1


class RGBReader(VideoReader):
    def __init__(self, src_path, width, height, src_format='rgb', bit_depth=8):
        super().__init__(src_path, width, height)
        if not src_path.endswith('.rgb'):
            src_path = src_path + '.rgb'
            self.src_path = src_path

        self.src_format = src_format
        self.bit_depth = bit_depth
        self.rgb_size = width * height * 3
        self.dtype = np.uint8
        self.max_val = 255
        if bit_depth > 8 and bit_depth <= 16:
            self.rgb_size = self.rgb_size * 2
            self.dtype = np.uint16
            self.max_val = (1 << bit_depth) - 1
        else:
            assert bit_depth == 8
        # pylint: disable=R1732
        self.file = open(src_path, "rb")
        # pylint: enable=R1732

    def read_one_frame(self, dst_format="420"):
        if self.eof:
            return self._none_exist_frame(dst_format)
        rgb = self.file.read(self.rgb_size)
        if not rgb:
            self.eof = True
            return self._none_exist_frame(dst_format)
        rgb = np.frombuffer(rgb, dtype=self.dtype).copy().reshape(3, self.height, self.width)
        rgb = rgb.astype(np.float32) / self.max_val

        return self._get_dst_format(rgb=rgb, src_format='rgb', dst_format=dst_format)

    def close(self):
        self.file.close()


class YUVReader(VideoReader):
    def __init__(self, src_path, width, height, src_format='420', skip_frame=0):
        super().__init__(src_path, width, height)
        if not src_path.endswith('.yuv'):
            src_path = src_path + '.yuv'
            self.src_path = src_path

        self.src_format = src_format
        self.y_size = width * height
        if src_format == '420':
            self.uv_size = width * height // 2
        else:
            assert False
        # pylint: disable=R1732
        self.file = open(src_path, "rb")
        # pylint: enable=R1732
        skipped_frame = 0
        while not self.eof and skipped_frame < skip_frame:
            y = self.file.read(self.y_size)
            uv = self.file.read(self.uv_size)
            if not y or not uv:
                self.eof = True
            skipped_frame += 1

    def read_one_frame(self, dst_format="420"):
        if self.eof:
            return self._none_exist_frame(dst_format)
        y = self.file.read(self.y_size)
        uv = self.file.read(self.uv_size)
        if not y or not uv:
            self.eof = True
            return self._none_exist_frame(dst_format)
        y = np.frombuffer(y, dtype=np.uint8).copy().reshape(1, self.height, self.width)
        uv = np.frombuffer(uv, dtype=np.uint8).copy().reshape(2, self.height // 2, self.width // 2)
        y = y.astype(np.float32) / 255
        uv = uv.astype(np.float32) / 255

        return self._get_dst_format(y=y, uv=uv, src_format='420', dst_format=dst_format)

    def close(self):
        self.file.close()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image
from ..transforms.functional import ycbcr420_to_rgb, rgb_to_ycbcr420


class VideoWriter():
    def __init__(self, dst_path, width, height):
        self.dst_path = dst_path
        self.width = width
        self.height = height

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="rgb"):
        '''
        y is 1xhxw Y float numpy array, in the range of [0, 1]
        uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
        rgb is 3xhxw float numpy array, in the range of [0, 1]
        '''
        raise NotImplementedError


class PNGWriter(VideoWriter):
    def __init__(self, dst_path, width, height):
        super().__init__(dst_path, width, height)
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="rgb"):
        if src_format == "420":
            rgb = ycbcr420_to_rgb(y, uv, order=1)
        rgb = rgb.transpose(1, 2, 0)

        png_path = os.path.join(self.dst_path,
                                f"im{str(self.current_frame_index).zfill(self.padding)}.png"
                                )
        img = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(png_path)

        self.current_frame_index += 1

    def close(self):
        self.current_frame_index = 1


class RGBWriter(VideoWriter):
    def __init__(self, dst_path, width, height, dst_format='rgb', bit_depth=8):
        super().__init__(dst_path, width, height)
        if not dst_path.endswith('.rgb'):
            dst_path = dst_path + '/out.rgb'
            self.dst_path = dst_path

        self.dst_format = dst_format
        self.bit_depth = bit_depth
        self.rgb_size = width * height * 3
        self.dtype = np.uint8
        self.max_val = 255
        if bit_depth > 8 and bit_depth <= 16:
            self.rgb_size = self.rgb_size * 2
            self.dtype = np.uint16
            self.max_val = (1 << bit_depth) - 1
        else:
            assert bit_depth == 8
        # pylint: disable=R1732
        self.file = open(dst_path, "wb")
        # pylint: enable=R1732

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="rgb"):
        if src_format == '420':
            rgb = ycbcr420_to_rgb(y, uv, order=1)
        rgb = np.clip(np.rint(rgb * self.max_val), 0, self.max_val).astype(self.dtype)

        self.file.write(rgb.tobytes())

    def close(self):
        self.file.close()


class YUVWriter(VideoWriter):
    def __init__(self, dst_path, width, height, dst_format='420'):
        super().__init__(dst_path, width, height)
        if not dst_path.endswith('.yuv'):
            dst_path = dst_path + '/out.yuv'
            self.dst_path = dst_path

        self.dst_format = dst_format
        self.y_size = width * height
        if dst_format == '420':
            self.uv_size = width * height // 2
        else:
            assert False
        self.eof = False
        # pylint: disable=R1732
        self.file = open(dst_path, "wb")
        # pylint: enable=R1732

    def write_one_frame(self, rgb=None, y=None, uv=None, src_format="420"):
        if src_format == 'rgb':
            y, uv = rgb_to_ycbcr420(rgb)
        y = np.clip(np.rint(y * 255), 0, 255).astype(np.uint8)
        uv = np.clip(np.rint(uv * 255), 0, 255).astype(np.uint8)

        self.file.write(y.tobytes())
        self.file.write(uv.tobytes())

    def close(self):
        self.file.close()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import numpy as np


from .common_model import CompressionModel
from .layers import conv3x3, DepthConvBlock2, ResidualBlockUpsample, ResidualBlockWithStride
from .video_net import UNet2
from ..utils.stream_helper import encode_i, decode_i, get_downsampled_shape, filesize, \
    get_state_dict


class IntraEncoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(3, 128, stride=2, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
        )
        self.enc_2 = nn.Sequential(
            ResidualBlockWithStride(128, 192, stride=2, inplace=inplace),
            DepthConvBlock2(192, 192, inplace=inplace),
            ResidualBlockWithStride(192, N, stride=2, inplace=inplace),
            DepthConvBlock2(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock2(N, N, inplace=inplace),
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            DepthConvBlock2(N, N, inplace=inplace),
            ResidualBlockUpsample(N, 192, 2, inplace=inplace),
            DepthConvBlock2(192, 192, inplace=inplace),
            ResidualBlockUpsample(192, 128, 2, inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            DepthConvBlock2(128, 128, inplace=inplace),
            ResidualBlockUpsample(128, 16, 2, inplace=inplace),
        )

    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        return self.dec_2(out)


class IntraNoAR(CompressionModel):
    def __init__(self, N=256, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='gaussian', z_channel=N,
                         ec_thread=ec_thread, stream_part=stream_part)

        self.enc = IntraEncoder(N, inplace)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock2(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            DepthConvBlock2(N, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock2(N, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(N * 4, N * 3, 1)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock2(N * 3, N * 3, inplace=inplace),
            DepthConvBlock2(N * 3, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
        )

        self.dec = IntraDecoder(N, inplace)
        self.refine = nn.Sequential(
            UNet2(16, 16, inplace=inplace),
            conv3x3(16, 3),
        )

        self.q_basic_enc = nn.Parameter(torch.ones((1, 128, 1, 1)))
        self.q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.q_scale_enc_fine = None
        self.q_basic_dec = nn.Parameter(torch.ones((1, 128, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.q_scale_dec_fine = None

    def get_q_for_inference(self, q_in_ckpt, q_index):
        q_scale_enc = self.q_scale_enc[:, 0, 0, 0] if q_in_ckpt else self.q_scale_enc_fine
        curr_q_enc = self.get_curr_q(q_scale_enc, self.q_basic_enc, q_index=q_index)
        q_scale_dec = self.q_scale_dec[:, 0, 0, 0] if q_in_ckpt else self.q_scale_dec_fine
        curr_q_dec = self.get_curr_q(q_scale_dec, self.q_basic_dec, q_index=q_index)
        return curr_q_enc, curr_q_dec

    def forward(self, x, q_in_ckpt=False, q_index=None):
        curr_q_enc, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = self.quant(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        _, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat = self.dec(y_hat, curr_q_dec)
        x_hat = self.refine(x_hat)

        y_for_bit = y_q
        z_for_bit = z_hat
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bits = torch.sum(bpp_y + bpp_z) * pixel_num
        bpp = bpp_y + bpp_z

        return {
            "x_hat": x_hat,
            "bit": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        q_scale_enc = ckpt["q_scale_enc"].reshape(-1)
        q_scale_dec = ckpt["q_scale_dec"].reshape(-1)
        return q_scale_enc, q_scale_dec

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        with torch.no_grad():
            q_scale_enc_fine = np.linspace(np.log(self.q_scale_enc[0, 0, 0, 0]),
                                           np.log(self.q_scale_enc[3, 0, 0, 0]), 64)
            self.q_scale_enc_fine = np.exp(q_scale_enc_fine)
            q_scale_dec_fine = np.linspace(np.log(self.q_scale_dec[0, 0, 0, 0]),
                                           np.log(self.q_scale_dec[3, 0, 0, 0]), 64)
            self.q_scale_dec_fine = np.exp(q_scale_dec_fine)

    def encode_decode(self, x, q_in_ckpt, q_index,
                      output_path=None, pic_width=None, pic_height=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        if output_path is None:
            encoded = self.forward(x, q_in_ckpt, q_index)
            result = {
                'bit': encoded['bit'].item(),
                'x_hat': encoded['x_hat'],
            }
            return result

        assert pic_height is not None
        assert pic_width is not None
        compressed = self.compress(x, q_in_ckpt, q_index)
        bit_stream = compressed['bit_stream']
        encode_i(pic_height, pic_width, q_in_ckpt, q_index, bit_stream, output_path)
        bit = filesize(output_path) * 8

        height, width, q_in_ckpt, q_index, bit_stream = decode_i(output_path)
        decompressed = self.decompress(bit_stream, height, width, q_in_ckpt, q_index)
        x_hat = decompressed['x_hat']

        result = {
            'bit': bit,
            'x_hat': x_hat,
        }
        return result

    def compress(self, x, q_in_ckpt, q_index):
        curr_q_enc, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = torch.round(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()

        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def decompress(self, bit_stream, height, width, q_in_ckpt, q_index):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        _, curr_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        self.entropy_coder.set_stream(bit_stream)
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)

        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        return {"x_hat": x_hat}

import math

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class EntropyCoder():
    def __init__(self, ec_thread=False, stream_part=1):
        super().__init__()

        from .MLCodec_rans import RansEncoder, RansDecoder
        self.encoder = RansEncoder(ec_thread, stream_part)
        self.decoder = RansDecoder(stream_part)

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from .MLCodec_CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
        cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def reset(self):
        self.encoder.reset()

    def encode_with_indexes(self, symbols, indexes, cdf, cdf_length, offset):
        self.encoder.encode_with_indexes(symbols.clamp(-30000, 30000).to(torch.int16).cpu().numpy(),
                                         indexes.to(torch.int16).cpu().numpy(),
                                         cdf, cdf_length, offset)

    def flush(self):
        self.encoder.flush()

    def get_encoded_stream(self):
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream):
        self.decoder.set_stream((np.frombuffer(stream, dtype=np.uint8)))

    def decode_stream(self, indexes, cdf, cdf_length, offset):
        rv = self.decoder.decode_stream(indexes.to(torch.int16).cpu().numpy(),
                                        cdf, cdf_length, offset)
        rv = torch.Tensor(rv)
        return rv


class Bitparm(nn.Module):
    def __init__(self, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        x = x * F.softplus(self.h) + self.b
        if self.final:
            return x

        return x + torch.tanh(x) * torch.tanh(self.a)


class AEHelper():
    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def set_entropy_coder(self, coder):
        self.entropy_coder = coder

    def set_cdf_info(self, quantized_cdf, cdf_length, offset):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()
        self._offset = offset.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset


class BitEstimator(AEHelper, nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        self.channel = channel

    def forward(self, x):
        return self.get_cdf(x)

    def get_logits_cdf(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

    def get_cdf(self, x):
        return torch.sigmoid(self.get_logits_cdf(x))

    def update(self, force=False, entropy_coder=None):
        if entropy_coder is not None:
            self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.channel), device=device)

            minima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) - i
                samples = samples[None, :, None, None]
                probs = self.forward(samples)
                probs = torch.squeeze(probs)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) + i
                samples = samples[None, :, None, None]
                probs = self.forward(samples)
                probs = torch.squeeze(probs)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device)

            samples = samples[None, :] + pmf_start[:, None, None]

            half = float(0.5)

            lower = self.forward(samples - half).squeeze(0)
            upper = self.forward(samples + half).squeeze(0)
            pmf = upper - lower

            pmf = pmf[:, 0, :]
            tail_mass = lower[:, 0, :1] + (1.0 - upper[:, 0, -1:])

            quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.set_cdf_info(quantized_cdf, cdf_length, offset)

    @staticmethod
    def build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C, dtype=torch.int).view(1, -1, 1, 1)
        return indexes.repeat(N, 1, H, W)

    @staticmethod
    def build_indexes_np(size):
        return BitEstimator.build_indexes(size).cpu().numpy()

    def encode(self, x):
        indexes = self.build_indexes(x.size())
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      *self.get_cdf_info())

    def decode_stream(self, size, dtype, device):
        output_size = (1, self.channel, size[0], size[1])
        indexes = self.build_indexes(output_size)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1), *self.get_cdf_info())
        val = val.reshape(indexes.shape)
        return val.to(dtype).to(device)


class GaussianEncoder(AEHelper):
    def __init__(self, distribution='laplace'):
        super().__init__()
        assert distribution in ['laplace', 'gaussian']
        self.distribution = distribution
        if distribution == 'laplace':
            self.cdf_distribution = torch.distributions.laplace.Laplace
            self.scale_min = 0.01
            self.scale_max = 64.0
            self.scale_level = 256
        elif distribution == 'gaussian':
            self.cdf_distribution = torch.distributions.normal.Normal
            self.scale_min = 0.11
            self.scale_max = 64.0
            self.scale_level = 256
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)

    @staticmethod
    def get_scale_table(min_val, max_val, levels):
        return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))

    def update(self, force=False, entropy_coder=None):
        if entropy_coder is not None:
            self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

        pmf_center = torch.zeros_like(self.scale_table) + 50
        scales = torch.zeros_like(pmf_center) + self.scale_table
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)
        for i in range(50, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.set_cdf_info(quantized_cdf, pmf_length+2, -pmf_center)

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.log_scale_min) / self.log_scale_step
        indexes = indexes.clamp_(0, self.scale_level - 1)
        return indexes.int()

    def encode(self, x, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      *self.get_cdf_info())

    def decode_stream(self, scales, dtype, device):
        indexes = self.build_indexes(scales)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1),
                                               *self.get_cdf_info())
        val = val.reshape(scales.shape)
        return val.to(device).to(dtype)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import torch
from torch import nn

from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
from ..utils.stream_helper import get_padding_size


class CompressionModel(nn.Module):
    def __init__(self, y_distribution, z_channel, mv_z_channel=None,
                 ec_thread=False, stream_part=1):
        super().__init__()

        self.y_distribution = y_distribution
        self.z_channel = z_channel
        self.mv_z_channel = mv_z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(z_channel)
        self.bit_estimator_z_mv = None
        if mv_z_channel is not None:
            self.bit_estimator_z_mv = BitEstimator(mv_z_channel)
        self.gaussian_encoder = GaussianEncoder(distribution=y_distribution)
        self.ec_thread = ec_thread
        self.stream_part = stream_part

        self.masks = {}

    def quant(self, x):
        return torch.round(x)

    def get_curr_q(self, q_scale, q_basic, q_index=None):
        q_scale = q_scale[q_index]
        return q_basic * q_scale

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = torch.clamp_min(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_y_laplace_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator):
        probs = bit_estimator.get_cdf(z + 0.5) - bit_estimator.get_cdf(z - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread, self.stream_part)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        if self.bit_estimator_z_mv is not None:
            self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, 4)
        y_pad = torch.nn.functional.pad(
            y,
            (padding_l, padding_r, padding_t, padding_b),
            mode="replicate",
        )
        return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)

    @staticmethod
    def get_to_y_slice_shape(height, width):
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 4)
        return (-padding_l, -padding_r, -padding_t, -padding_b)

    def slice_to_y(self, param, slice_shape):
        return torch.nn.functional.pad(param, slice_shape)

    @staticmethod
    def separate_prior(params):
        return params.chunk(3, 1)

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    def get_mask_four_parts(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
            mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)

            micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
            mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
            mask_1 = mask_1[:height, :width]
            mask_1 = torch.unsqueeze(mask_1, 0)
            mask_1 = torch.unsqueeze(mask_1, 0)

            micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
            mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
            mask_2 = mask_2[:height, :width]
            mask_2 = torch.unsqueeze(mask_2, 0)
            mask_2 = torch.unsqueeze(mask_2, 0)

            micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
            mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
            mask_3 = mask_3[:height, :width]
            mask_3 = torch.unsqueeze(mask_3, 0)
            mask_3 = torch.unsqueeze(mask_3, 0)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                           x_1_0, x_1_1, x_1_2, x_1_3,
                           x_2_0, x_2_1, x_2_2, x_2_3,
                           x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)

    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        quant_step, scales, means = self.separate_prior(common_params)
        dtype = y.dtype
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)

        quant_step = torch.clamp_min(quant_step, 0.5)
        y = y / quant_step
        y_0, y_1, y_2, y_3 = y.chunk(4, 1)

        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        y_res_0_0, y_q_0_0, y_hat_0_0, s_hat_0_0 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_res_1_1, y_q_1_1, y_hat_1_1, s_hat_1_1 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_1)
        y_res_2_2, y_q_2_2, y_hat_2_2, s_hat_2_2 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_2)
        y_res_3_3, y_q_3_3, y_hat_3_3, s_hat_3_3 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_3)
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)

        y_hat_so_far = y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)

        y_res_0_3, y_q_0_3, y_hat_0_3, s_hat_0_3 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_3)
        y_res_1_2, y_q_1_2, y_hat_1_2, s_hat_1_2 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_2)
        y_res_2_1, y_q_2_1, y_hat_2_1, s_hat_2_1 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_1)
        y_res_3_0, y_q_3_0, y_hat_3_0, s_hat_3_0 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_0)
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)

        y_res_0_2, y_q_0_2, y_hat_0_2, s_hat_0_2 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_2)
        y_res_1_3, y_q_1_3, y_hat_1_3, s_hat_1_3 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_3)
        y_res_2_0, y_q_2_0, y_hat_2_0, s_hat_2_0 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_0)
        y_res_3_1, y_q_3_1, y_hat_3_1, s_hat_3_1 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_1)
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)

        y_res_0_1, y_q_0_1, y_hat_0_1, s_hat_0_1 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_1)
        y_res_1_0, y_q_1_0, y_hat_1_0, s_hat_1_0 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_0)
        y_res_2_3, y_q_2_3, y_hat_2_3, s_hat_2_3 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_3)
        y_res_3_2, y_q_3_2, y_hat_3_2, s_hat_3_2 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_2)

        y_res = self.combine_four_parts(y_res_0_0, y_res_0_1, y_res_0_2, y_res_0_3,
                                        y_res_1_0, y_res_1_1, y_res_1_2, y_res_1_3,
                                        y_res_2_0, y_res_2_1, y_res_2_2, y_res_2_3,
                                        y_res_3_0, y_res_3_1, y_res_3_2, y_res_3_3)
        y_q = self.combine_four_parts(y_q_0_0, y_q_0_1, y_q_0_2, y_q_0_3,
                                      y_q_1_0, y_q_1_1, y_q_1_2, y_q_1_3,
                                      y_q_2_0, y_q_2_1, y_q_2_2, y_q_2_3,
                                      y_q_3_0, y_q_3_1, y_q_3_2, y_q_3_3)
        y_hat = self.combine_four_parts(y_hat_0_0, y_hat_0_1, y_hat_0_2, y_hat_0_3,
                                        y_hat_1_0, y_hat_1_1, y_hat_1_2, y_hat_1_3,
                                        y_hat_2_0, y_hat_2_1, y_hat_2_2, y_hat_2_3,
                                        y_hat_3_0, y_hat_3_1, y_hat_3_2, y_hat_3_3)
        scales_hat = self.combine_four_parts(s_hat_0_0, s_hat_0_1, s_hat_0_2, s_hat_0_3,
                                             s_hat_1_0, s_hat_1_1, s_hat_1_2, s_hat_1_3,
                                             s_hat_2_0, s_hat_2_1, s_hat_2_2, s_hat_2_3,
                                             s_hat_3_0, s_hat_3_1, s_hat_3_2, s_hat_3_3)

        y_hat = y_hat * quant_step

        if write:
            y_q_w_0 = y_q_0_0 + y_q_1_1 + y_q_2_2 + y_q_3_3
            y_q_w_1 = y_q_0_3 + y_q_1_2 + y_q_2_1 + y_q_3_0
            y_q_w_2 = y_q_0_2 + y_q_1_3 + y_q_2_0 + y_q_3_1
            y_q_w_3 = y_q_0_1 + y_q_1_0 + y_q_2_3 + y_q_3_2
            scales_w_0 = s_hat_0_0 + s_hat_1_1 + s_hat_2_2 + s_hat_3_3
            scales_w_1 = s_hat_0_3 + s_hat_1_2 + s_hat_2_1 + s_hat_3_0
            scales_w_2 = s_hat_0_2 + s_hat_1_3 + s_hat_2_0 + s_hat_3_1
            scales_w_3 = s_hat_0_1 + s_hat_1_0 + s_hat_2_3 + s_hat_3_2
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3,\
                scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat
        return y_res, y_q, y_hat, scales_hat

    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior):
        quant_step, scales, means = self.separate_prior(common_params)
        dtype = means.dtype
        device = means.device
        _, _, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)
        quant_step = torch.clamp_min(quant_step, 0.5)

        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        scales_r = scales_0 * mask_0 + scales_1 * mask_1 + scales_2 * mask_2 + scales_3 * mask_3
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_0 = (y_q_r + means_0) * mask_0
        y_hat_1_1 = (y_q_r + means_1) * mask_1
        y_hat_2_2 = (y_q_r + means_2) * mask_2
        y_hat_3_3 = (y_q_r + means_3) * mask_3
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)
        scales_r = scales_0 * mask_3 + scales_1 * mask_2 + scales_2 * mask_1 + scales_3 * mask_0
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_3 = (y_q_r + means_0) * mask_3
        y_hat_1_2 = (y_q_r + means_1) * mask_2
        y_hat_2_1 = (y_q_r + means_2) * mask_1
        y_hat_3_0 = (y_q_r + means_3) * mask_0
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        scales_r = scales_0 * mask_2 + scales_1 * mask_3 + scales_2 * mask_0 + scales_3 * mask_1
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_2 = (y_q_r + means_0) * mask_2
        y_hat_1_3 = (y_q_r + means_1) * mask_3
        y_hat_2_0 = (y_q_r + means_2) * mask_0
        y_hat_3_1 = (y_q_r + means_3) * mask_1
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)
        scales_r = scales_0 * mask_1 + scales_1 * mask_0 + scales_2 * mask_3 + scales_3 * mask_2
        y_q_r = self.gaussian_encoder.decode_stream(scales_r, dtype, device)
        y_hat_0_1 = (y_q_r + means_0) * mask_1
        y_hat_1_0 = (y_q_r + means_1) * mask_0
        y_hat_2_3 = (y_q_r + means_2) * mask_3
        y_hat_3_2 = (y_q_r + means_3) * mask_2
        y_hat_curr_step = torch.cat((y_hat_0_1, y_hat_1_0, y_hat_2_3, y_hat_3_2), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far * quant_step

        return y_hat

import torch
from torch import nn
import torch.nn.functional as F

from .layers import subpel_conv1x1, conv3x3, DepthConvBlock, DepthConvBlock2


backward_grid = [{} for _ in range(9)]    # 0~7 for GPU, -1 for CPU


def add_grid_cache(flow):
    device_id = -1 if flow.device == torch.device('cpu') else flow.device.index
    if str(flow.size()) not in backward_grid[device_id]:
        N, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=flow.device, dtype=torch.float32).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=flow.device, dtype=torch.float32).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)


def torch_warp(feature, flow):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    add_grid_cache(flow)
    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = (backward_grid[device_id][str(flow.size())] + flow)
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


def flow_warp(im, flow):
    warp = torch_warp(im, flow)
    return warp


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)

    return outfeature


def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature


class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, end_with_relu=False,
                 bottleneck=False, inplace=False):
        super().__init__()
        in_channel = channel // 2 if bottleneck else channel
        self.first_layer = nn.LeakyReLU(negative_slope=slope, inplace=False)
        self.conv1 = nn.Conv2d(channel, in_channel, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.conv2 = nn.Conv2d(in_channel, channel, 3, padding=1)
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return identity + out


class MEBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([MEBasic() for _ in range(self.L)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                self.moduleBasic[level](torch.cat([im1_list[img_index],
                                                   flow_warp(im2_list[img_index], flow_up),
                                                   flow_up], 1))

        return flow


class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
            DepthConvBlock(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


class UNet2(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock2(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock2(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock2(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock2(128, 128, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock2(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock2(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


def get_hyper_enc_dec_models(y_channel, z_channel, reduce_enc_layer=False, inplace=False):
    if reduce_enc_layer:
        enc = nn.Sequential(
            nn.Conv2d(y_channel, z_channel, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
        )
    else:
        enc = nn.Sequential(
            conv3x3(y_channel, z_channel),
            nn.LeakyReLU(inplace=inplace),
            conv3x3(z_channel, z_channel),
            nn.LeakyReLU(inplace=inplace),
            conv3x3(z_channel, z_channel, stride=2),
            nn.LeakyReLU(inplace=inplace),
            conv3x3(z_channel, z_channel),
            nn.LeakyReLU(inplace=inplace),
            conv3x3(z_channel, z_channel, stride=2),
        )

    dec = nn.Sequential(
        conv3x3(z_channel, y_channel),
        nn.LeakyReLU(inplace=inplace),
        subpel_conv1x1(y_channel, y_channel, 2),
        nn.LeakyReLU(inplace=inplace),
        conv3x3(y_channel, y_channel),
        nn.LeakyReLU(inplace=inplace),
        subpel_conv1x1(y_channel, y_channel, 2),
        nn.LeakyReLU(inplace=inplace),
        conv3x3(y_channel, y_channel),
    )

    return enc, dec

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time

import torch
from torch import nn
import numpy as np

from .common_model import CompressionModel
from .video_net import ME_Spynet, ResBlock, UNet, bilinearupsacling, bilineardownsacling, \
    get_hyper_enc_dec_models, flow_warp
from .layers import subpel_conv3x3, subpel_conv1x1, DepthConvBlock, \
    ResidualBlockWithStride, ResidualBlockUpsample
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
    get_state_dict


g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


class OffsetDiversity(nn.Module):
    def __init__(self, in_channel=g_ch_1x, aux_feature_num=g_ch_1x+3+2,
                 offset_num=2, group_num=16, max_residue_magnitude=40, inplace=False):
        super().__init__()
        self.in_channel = in_channel
        self.offset_num = offset_num
        self.group_num = group_num
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(aux_feature_num, g_ch_2x, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, g_ch_2x, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, 3 * group_num * offset_num, 3, 1, 1),
        )
        self.fusion = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)

    def forward(self, x, aux_feature, flow):
        B, C, H, W = x.shape
        out = self.conv_offset(aux_feature)
        out = bilinearupsacling(out)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.repeat(1, self.group_num * self.offset_num, 1, 1)

        # warp
        offset = offset.view(B * self.group_num * self.offset_num, 2, H, W)
        mask = mask.view(B * self.group_num * self.offset_num, 1, H, W)
        x = x.repeat(1, self.offset_num, 1, 1)
        x = x.view(B * self.group_num * self.offset_num, C // self.group_num, H, W)
        x = flow_warp(x, offset)
        x = x * mask
        x = x.view(B, C * self.offset_num, H, W)
        x = self.fusion(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_2x, g_ch_4x, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(g_ch_4x, inplace=inplace)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv3_up = subpel_conv3x3(g_ch_4x, g_ch_2x, 2)
        self.res_block3_up = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3_out = nn.Conv2d(g_ch_4x, g_ch_4x, 3, padding=1)
        self.res_block3_out = ResBlock(g_ch_4x, inplace=inplace)
        self.conv2_up = subpel_conv3x3(g_ch_2x * 2, g_ch_1x, 2)
        self.res_block2_up = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2_out = nn.Conv2d(g_ch_2x * 2, g_ch_2x, 3, padding=1)
        self.res_block2_out = ResBlock(g_ch_2x, inplace=inplace)
        self.conv1_out = nn.Conv2d(g_ch_1x * 2, g_ch_1x, 3, padding=1)
        self.res_block1_out = ResBlock(g_ch_1x, inplace=inplace)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out

        return context1, context2, context3


class MvEnc(nn.Module):
    def __init__(self, input_channel, channel, inplace=False):
        super().__init__()
        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(input_channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
        )
        self.enc_2 = ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace)

        self.adaptor_0 = DepthConvBlock(channel, channel, inplace=inplace)
        self.adaptor_1 = DepthConvBlock(channel * 2, channel, inplace=inplace)
        self.enc_3 = nn.Sequential(
            ResidualBlockWithStride(channel, channel, stride=2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            nn.Conv2d(channel, channel, 3, stride=2, padding=1),
        )

    def forward(self, x, context, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        out = self.enc_2(out)
        if context is None:
            out = self.adaptor_0(out)
        else:
            out = self.adaptor_1(torch.cat((out, context), dim=1))
        return self.enc_3(out)


class MvDec(nn.Module):
    def __init__(self, output_channel, channel, inplace=False):
        super().__init__()
        self.dec_1 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace),
            ResidualBlockUpsample(channel, channel, 2, inplace=inplace),
            DepthConvBlock(channel, channel, inplace=inplace)
        )
        self.dec_2 = ResidualBlockUpsample(channel, channel, 2, inplace=inplace)
        self.dec_3 = nn.Sequential(
            DepthConvBlock(channel, channel, inplace=inplace),
            subpel_conv1x1(channel, output_channel, 2),
        )

    def forward(self, x, quant_step):
        feature = self.dec_1(x)
        out = self.dec_2(feature)
        out = out * quant_step
        mv = self.dec_3(out)
        return mv, feature


class ContextualEncoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x + 3, g_ch_2x, 3, stride=2, padding=1)
        self.res1 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_2x * 2, g_ch_4x, 3, stride=2, padding=1)
        self.res2 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_4x * 2, g_ch_8x, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3, quant_step):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = feature * quant_step
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.up1 = subpel_conv3x3(g_ch_16x, g_ch_8x, 2)
        self.up2 = subpel_conv3x3(g_ch_8x, g_ch_4x, 2)
        self.res1 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up3 = subpel_conv3x3(g_ch_4x * 2, g_ch_2x, 2)
        self.res2 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up4 = subpel_conv3x3(g_ch_2x * 2, 32, 2)

    def forward(self, x, context2, context3, quant_step):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = feature * quant_step
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=g_ch_1x, res_channel=32, inplace=False):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, g_ch_1x, 3, stride=1, padding=1)
        self.unet_1 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.unet_2 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.recon_conv = nn.Conv2d(g_ch_1x, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class DMC(CompressionModel):
    def __init__(self, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='laplace', z_channel=g_ch_16x, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)

        channel_mv = 64
        channel_N = 64

        self.optic_flow = ME_Spynet()
        self.align = OffsetDiversity(inplace=inplace)

        self.mv_encoder = MvEnc(2, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N, inplace=inplace)

        self.mv_y_prior_fusion_adaptor_0 = DepthConvBlock(channel_mv * 1, channel_mv * 2,
                                                          inplace=inplace)
        self.mv_y_prior_fusion_adaptor_1 = DepthConvBlock(channel_mv * 2, channel_mv * 2,
                                                          inplace=inplace)

        self.mv_y_prior_fusion = nn.Sequential(
            DepthConvBlock(channel_mv * 2, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
        )

        self.mv_y_spatial_prior_adaptor_1 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_2 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)
        self.mv_y_spatial_prior_adaptor_3 = nn.Conv2d(channel_mv * 4, channel_mv * 3, 1)

        self.mv_y_spatial_prior = nn.Sequential(
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 3, inplace=inplace),
            DepthConvBlock(channel_mv * 3, channel_mv * 2, inplace=inplace),
        )

        self.mv_decoder = MvDec(2, channel_mv, inplace=inplace)

        self.feature_adaptor_I = nn.Conv2d(3, g_ch_1x, 3, stride=1, padding=1)
        self.feature_adaptor = nn.ModuleList([nn.Conv2d(g_ch_1x, g_ch_1x, 1) for _ in range(3)])
        self.feature_extractor = FeatureExtractor(inplace=inplace)
        self.context_fusion_net = MultiScaleContextFusion(inplace=inplace)

        self.contextual_encoder = ContextualEncoder(inplace=inplace)

        self.contextual_hyper_prior_encoder, self.contextual_hyper_prior_decoder = \
            get_hyper_enc_dec_models(g_ch_16x, g_ch_16x, True, inplace=inplace)

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(g_ch_4x, g_ch_8x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1),
        )

        self.y_prior_fusion_adaptor_0 = DepthConvBlock(g_ch_16x * 2, g_ch_16x * 3,
                                                       inplace=inplace)
        self.y_prior_fusion_adaptor_1 = DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3,
                                                       inplace=inplace)

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)

        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 2, inplace=inplace),
        )

        self.contextual_decoder = ContextualDecoder(inplace=inplace)
        self.recon_generation_net = ReconGeneration(inplace=inplace)

        self.mv_y_q_basic_enc = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_enc_fine = None
        self.mv_y_q_basic_dec = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.mv_y_q_scale_dec_fine = None

        self.y_q_basic_enc = nn.Parameter(torch.ones((1, g_ch_2x * 2, 1, 1)))
        self.y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_enc_fine = None
        self.y_q_basic_dec = nn.Parameter(torch.ones((1, g_ch_2x, 1, 1)))
        self.y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_dec_fine = None

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)

        with torch.no_grad():
            mv_y_q_scale_enc_fine = np.linspace(np.log(self.mv_y_q_scale_enc[0, 0, 0, 0]),
                                                np.log(self.mv_y_q_scale_enc[3, 0, 0, 0]), 64)
            self.mv_y_q_scale_enc_fine = np.exp(mv_y_q_scale_enc_fine)
            mv_y_q_scale_dec_fine = np.linspace(np.log(self.mv_y_q_scale_dec[0, 0, 0, 0]),
                                                np.log(self.mv_y_q_scale_dec[3, 0, 0, 0]), 64)
            self.mv_y_q_scale_dec_fine = np.exp(mv_y_q_scale_dec_fine)

            y_q_scale_enc_fine = np.linspace(np.log(self.y_q_scale_enc[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_enc[3, 0, 0, 0]), 64)
            self.y_q_scale_enc_fine = np.exp(y_q_scale_enc_fine)
            y_q_scale_dec_fine = np.linspace(np.log(self.y_q_scale_dec[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_dec[3, 0, 0, 0]), 64)
            self.y_q_scale_dec_fine = np.exp(y_q_scale_dec_fine)

    def multi_scale_feature_extractor(self, dpb, index):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            index = index % 4
            index_map = [0, 1, 0, 2]
            index = index_map[index]
            feature = self.feature_adaptor[index](dpb["ref_feature"])
        return self.feature_extractor(feature)

    def motion_compensation(self, dpb, mv, index):
        warpframe = flow_warp(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb, index)
        context1_init = flow_warp(ref_feature1, mv)
        context1 = self.align(ref_feature1, torch.cat(
            (context1_init, warpframe, mv), dim=1), mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scale_enc = ckpt["y_q_scale_enc"].reshape(-1)
        y_q_scale_dec = ckpt["y_q_scale_dec"].reshape(-1)
        mv_y_q_scale_enc = ckpt["mv_y_q_scale_enc"].reshape(-1)
        mv_y_q_scale_dec = ckpt["mv_y_q_scale_dec"].reshape(-1)
        return y_q_scale_enc, y_q_scale_dec, mv_y_q_scale_enc, mv_y_q_scale_dec

    def mv_prior_param_decoder(self, mv_z_hat, dpb, slice_shape=None):
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        mv_params = self.slice_to_y(mv_params, slice_shape)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            mv_params = self.mv_y_prior_fusion_adaptor_0(mv_params)
        else:
            mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
            mv_params = self.mv_y_prior_fusion_adaptor_1(mv_params)
        mv_params = self.mv_y_prior_fusion(mv_params)
        return mv_params

    def res_prior_param_decoder(self, z_hat, dpb, context3, slice_shape=None):
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        hierarchical_params = self.slice_to_y(hierarchical_params, slice_shape)
        temporal_params = self.temporal_prior_encoder(context3)
        ref_y = dpb["ref_y"]
        if ref_y is None:
            params = torch.cat((temporal_params, hierarchical_params), dim=1)
            params = self.y_prior_fusion_adaptor_0(params)
        else:
            params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
            params = self.y_prior_fusion_adaptor_1(params)
        params = self.y_prior_fusion(params)
        return params

    def get_recon_and_feature(self, y_hat, context1, context2, context3, y_q_dec):
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3, y_q_dec)
        feature, x_hat = self.recon_generation_net(recon_image_feature, context1)
        x_hat = x_hat.clamp_(0, 1)
        return x_hat, feature

    def motion_estimation_and_mv_encoding(self, x, dpb, mv_y_q_enc):
        est_mv = self.optic_flow(x, dpb["ref_frame"])
        ref_mv_feature = dpb["ref_mv_feature"]
        mv_y = self.mv_encoder(est_mv, ref_mv_feature, mv_y_q_enc)
        return mv_y

    def get_q_for_inference(self, q_in_ckpt, q_index):
        mv_y_q_scale_enc = self.mv_y_q_scale_enc if q_in_ckpt else self.mv_y_q_scale_enc_fine
        mv_y_q_enc = self.get_curr_q(mv_y_q_scale_enc, self.mv_y_q_basic_enc, q_index=q_index)
        mv_y_q_scale_dec = self.mv_y_q_scale_dec if q_in_ckpt else self.mv_y_q_scale_dec_fine
        mv_y_q_dec = self.get_curr_q(mv_y_q_scale_dec, self.mv_y_q_basic_dec, q_index=q_index)

        y_q_scale_enc = self.y_q_scale_enc if q_in_ckpt else self.y_q_scale_enc_fine
        y_q_enc = self.get_curr_q(y_q_scale_enc, self.y_q_basic_enc, q_index=q_index)
        y_q_scale_dec = self.y_q_scale_dec if q_in_ckpt else self.y_q_scale_dec_fine
        y_q_dec = self.get_curr_q(y_q_scale_dec, self.y_q_basic_dec, q_index=q_index)
        return mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec

    def compress(self, x, dpb, q_in_ckpt, q_index, frame_idx):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)
        mv_y = self.motion_estimation_and_mv_encoding(x, dpb, mv_y_q_enc)
        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = torch.round(mv_z)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)
        mv_y_q_w_0, mv_y_q_w_1, mv_y_q_w_2, mv_y_q_w_3, \
            mv_scales_w_0, mv_scales_w_1, mv_scales_w_2, mv_scales_w_3, mv_y_hat = \
            self.compress_four_part_prior(
                mv_y, mv_params,
                self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
                self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)
        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat, frame_idx)

        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = torch.round(z)
        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = \
            self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        self.entropy_coder.reset()
        self.bit_estimator_z_mv.encode(mv_z_hat)
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(mv_y_q_w_0, mv_scales_w_0)
        self.gaussian_encoder.encode(mv_y_q_w_1, mv_scales_w_1)
        self.gaussian_encoder.encode(mv_y_q_w_2, mv_scales_w_2)
        self.gaussian_encoder.encode(mv_y_q_w_3, mv_scales_w_3)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()

        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)
        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature": mv_feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
            "bit_stream": bit_stream,
        }
        return result

    def decompress(self, dpb, string, height, width, q_in_ckpt, q_index, frame_idx):
        _, mv_y_q_dec, _, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        self.entropy_coder.set_stream(string)
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(z_size, dtype, device)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)
        mv_y_hat = self.decompress_four_part_prior(mv_params,
                                                   self.mv_y_spatial_prior_adaptor_1,
                                                   self.mv_y_spatial_prior_adaptor_2,
                                                   self.mv_y_spatial_prior_adaptor_3,
                                                   self.mv_y_spatial_prior)

        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)
        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat, frame_idx)

        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)

        return {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_mv_feature": mv_feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
        }

    def encode_decode(self, x, dpb, q_in_ckpt, q_index, output_path=None,
                      pic_width=None, pic_height=None, frame_idx=0):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        if output_path is not None:
            device = x.device
            torch.cuda.synchronize(device=device)
            t0 = time.time()
            encoded = self.compress(x, dpb, q_in_ckpt, q_index, frame_idx)
            encode_p(encoded['bit_stream'], q_in_ckpt, q_index, frame_idx, output_path)
            bits = filesize(output_path) * 8
            torch.cuda.synchronize(device=device)
            t1 = time.time()
            q_in_ckpt, q_index, frame_idx, string = decode_p(output_path)

            decoded = self.decompress(dpb, string, pic_height, pic_width,
                                      q_in_ckpt, q_index, frame_idx)
            torch.cuda.synchronize(device=device)
            t2 = time.time()
            result = {
                "dpb": decoded["dpb"],
                "bit": bits,
                "encoding_time": t1 - t0,
                "decoding_time": t2 - t1,
            }
            return result

        encoded = self.forward_one_frame(x, dpb, q_in_ckpt=q_in_ckpt, q_index=q_index,
                                         frame_idx=frame_idx)
        result = {
            "dpb": encoded['dpb'],
            "bit": encoded['bit'].item(),
            "encoding_time": 0,
            "decoding_time": 0,
        }
        return result

    def forward_one_frame(self, x, dpb, q_in_ckpt=False, q_index=None, frame_idx=0):
        mv_y_q_enc, mv_y_q_dec, y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        est_mv = self.optic_flow(x, dpb["ref_frame"])
        mv_y = self.mv_encoder(est_mv, dpb["ref_mv_feature"], mv_y_q_enc)

        mv_y_pad, slice_shape = self.pad_for_y(mv_y)
        mv_z = self.mv_hyper_prior_encoder(mv_y_pad)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_prior_param_decoder(mv_z_hat, dpb, slice_shape)
        _, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_four_part_prior(
            mv_y, mv_params, self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
            self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        mv_hat, mv_feature = self.mv_decoder(mv_y_hat, mv_y_q_dec)

        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat, frame_idx)

        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.contextual_hyper_prior_encoder(y_pad)
        z_hat = self.quant(z)
        params = self.res_prior_param_decoder(z_hat, dpb, context3, slice_shape)
        _, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)

        _, _, H, W = x.size()
        pixel_num = H * W

        y_for_bit = y_q
        mv_y_for_bit = mv_y_q
        z_for_bit = z_hat
        mv_z_for_bit = mv_z_hat
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y, dim=(1, 2, 3)) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "dpb": {
                    "ref_frame": x_hat,
                    "ref_feature": feature,
                    "ref_mv_feature": mv_feature,
                    "ref_y": y_hat,
                    "ref_mv_y": mv_y_hat,
                },
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                }

# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import nn


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2, inplace=False):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out = out + identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, leaky_relu_slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=inplace)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = conv1x1(in_ch, out_ch)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01, inplace=False):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dw_ch, 1, stride=stride),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2,
                                    groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)

        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class ConvFFN(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        internal_ch = max(min(in_ch * 4, 1024), in_ch * 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, internal_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
            nn.Conv2d(internal_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)


class ConvFFN2(nn.Module):
    def __init__(self, in_ch, slope=0.1, inplace=False):
        super().__init__()
        expansion_factor = 2
        slope = 0.1
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = x1 * self.relu(x2)
        return identity + self.conv_out(out)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1,
                 slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv, inplace=inplace),
            ConvFFN(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class DepthConvBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1,
                 slope_depth_conv=0.01, slope_ffn=0.1, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv, inplace=inplace),
            ConvFFN2(out_ch, slope=slope_ffn, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)

