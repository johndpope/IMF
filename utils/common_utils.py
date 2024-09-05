import torch
import random
from torch.nn import functional as F
from PIL import Image


class AlignerCantFindFaceError(Exception):
    pass

class MaskerCantFindFaceError(Exception):
    pass


def tensor2im(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = (var + 1) / 2
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype("uint8"))


def tensor2im_no_tfm(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = var * 255
    return Image.fromarray(var.astype("uint8"))


def printer(obj, tabs=0):
    for (key, value) in obj.items():
        try:
            _ = value.items()
            print(" " * tabs + str(key) + ":")
            printer(value, tabs + 4)
        except:
            print(f" " * tabs + str(key) + " : " + str(value))


def get_keys(d, name, key="state_dict"):
    if key in d:
        d = d[key]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name) + 1] == name + '.'}
    return d_filt


def setup_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
