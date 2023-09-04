""" Utility functions """

import os.path as osp

import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as transF


def resize(image, size, interp):
    """ Resize an image to a fixed size, or according to max edge. """
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size/ max(h, w)
        h_new, w_new = [int(round(h * scale)), int(round(w * scale))]
        scale = (scale, scale)
    elif isinstance(size, float):
        scale = size
        h_new, w_new = [int(round(h * scale)), int(round(w * scale))]
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST,
        'cubic': cv2.INTER_CUBIC
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def numpy_image_to_torch(image):
    """ Noramlize image tensor and reorder the dimensions """
    if image.ndim == 3:
        image = image.transpose((2, 0, 1)) # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None] # add channel dim
    else:
        raise ValueError(f'Input with shape {image.shape} is not an expected image.')
    img_torch = torch.from_numpy(image.astype(np.float32))
    return img_torch


def crop(image, size, *, random=True, other=None, camera=None, return_bbox=False, centroid=None):
    """ Random or deterministic crop of an image, adjust depth and intrinsics. """
    h, w = image.shape[:2]
    h_new, w_new = (size, size) if isinstance(size, int) else size
    if random:
        top = np.random.randint(0, h - h_new + 1)
        left = np.random.randint(0, w - w_new + 1)
    elif centroid is not None:
        x, y = centroid
        top = np.clip(int(y) - h_new // 2, 0, h - h_new)
        left = np.clip(int(x) - w_new // 2, 0, w - w_new)
    else:
        top = left = 0

    image = image[top:top+h_new, left:left+w_new]
    ret = [image]
    if other is not None:
        ret += [other[top:top+h_new, left:left+w_new]]
    if camera is not None:
        ret += [camera.crop[(top, left), (h_new, w_new)]]
    if return_bbox:
        ret += [(top, top+h_new, left, left+w_new)]
    return ret


def read_image(path, grayscale=None):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    return image


def normalize_color(images):
    """ Normalize colors """
    mean = torch.as_tensor([0.4914, 0.4822, 0.4465], device=images.device)
    std = torch.as_tensor([0.2023, 0.1994, 0.2010], device=images.device)
    return (images/255.0).sub_(mean[:, None, None]).div_(std[:, None, None])
