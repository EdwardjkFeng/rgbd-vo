import numpy as np
import torch
import cv2


def resize(image, new_size, interp):
    if interp == 'bilinear':
        mode = cv2.INTER_LINEAR
    elif interp == 'nearest':
        mode = cv2.INTER_NESREST
    else:
        raise NotImplementedError
    return cv2.resize(image, new_size, interpolation=mode)


def numpy_image_to_torch(img_numpy):
    img_torch = torch.from_numpy(img_numpy).permute(2, 0, 1)
    return img_torch 