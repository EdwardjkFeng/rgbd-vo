import cv2
import numpy as np
import torch
import torch.nn.functional as func
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image

from dataset.tum_rgbd import TUM
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def feature_gradient(F, normalize_gradient=True):
    """Calcualte the gradient on the feature space using Sobel operator

    Args:
        img: input image
        normalize_gradient: whether to normalized the gradient. Defaults to True.
    """
    B, C, H, W = F.shape

    sobel = torch.asarray([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]]).view(1, 1, 3, 3).to(F)

    F_pad = func.pad(F.view(-1, 1, H, W), (1, 1, 1, 1), mode="replicate")
    dF_dx = func.conv2d(F_pad, sobel, stride=1, padding=0)
    dF_dy = func.conv2d(F_pad, sobel.transpose(2, 3), stride=1, padding=0)

    if normalize_gradient:
        mag = torch.sqrt((dF_dx**2) + (dF_dy**2) + 1e-8)
        dF_dx = dF_dx / mag
        dF_dy = dF_dy / mag

    return dF_dx.view(B, C, H, W), dF_dy.view(B, C, H, W)


class Args():
    pass

args = Args()
args.conf = './config/f2f_dia.yaml'

if args.conf:
    conf =OmegaConf.load(args.conf)

loader = TUM(conf.data).get_dataset()

torch_loader = DataLoader(
    loader,
    shuffle=False,
    num_workers=4,
)

for batch in torch_loader:
    image1, image2, depth1, depth2, _, _ = batch['data']
    B, C, H, W = image1.shape

    d_depth1 = torch.cat(feature_gradient(depth1), dim=1)
    print(d_depth1[:, 0].mean(), d_depth1[:, 1].mean())
    d_depth1 = torch.where(d_depth1 > 1.5 * d_depth1.mean(), d_depth1, 0.0)
    print(d_depth1.shape)
    gradient = flow_to_image(d_depth1)
    print(gradient.shape)

    plt.figure()
    plt.imshow(gradient.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.show()