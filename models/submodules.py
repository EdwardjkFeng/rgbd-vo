"""
Submodules to build up CNN
"""
import torch.nn as nn
import torch
import numpy as np

from torch.nn import init
from torchvision import transforms


def color_normalize(color):
    # mean = torch.Tensor([0.485, 0.456, 0.406]).type_as(color)
    # std = torch.Tensor([0.229, 0.224, 0.225]).type_as(color)
    # return (color - mean[:, None, None]) / std[:, None, None]
    rgb_mean = torch.Tensor([0.4914, 0.4822, 0.4465]).type_as(color)
    rgb_std = torch.Tensor([0.2023, 0.1994, 0.2010]).type_as(color)
    return (color - rgb_mean.view(1, 3, 1, 1)) / rgb_std.view(1, 3, 1, 1)


def convLayer(
    batchNorm,
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    dilation=1,
    bias=False,
    mode="2d",
):
    """A wrapper of convolution-batchnorm-ReLU module"""
    conv = {"1d": nn.Conv1d, "2d": nn.Conv2d}
    bn = {"1d": nn.BatchNorm1d, "2d": nn.BatchNorm2d}
    if batchNorm:
        return nn.Sequential(
            conv[mode](
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2 + dilation - 1,
                dilation=dilation,
                bias=bias,
            ),
            bn[mode](out_planes),
            nn.ELU(inplace=True),
        )
    else:
        return nn.Sequential(
            conv[mode](
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2 + dilation - 1,
                dilation=dilation,
                bias=True,
            ),
            nn.ELU(inplace=True),
        )


def fcLayer(in_features, out_features, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias), nn.ReLU(inplace=True)
    )


def initialize_weights(modules, method="xavier"):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if m.bias is not None:
                m.bias.data.zero_()
            if method == "xavier":
                init.xavier_uniform_(m.weight)
            elif method == "kaiming":
                init.kaiming_uniform_(m.weight)
            else:
                raise NotImplementedError()
