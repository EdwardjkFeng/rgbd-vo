"""
Classes for constructing pyramids (ImagePyramid, FeaturePyramid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as func


class ImagePyramids(nn.Module):
    """Construct the pyramids in the image / depth space"""
    def __init__(self, scales, pool='avg') -> None:
        super().__init__()
        if pool == 'avg':
            self.multiscales = [nn.AvgPool2d(1<<i, 1<<i) for i in scales]
        elif pool == 'max':
            self.multiscales = [nn.MaxPool2d(1<<i, 1<<i) for i in scales]
        else:
            raise NotImplementedError()

    def forward(self, x):
        if x.dtype == torch.bool:
            x = x.to(torch.float32)
            x_out = [f(x).to(torch.bool) for f in self.multiscales]
        else:
            x_out = [f(x) for f in self.multiscales]
        return x_out
    

class ImagePyramidsInterpolate(nn.Module):
    
    def __init__(self, scales, mode='bilinear') -> None:
        super().__init__()
        self.scales = scales
        self.mode = mode
    
    def forward(self, x):
        B, C, H, W = x.shape
        xout = [func.interpolate(x, scale_factor=1<<i, mode=self.mode, align_corners=True) for i in self.scales]
        return xout
    