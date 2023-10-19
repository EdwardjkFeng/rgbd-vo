"""
Encoder model which consists of (A) view encoder, (B) feature encoder, (C) uncertainty encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models

from models.submodules import convLayer as conv
from models.submodules import fcLayer, initialize_weights


class Encoder(nn.Module):
    """
    Encoder = View Encoder -> Feature Encoder | Uncertainty Encoder
    """
    def __init__(
        self,
        C: int = 4,
        encode_channels: list = [32, 64, 96, 128],
        feature_channel: int = 1,
        feature_extractor: str = 'conv',
        uncertainty_channel: int = 1,
        uncertainty: str = 'identity',
    ):
        """_summary_

        Args:
            C: _description_. Defaults to 4.
            uncertainty: _description_. Defaults to 'None'.
            feature_channel: _description_. Defaults to 1.
            feature_extractor: _description_. Defaults to 'conv'.
            uncertainty_channel: _description_. Defaults to 1.
        """
        super().__init__()
        assert uncertainty_channel == feature_channel \
            or uncertainty_channel == 1

        self.encode_channels = encode_channels
        self.uncertainty_type = uncertainty
        self.feature_channel = feature_channel
        self.uncertainty_channel = uncertainty_channel

        self.f_channels = [32, 64, 96, 128]

        """ =============================================================== """
        """                Initialize the View Encoder                      """
        """ =============================================================== """
        self.view_encoder = ViewEncoder(
            C=C, 
            out_channels=self.encode_channels,
        )

        """ =============================================================== """
        """              Initialize the Feature Encoder                     """
        """ =============================================================== """
        self.feature_encoder = FeatureEncoder(
            out_channel=self.feature_channel, 
            extractor=self.feature_extractor,
        )

        """ =============================================================== """
        """              Initialize the Uncertainty Encoder                 """
        """ =============================================================== """
        if self.out_uncertainty:
            self.uncertainty_encoder = UncertaintyEncoder(
                feature_channel=self.feature_channel,
                uncertainty_channel=self.uncertainty_channel,
                uncertainty=self.uncertainty_type,
            )

    def forward(self, x):
        encoded_x = self.view_encoder(x)
        f = self.feature_encoder(encoded_x)
        sigma = self.uncertainty_encoder(encoded_x)
            
        return f, sigma, encoded_x


class ViewEncoder(nn.Module):
    """
    View encoder
    """
    def __init__(
        self,
        C: int = 4,
        out_channels: list = [32, 64, 96, 128],
    ):
        self.net0 = nn.Sequential(
            conv(True, C,  16, 3), 
            conv(True, 16, 32, 3, dilation=2),
            conv(True, 32, self.f_channels[0], 3, dilation=2))
        self.net1 = nn.Sequential(
            conv(True, 32, 32, 3),
            conv(True, 32, 64, 3, dilation=2),
            conv(True, 64, self.f_channels[1], 3, dilation=2))
        self.net2 = nn.Sequential(
            conv(True, 64, 64, 3),
            conv(True, 64, 96, 3, dilation=2),
            conv(True, 96, self.f_channels[2], 3, dilation=2))
        self.net3 = nn.Sequential(
            conv(True, 96, 96, 3),
            conv(True, 96, 128, 3, dilation=2),
            conv(True, 128, self.f_channels[3], 3, dilation=2))
        initialize_weights(self.net0)
        initialize_weights(self.net1)
        initialize_weights(self.net2)
        initialize_weights(self.net3)

        self.downsample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x0 = self.net0(x)
        x0s= self.downsample(x0)
        x1 = self.net1(x0s)
        x1s= self.downsample(x1)
        x2 = self.net2(x1s)
        x2s= self.downsample(x2)
        x3 = self.net3(x2s)
        out = [x0, x1, x2, x3]

        return out


class FeatureEncoder(nn.Module):
    """
    Feature encoder
    """
    def __init__(
        self,
        out_channel: int = 1,
        extractor: str = 'conv',
    ):
        super().__init__()
        self.output_C = out_channel
        self.extractor = extractor

        if self.extractor != 'average' and self.extractor != 'skip':
            if self.extractor == 'conv':
                self.f_conv0 = conv(True, self.f_channels[0], self.output_C, kernel_size=1)
                self.f_conv1 = conv(True, self.f_channels[1], self.output_C, kernel_size=1)
                self.f_conv2 = conv(True, self.f_channels[2], self.output_C, kernel_size=1)
                self.f_conv3 = conv(True, self.f_channels[3], self.output_C, kernel_size=1)
            elif self.feature_extract == '1by1':
                self.f_conv0 = nn.Conv2d(self.f_channels[0], self.output_C, kernel_size=(1, 1))
                self.f_conv1 = nn.Conv2d(self.f_channels[1], self.output_C, kernel_size=(1, 1))
                self.f_conv2 = nn.Conv2d(self.f_channels[2], self.output_C, kernel_size=(1, 1))
                self.f_conv3 = nn.Conv2d(self.f_channels[3], self.output_C, kernel_size=(1, 1))
            elif self.feature_extract == 'prob_fuse':
                self.output_C = 8 * 2
                self.f_conv0 = conv(True, self.f_channels[0], self.output_C, kernel_size=1)
                self.f_conv1 = conv(True, self.f_channels[1], self.output_C, kernel_size=1)
                self.f_conv2 = conv(True, self.f_channels[2], self.output_C, kernel_size=1)
                self.f_conv3 = conv(True, self.f_channels[3], self.output_C, kernel_size=1)
            else:
                raise NotImplementedError("not supported feature extraction option")
            initialize_weights((self.f_conv0, self.f_conv1, self.f_conv2, self.f_conv3))
    
    def __prob_fuse(self, x, conv):
        x_ = conv(x)
        B, C, H, W = x_.shape
        f, p = x_.split(int(C/2), dim=1)
        p = func.sigmoid(p)
        out = torch.sum(f * p, dim=1, keepdim=True)
        return out

    def forward(self, x):
        x0, x1, x2, x3 = x

        if self.feature_extract == 'skip':
            f = [x0, x1, x2, x3]
        elif self.feature_extract == 'average':
            x = (x0, x1, x2, x3)
            f = [self.__Nto1(a) for a in x]
        elif self.feature_extract == 'prob_fuse':
            f0 = self._prob_fuse(x0, self.f_conv0)
            f1 = self._prob_fuse(x1, self.f_conv1)
            f2 = self._prob_fuse(x2, self.f_conv2)
            f3 = self._prob_fuse(x3, self.f_conv3)
            f = [f0, f1, f2, f3]
        elif self.feature_extract in ('conv', '1by1'):
            f0 = self.f_conv0(x0)
            f1 = self.f_conv1(x1)
            f2 = self.f_conv2(x2)
            f3 = self.f_conv3(x3)
            f = [f0, f1, f2, f3]

        return f


class UncertaintyEncoder(nn.Module):
    """
    Uncertainty encoder
    """
    def __init__(
        self,
        feature_channel: int = 1,
        uncertainty_channel: int = 1,
        uncertainty: str = 'identity',
    ):
        self.feature_C = feature_channel
        self.uncertainty_C = uncertainty_channel
        self.uncertainty_type = uncertainty

        if self.uncertainty != 'identity':
            if self.uncertainty_type == 'feature':
                self.sigma_conv0 = conv(True, self.f_channels[0], self.feature_C, 1)
                self.sigma_conv1 = conv(True, self.f_channels[1], self.feature_C, 1)
                self.sigma_conv2 = conv(True, self.f_channels[2], self.feature_C, 1)
                self.sigma_conv3 = conv(True, self.f_channels[3], self.feature_C, 1)
            elif self.uncertainty_type in ('gaussian', 'laplacian', 'old_gaussian', 'old_laplacian', 'sigmoid'):
                self.sigma_conv0 = nn.Sequential(
                    conv(True, self.f_channels[0], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
                self.sigma_conv1 = nn.Sequential(
                    conv(True, self.f_channels[1], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
                self.sigma_conv2 = nn.Sequential(
                    conv(True, self.f_channels[2], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
                self.sigma_conv3 = nn.Sequential(
                    conv(True, self.f_channels[3], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
            initialize_weights((self.sigma_conv0, self.sigma_conv1, self.sigma_conv2, self.sigma_conv3))


    def forward(self, x):
        x0, x1, x2, x3 = x

        if self.uncertainty_type == 'feature':
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
        elif self.uncertainty_type == 'identity':
            sigma = [torch.ones_like(f_i) for f_i in f]
        elif self.uncertainty_type == 'gaussian':
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [torch.exp(0.5 * torch.clamp(sigma_i, min=-6, max=6)) for sigma_i in sigma]
        elif self.uncertainty_type == 'laplacian':
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [torch.exp(torch.clamp(sigma_i, min=-3, max=3)) for sigma_i in sigma]
        elif self.uncertainty_type == 'sigmoid':
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [torch.sigmoid(sigma_i) for sigma_i in sigma]
        elif self.uncertainty_type == 'old_gaussian':
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [torch.exp(0.5 * torch.clamp(sigma_i, min=1e-3, max=1e3)) for sigma_i in sigma] 
        elif self.uncertainty_type == 'old_laplacian':
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [torch.exp(torch.clamp(sigma_i, min=1e-3, max=1e3)) for sigma_i in sigma] 
        else:
            raise NotImplementedError()

        if self.uncertainty_C == 1 and self.uncertainty_type != 'identity' and self.feature_C != 1:
            sigma = [
                sigma_i.repeat((1, self.feature_C, 1, 1)) for sigma_i in sigma
            ]

        return sigma