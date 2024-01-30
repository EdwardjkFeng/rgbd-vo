"""
Classes for constructing pyramids (ImagePyramid, FeaturePyramid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision

from models.submodules import convLayer as conv
from models.submodules import fcLayer, initialize_weights

from .utils import checkpointed
from utils.visualize import create_mosaic


class ImagePyramids(nn.Module):
    """Construct the pyramids in the image / depth space"""

    def __init__(self, scales, pool="avg") -> None:
        super().__init__()
        if pool == "avg":
            self.multiscales = [nn.AvgPool2d(1 << i, 1 << i) for i in scales]
        elif pool == "max":
            self.multiscales = [nn.MaxPool2d(1 << i, 1 << i) for i in scales]
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
    def __init__(self, scales, mode="bilinear") -> None:
        super().__init__()
        self.scales = scales
        self.mode = mode

    def forward(self, x):
        B, C, H, W = x.shape
        xout = [
            func.interpolate(x, scale_factor=1 << i, mode=self.mode, align_corners=True)
            for i in self.scales
        ]
        return xout


# class ImagePyramids(nn.Module):

#     def __init__(self, scales, pool='avg') -> None:
#         super().__init__()
#         self.scales = scales
#         if pool == 'avg':
#             self.mode = 'bilinear'
#         elif pool == 'max':
#             self.mode = 'nearest'

#     def forward(self, x):
#         B, C, H, W = x.shape
#         xout = [func.interpolate(x, scale_factor=1 / (1<<i), mode=self.mode) for i in self.scales]
#         return xout


"""
Encoder model which consists of (A) view encoder, (B) feature encoder, (C) uncertainty encoder
"""
class FeaturePyramid(nn.Module):
    """
    Encoder = View Encoder -> Feature Encoder | Uncertainty Encoder
    """

    def __init__(
        self,
        view_encoder: str = "UNet",
        encode_channels: list = [32, 64, 96, 128],
        feature_channel: int = 32,
        feature_extractor: str = "conv",
        out_uncertainty: bool = True,
        uncertainty_channel: int = 1,
        uncertainty_type: str = "laplacian",
        decoder_norm: str = "nn.BatchNorm2d",
        do_checkpointed: bool = False,
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
        assert uncertainty_channel == feature_channel or uncertainty_channel == 1

        # self.encode_channels = encode_channels
        # self.uncertainty_type = uncertainty
        # self.feature_channel = feature_channel
        # self.uncertainty_channel = uncertainty_channel

        self.f_channels = [32, 64, 96, 128]

        """ =============================================================== """
        """                Initialize the View Encoder                      """
        """ =============================================================== """
        if view_encoder == "ConvRGBD2":
            self.view_encoder = ViewEncoder(
                C=8, # concatenated 2 view RGBD
                out_channels=encode_channels,
            )
        elif view_encoder == "ConvRGBD":
            self.view_encoder = ViewEncoder(
                C=4, # concatenated RGBD
                out_channels=encode_channels,
            )
        elif view_encoder == "UNet":
            self.view_encoder = UNetViewEncoder(
                out_channels=encode_channels,
            )
        else:
            raise NotImplementedError(view_encoder)

        """ =============================================================== """
        """              Initialize the Feature Encoder                     """
        """ =============================================================== """
        self.feature_encoder = FeatureEncoder(
            in_channels=encode_channels,
            out_channel=feature_channel,
            extractor=feature_extractor,
        )

        """ =============================================================== """
        """              Initialize the Uncertainty Encoder                 """
        """ =============================================================== """
        if out_uncertainty:
            self.uncertainty_encoder = UncertaintyEncoder(
                in_channels=encode_channels,
                feature_channel=feature_channel,
                uncertainty_channel=uncertainty_channel,
                uncertainty=uncertainty_type,
            )

    def forward(self, x):
        encoded_x = self.view_encoder(x)
        f = self.feature_encoder(encoded_x)
        sigma = self.uncertainty_encoder(encoded_x)

        return f, sigma, encoded_x


class ViewEncoder(nn.Module):
    """
    View encoder proposed in "Deep Probabilistic Feature-metric tracking"
    """

    def __init__(
        self,
        C: int = 4,
        out_channels: list = [32, 64, 96, 128],
    ):
        super().__init__()
        C0, C1, C2, C3 = out_channels
        self.net0 = nn.Sequential(
            conv(True, C, 16, 3),
            conv(True, 16, C0, 3, dilation=2),
            conv(True, C0, C0, 3, dilation=2),
        )
        self.net1 = nn.Sequential(
            conv(True, C0, C0, 3),
            conv(True, C0, C1, 3, dilation=2),
            conv(True, C1, C1, 3, dilation=2),
        )
        self.net2 = nn.Sequential(
            conv(True, C1, C1, 3),
            conv(True, C1, C2, 3, dilation=2),
            conv(True, C2, C2, 3, dilation=2),
        )
        self.net3 = nn.Sequential(
            conv(True, C2, C2, 3),
            conv(True, C2, C3, 3, dilation=2),
            conv(True, C3, C3, 3, dilation=2),
        )
        initialize_weights(self.net0)
        initialize_weights(self.net1)
        initialize_weights(self.net2)
        initialize_weights(self.net3)

        self.downsample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x0 = self.net0(x)
        x0s = self.downsample(x0)
        x1 = self.net1(x0s)
        x1s = self.downsample(x1)
        x2 = self.net2(x1s)
        x2s = self.downsample(x2)
        x3 = self.net3(x2s)
        out = [x0, x1, x2, x3]

        return out


class UNetViewEncoder(nn.Module):
    def __init__(
        self,
        out_channels: list = [32, 64, 96, 128],
        backbone = "vgg16",
        do_checkpointed = True,
        num_downsample = 4,
    ):
        super().__init__()
        # Encoder
        self.rgbd_adapter = AdaptationBlock(4, 3)

        self.encoder, skip_dims = self.build_encoder(
            backbone, num_downsample, do_checkpointed
        )

        # Decoder
        if out_channels is not None:
            assert len(out_channels) == (len(skip_dims) - 1)
            Block = checkpointed(DecoderBlock, do=do_checkpointed)
            norm = nn.BatchNorm2d

            previous = skip_dims[-1]
            decoder = []
            for out, skip in zip(out_channels, skip_dims[:-1][::-1]):
                decoder.append(Block(previous, skip, out, norm=norm))
                previous = out
            self.decoder = nn.ModuleList(decoder)

    def build_encoder(
        self,
        backbone,
        num_downsample,
        do_checkpointed,
        ):

        Encoder = getattr(torchvision.models, backbone)
        encoder = Encoder(weights='DEFAULT')
        Block = checkpointed(torch.nn.Sequential, do=do_checkpointed)

        if backbone.startswith('vgg'):
            # Parse the layers and pack them into downsampling blocks
            # It's easy for VGG-style nets because of their linear structure.
            # This does not handle strided convs and residual connections
            skip_dims = []
            previous_dim = None
            blocks = [[]]
            for i, layer in enumerate(encoder.features):
                if isinstance(layer, torch.nn.Conv2d):
                    previous_dim = layer.out_channels
                elif isinstance(layer, torch.nn.MaxPool2d):
                    assert previous_dim is not None
                    skip_dims.append(previous_dim)
                    if (num_downsample + 1) == len(blocks):
                        break
                    blocks.append([])
                blocks[-1].append(layer)
            assert (num_downsample + 1) == len(blocks)
            encoder = [Block(*b) for b in blocks]
        elif backbone.startswith('resnet'):
            # Manually define the splits - this could be improved
            assert backbone[len('resnet'):] in ['18', '34', '50', '101']
            block1 = torch.nn.Sequential(encoder.conv1, encoder.bn1,
                                         encoder.relu)
            block2 = torch.nn.Sequential(encoder.maxpool, encoder.layer1)
            block3 = encoder.layer2
            block4 = encoder.layer3
            blocks = [block1, block2, block3, block4]
            encoder = [torch.nn.Identity()] + [Block(b) for b in blocks]
            if backbone[len('resnet'):] in ['50', '101']:
                skip_dims = [3, 64, 256, 512, 1024]
            elif backbone[len('resnet'):] in ['18', '34']:
                skip_dims = [3, 64, 64, 128, 256]
        else:
            raise NotImplementedError(backbone)
        
        encoder = nn.ModuleList(encoder)
        return encoder, skip_dims
        
    def forward(self, x):
        skip_features = []
        features = x
        if features.shape[1] == 4:
            features = self.rgbd_adapter(features)
            # cv2.namedWindow("adapter input", cv2.WINDOW_NORMAL)
            # cv2.imshow("adapter input", create_mosaic(
            #     [features],
            #     cmap=cv2.COLORMAP_JET, 
            #     normalize=True,
            # ))

        for block in self.encoder:
            features = block(features)
            print("Encoder layers: {}".format(features.shape))
            skip_features.append(features)
            # Show backbone features
            # cv2.namedWindow("backbone features", cv2.WINDOW_NORMAL)
            # cv2.imshow("backbone features", create_mosaic(
            #     [features],
            #     cmap=cv2.COLORMAP_JET, 
            #     normalize=True,
            # ))

        if self.decoder:
            pre_features = [skip_features[-1]]
            for block, skip in zip(self.decoder, skip_features[:-1][::-1]):
                pre_features.append(block(pre_features[-1], skip))
                print("Decoder layers: {}".format(pre_features[-1].shape))
            out_features = pre_features[-4:] # fine to coarse
        else:
            out_features = skip_features

        return out_features


class FeatureEncoder(nn.Module):
    """
    Feature encoder
    """

    def __init__(
        self,
        in_channels: list = [32, 64, 96, 128],
        out_channel: int = 1,
        extractor: str = "conv",
    ):
        super().__init__()
        self.output_C = out_channel
        self.feature_extract = extractor

        if self.feature_extract != "average" and self.feature_extract != "skip":
            if self.feature_extract == "conv":
                self.f_conv0 = conv(
                    True, in_channels[0], self.output_C, kernel_size=1
                )
                self.f_conv1 = conv(
                    True, in_channels[1], self.output_C, kernel_size=1
                )
                self.f_conv2 = conv(
                    True, in_channels[2], self.output_C, kernel_size=1
                )
                self.f_conv3 = conv(
                    True, in_channels[3], self.output_C, kernel_size=1
                )
            elif self.feature_extract == "1by1":
                self.f_conv0 = nn.Conv2d(
                    in_channels[0], self.output_C, kernel_size=(1, 1)
                )
                self.f_conv1 = nn.Conv2d(
                    in_channels[1], self.output_C, kernel_size=(1, 1)
                )
                self.f_conv2 = nn.Conv2d(
                    in_channels[2], self.output_C, kernel_size=(1, 1)
                )
                self.f_conv3 = nn.Conv2d(
                    in_channels[3], self.output_C, kernel_size=(1, 1)
                )
            elif self.feature_extract == "prob_fuse":
                self.output_C = 8 * 2
                self.f_conv0 = conv(
                    True, in_channels[0], self.output_C, kernel_size=1
                )
                self.f_conv1 = conv(
                    True, in_channels[1], self.output_C, kernel_size=1
                )
                self.f_conv2 = conv(
                    True, in_channels[2], self.output_C, kernel_size=1
                )
                self.f_conv3 = conv(
                    True, in_channels[3], self.output_C, kernel_size=1
                )
            else:
                raise NotImplementedError("not supported feature extraction option")
            initialize_weights((self.f_conv0, self.f_conv1, self.f_conv2, self.f_conv3))

    def __prob_fuse(self, x, conv):
        x_ = conv(x)
        B, C, H, W = x_.shape
        f, p = x_.split(int(C / 2), dim=1)
        p = func.sigmoid(p)
        out = torch.sum(f * p, dim=1, keepdim=True)
        return out

    def forward(self, x):
        x0, x1, x2, x3 = x

        if self.feature_extract == "skip":
            f = [x0, x1, x2, x3]
        elif self.feature_extract == "average":
            x = (x0, x1, x2, x3)
            f = [self.__Nto1(a) for a in x]
        elif self.feature_extract == "prob_fuse":
            f0 = self.__prob_fuse(x0, self.f_conv0)
            f1 = self.__prob_fuse(x1, self.f_conv1)
            f2 = self.__prob_fuse(x2, self.f_conv2)
            f3 = self.__prob_fuse(x3, self.f_conv3)
            f = [f0, f1, f2, f3]
        elif self.feature_extract in ("conv", "1by1"):
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
        in_channels: list = [32, 64, 96, 128],
        feature_channel: int = 1,
        uncertainty_channel: int = 1,
        uncertainty: str = "laplacian",
    ):
        super().__init__()
        self.feature_C = feature_channel
        self.uncertainty_C = uncertainty_channel
        self.uncertainty_type = uncertainty
        print(uncertainty_channel)

        if self.uncertainty_type != "identity":
            if self.uncertainty_type == "feature":
                self.sigma_conv0 = conv(True, in_channels[0], self.feature_C, 1)
                self.sigma_conv1 = conv(True, in_channels[1], self.feature_C, 1)
                self.sigma_conv2 = conv(True, in_channels[2], self.feature_C, 1)
                self.sigma_conv3 = conv(True, in_channels[3], self.feature_C, 1)
            elif self.uncertainty_type in (
                "gaussian",
                "laplacian",
                "old_gaussian",
                "old_laplacian",
                "sigmoid",
            ):
                self.sigma_conv0 = nn.Sequential(
                    conv(True, in_channels[0], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
                self.sigma_conv1 = nn.Sequential(
                    conv(True, in_channels[1], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
                self.sigma_conv2 = nn.Sequential(
                    conv(True, in_channels[2], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
                self.sigma_conv3 = nn.Sequential(
                    conv(True, in_channels[3], 16, kernel_size=1),
                    nn.Conv2d(16, self.uncertainty_C, kernel_size=1),
                )
            initialize_weights(
                (self.sigma_conv0, self.sigma_conv1, self.sigma_conv2, self.sigma_conv3)
            )

    def forward(self, x):
        x0, x1, x2, x3 = x
        B = x0.shape[0]

        if self.uncertainty_type == "feature":
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
        elif self.uncertainty_type == "identity":
            sigma = [torch.ones((B, 1, *f_i.shape[-2:])).to(f_i) for f_i in x]
            # sigma = [torch.ones_like(f_i) for f_i in x]
        elif self.uncertainty_type == "gaussian":
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [
                torch.exp(0.5 * torch.clamp(sigma_i, min=-6, max=6))
                for sigma_i in sigma
            ]
        elif self.uncertainty_type == "laplacian":
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [
                torch.exp(torch.clamp(sigma_i, min=-3, max=3)) for sigma_i in sigma
            ]
        elif self.uncertainty_type == "sigmoid":
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [torch.sigmoid(sigma_i) for sigma_i in sigma]
        elif self.uncertainty_type == "old_gaussian":
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [
                torch.exp(0.5 * torch.clamp(sigma_i, min=1e-3, max=1e3))
                for sigma_i in sigma
            ]
        elif self.uncertainty_type == "old_laplacian":
            sigma0 = self.sigma_conv0(x0)
            sigma1 = self.sigma_conv1(x1)
            sigma2 = self.sigma_conv2(x2)
            sigma3 = self.sigma_conv3(x3)
            sigma = [sigma0, sigma1, sigma2, sigma3]
            sigma = [
                torch.exp(torch.clamp(sigma_i, min=1e-3, max=1e3)) for sigma_i in sigma
            ]
        else:
            raise NotImplementedError()

        if (
            self.uncertainty_C == 1
            and self.uncertainty_type != "identity"
            and self.feature_C != 1
        ):
            sigma = [sigma_i.repeat((1, self.feature_C, 1, 1)) for sigma_i in sigma]

        return sigma


class DecoderBlock(nn.Module):
    def __init__(self, previous, skip, out, num_convs=1, norm=nn.BatchNorm2d):
        super().__init__()

        self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False)

        layers = []
        for i in range(num_convs):
            conv = nn.Conv2d(
                previous+skip if i == 0 else out, out,
                kernel_size=3, padding=1, bias=norm is None)
            layers.append(conv)
            if norm is not None:
                layers.append(norm(out))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, previous, skip):
        upsampled = self.upsample(previous)
        # If the shape of the input map `skip` is not a multiple of 2,
        # it will not match the shape of the upsampled map `upsampled`.
        # If the downsampling uses ceil_mode=False, we nedd to crop `skip`.
        # If it uses ceil_mode=True (not supported here), we should pad it.
        _, _, hu, wu = upsampled.shape
        _, _, hs, ws = skip.shape
        assert (hu <= hs) and (wu <= ws), 'Using ceil_mode=True in pooling?'
        # assert (hu == hs) and (wu == ws), 'Careful about padding'
        skip = skip[:, :, :hu, :wu]
        return self.layers(torch.cat([upsampled, skip], dim=1))
    

class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)



if __name__ == '__main__':

    from dataset import dataloader
    import argparse
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    import torchvision.utils as torch_utils


    def color_normalize(image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean, std = image.new_tensor(mean), image.new_tensor(std)
        image = (image - mean[:, None, None]) / std[:, None, None]
        return image

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default="config/default.yaml")
    args = parser.parse_args()

    if args.conf:
        conf = OmegaConf.load(args.conf)

    loader = dataloader.load_data(conf.data["name"], conf.data)
    
    torch_loader = DataLoader(
        loader,
        shuffle=False, 
        num_workers=4
    )
    
    net = FeaturePyramid(
        view_encoder=conf.model.view_encoder,
        encode_channels=conf.model.decoder,
        feature_channel=conf.model.output_dim,
        out_uncertainty=conf.model.compute_uncertainty,
        decoder_norm=conf.model.decoder_norm,
        do_checkpointed=conf.model.checkpointed,
    ).cuda()
    with torch.no_grad():
        import cv2
        for batch in torch_loader:
            color0, color1, depth0, depth1, transform, calib = batch['data']
            B, C, H, W = color0.shape

            invD0 = torch.clamp(1.0 / depth0, 0, 10)
            cat_rgbd = torch.cat((color_normalize(color0), invD0), dim=1).to(torch.float32).cuda() * 255

            feat_maps, uncer_maps, _ = net(color0.to(torch.float32).cuda()*255)
            # feat_maps, uncer_maps, _ = net(cat_rgbd)
            print([uncer_maps[i].shape for i in range(len(conf.model.output_scales))])
            feat_maps = torch_utils.make_grid(
                feat_maps[-1].view(-1, 1, H, W),
                nrow=4
            )

            feat_maps = feat_maps.detach().cpu().numpy().transpose(1, 2, 0)

            uncer_maps = torch_utils.make_grid(
                uncer_maps[-1][:, 0].view(-1, 1, H, W),
                nrow=1
            )

            uncer_maps = uncer_maps.detach().cpu().numpy().transpose(1, 2, 0)

            # image = torch_utils.make_grid(color0, nrow=4)
            # image = image.numpy().transpose(1, 2, 0)[:, :, [2, 1, 0]]

            
            cv2.namedWindow("uncertainty maps", cv2.WINDOW_NORMAL)
            cv2.imshow("uncertainty maps", create_mosaic(
                [uncer_maps],
                cmap=cv2.COLORMAP_JET, 
                # normalize=True,
            ))
            cv2.namedWindow("Concatenated RGBD", cv2.WINDOW_NORMAL)
            cv2.imshow("Concatenated RGBD", create_mosaic(
                [cat_rgbd],
                cmap=cv2.COLORMAP_JET, 
                # normalize=True,
            ))
            cv2.namedWindow("Input RGBD", cv2.WINDOW_NORMAL)
            cv2.imshow("Input RGBD", create_mosaic(
                [color0, depth0],
                cmap=cv2.COLORMAP_JET, 
                normalize=True,
            ))
            k = cv2.waitKey(10) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
        cv2.waitKey(0)
        cv2.destroyAllWindows()