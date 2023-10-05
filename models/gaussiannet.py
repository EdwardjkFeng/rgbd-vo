"""
A dumpy model that computes an image pyramid with appropriate blurring.
"""

import torch
import kornia

from .base_model import BaseModel

class GaussianNet(BaseModel):
    default_conf = {
        'output_scales': [1, 4, 16],  # what scales to adapt and output
        'kernel_size_factor': 3,
    }

    def _init(self, conf):
        self.scales = conf['output_scales']

    def _forward(self, date):
        image = data['image']
        scale_prev = 1
        pyramid = []
        for scale in self.conf.output_scales:
            sigma = scale / scale_prev
            ksize = int(self.conf.kernel_size_factor * sigma)
            if ksize % 2 == 0:
                ksize -= 1
            image = kornia.filters.gaussian_blur2d(
                image, kernel_size=(ksize, ksize), sigma=(sigma, sigma)
            )
            if sigma != 1:
                image = torch.nn.functional.interpolate(
                    image, scale_factor = 1/sigma, mode='bilinear',
                    align_corners = False
                )
            pyramid.append(image)
            scale_prev = scale
        return {'feature_maps': pyramid}

    def loss(self, pred, data):
        raise NotImplementedError
    
    def metrics(self, pred, data):
        raise NotImplementedError
    


if __name__ == '__main__':

    from dataset.tum_rgbd import TUM
    import argparse
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    import torchvision.utils as torch_utils

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str)
    args = parser.parse_args()

    if args.conf:
        conf =OmegaConf.load(args.conf)

    loader = [TUM(conf.data).get_dataset()[i] for i in range(3)]

    torch_loader = DataLoader(loader, batch_size=16,
                                   shuffle=False, num_workers=4)
    
    net = GaussianNet(conf.model)
    for batch in torch_loader:
        item = batch
        color0, color1, depth0, depth1, transform, calib = item['data']
        B, C, H, W = color0.shape

        bcolor0_img = torch_utils.make_grid(color0, nrow=4)
        data = {'image': color0}
        feat_maps = net(data)['feature_maps']
        blurred_image = torch_utils.make_grid(feat_maps[-1], nrow=4)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(bcolor0_img.numpy().transpose(1, 2, 0))
        plt.imshow(blurred_image.numpy().transpose(1, 2, 0))
        plt.show()
            