""" Segmentation models """
import cv2
import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms.functional as F
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)

from torchvision.utils import draw_segmentation_masks

from utils.visualize import manage_visualization

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class DynamicObjectSegmentation(nn.Module):
    def __init__(
            self,
            model,
            ) -> None:
        pass



class MaskRCNN(nn.Module):
    def __init__(
        self, 
        model = "maskrcnn_resnet50_fpn_v2",
        target_labels = [1],
        prob_threshold = 0.5,
        score_theshold = 0.65,
        vis_mask = True,
    ) -> None:
        super().__init__()

        if model == "maskrcnn_resnet50_fpn_v2":
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.transforms = weights.transforms()
            self.net = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False)
        elif model == "maskrcnn_resnet50_fpn":
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.transforms = weights.transforms()
            self.net = maskrcnn_resnet50_fpn(weights=weights, progress=False)
        else:
            raise NotImplementedError()

        self.net.cuda().eval()

        self.targets = target_labels if len(target_labels) > 0 else None
        self.prob_th = prob_threshold
        self.score_th = score_theshold
        self.vis_mask = vis_mask

    def forward(self, x):
        with torch.no_grad():
            # pred = self.transforms(x)
            concat_mask =[]
            pred = self.net(x)
            for b in range(len(pred)):       
                mask = pred[b]["masks"]
                labels = pred[b]["labels"]
                scores = pred[b]["scores"]

                soi = scores > self.score_th
                if self.targets:
                    loi = torch.zeros_like(labels)
                    for l in self.targets:
                        loi |= labels == l
                    idxoi = (loi & soi).nonzero(as_tuple=False).squeeze()
                else:
                    idxoi = soi.nonzero(as_tuple=False).squeeze()
                
                binary_masks = (mask > self.prob_th).squeeze(1)[idxoi]
                if binary_masks.ndim < 3:
                    binary_masks = binary_masks[None]
                concat_mask.append(torch.any(binary_masks, dim=0, keepdim=True))

                if self.vis_mask and b == 0:
                    rgb = (x.squeeze().cpu() * 255).to(torch.uint8)
                    masked_img = draw_segmentation_masks(
                        rgb, 
                        binary_masks.cpu(), 
                        alpha=0.9,
                    )
                    masked_img = F.to_pil_image(masked_img)
                    masked_img = np.asarray(masked_img)[:, :, [2, 1, 0]]

                    cv2.namedWindow("Segmented Moving objects", cv2.WINDOW_NORMAL)
                    cv2.imshow("Segmented Moving objects", masked_img)
                    manage_visualization()

            concat_mask = torch.stack(concat_mask, dim=0)
        
        return concat_mask