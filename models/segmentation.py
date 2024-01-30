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


from kornia.contrib.visual_prompter import VisualPrompter
from kornia.contrib.models.sam import SamConfig
from kornia.contrib.models import SegmentationResults
from kornia.geometry.keypoints import Keypoints
from kornia.geometry.boxes import Boxes
from kornia.utils import get_cuda_device_if_available, tensor_to_image

from kornia.morphology import dilation

from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

from utils.visualize import manage_visualization, create_mosaic

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class SamPrompter(nn.Module):

    def __init__(self, model = "mobile_sam", pretrain = True) -> None:
        super().__init__()

        device = get_cuda_device_if_available()

        samconfig = SamConfig(model_type=model, pretrained=pretrain)
        self.prompter = VisualPrompter(samconfig, device=device)

    # def forward(self, x, keypoints, labels):
    #     self.prompter.set_image(x.squeeze())
    #     keypoints = Keypoints(keypoints)
    #     labels.to(torch.float32)

    #     prediction = self.prompter.predict(keypoints, labels, multimask_output=False)

    #     # visualize mask
    #     rgb = (x.squeeze().cpu() * 255).to(torch.uint8)
    #     masked_img = draw_segmentation_masks(
    #         rgb, 
    #         prediction.binary_masks.squeeze().cpu(), 
    #         alpha=0.9,
    #     )
    #     masked_img = F.to_pil_image(masked_img)
    #     masked_img = np.asarray(masked_img)[:, :, [2, 1, 0]]

    #     cv2.namedWindow("Segmented Moving objects", cv2.WINDOW_NORMAL)
    #     cv2.imshow("Segmented Moving objects", masked_img)
    #     manage_visualization()

    #     return prediction.binary_masks
    def forward(self, x, boxes):
        self.prompter.set_image(x.squeeze())
        boxes = Boxes.from_tensor(boxes)

        prediction = self.prompter.predict(boxes=boxes, multimask_output=False)

        # visualize mask
        rgb = (x.squeeze().cpu() * 255).to(torch.uint8)
        masked_img = draw_segmentation_masks(
            rgb, 
            prediction.binary_masks.squeeze().cpu(), 
            alpha=0.9,
        )
        masked_img = F.to_pil_image(masked_img)
        masked_img = np.asarray(masked_img)[:, :, [2, 1, 0]]

        cv2.namedWindow("Segmented Moving objects", cv2.WINDOW_NORMAL)
        cv2.imshow("Segmented Moving objects", masked_img)
        manage_visualization()

        return prediction.binary_masks
    # def forward(self, x, masks):
    #     self.prompter.set_image(x.squeeze())

    #     prediction = self.prompter.predict(masks=masks, multimask_output=False)

    #     # visualize mask
    #     rgb = (x.squeeze().cpu() * 255).to(torch.uint8)
    #     masked_img = draw_segmentation_masks(
    #         rgb, 
    #         prediction.binary_masks.squeeze().cpu(), 
    #         alpha=0.9,
    #     )
    #     masked_img = F.to_pil_image(masked_img)
    #     masked_img = np.asarray(masked_img)[:, :, [2, 1, 0]]

    #     cv2.namedWindow("Segmented Moving objects", cv2.WINDOW_NORMAL)
    #     cv2.imshow("Segmented Moving objects", masked_img)
    #     manage_visualization()

    #     print(prediction.logits)
    #     return prediction.binary_masks



# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class MaskRCNN(nn.Module):
    def __init__(
        self, 
        model = "maskrcnn_resnet50_fpn_v2",
        target_labels = [],
        prob_threshold = 0.5,
        score_theshold = 0.65,
        vis_mask = True,
        # dilation = True,
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
        self.dilation = dilation

    def forward(self, x):
        with torch.no_grad():
            # pred = self.transforms(x)
            concat_masks = []
            concat_boxes = [] 
            pred = self.net(x)
            for b in range(len(pred)):
                mask = pred[b]["masks"]
                labels = pred[b]["labels"]
                scores = pred[b]["scores"]
                boxes = pred[b]["boxes"]

                soi = scores > self.score_th
                if self.targets:
                    loi = torch.zeros_like(labels)
                    for l in self.targets:
                        loi |= labels == l
                    idxoi = (loi & soi).nonzero(as_tuple=False).squeeze()
                else:
                    idxoi = soi.nonzero(as_tuple=False).squeeze()
                
                # Maintain only mask for instance of interesting classes
                binary_masks = (mask > self.prob_th)[idxoi]

                if binary_masks.ndim <= 3:
                    binary_masks = binary_masks[None]
                print(binary_masks.shape)
                # concat_masks.append(torch.any(binary_masks, dim=0, keepdim=True))

                # if self.dilation:
                #     kernel = torch.ones(3, 3).cuda()
                #     binary_masks = dilation(binary_masks, kernel)
                concat_masks.append(binary_masks)
                
                if idxoi.nelement() != 0:
                    boxes = boxes[idxoi]
                    concat_boxes.append(
                        boxes[None] if boxes.ndim < 2 else boxes
                    )

            concat_masks = torch.stack(concat_masks, dim=0)
            print(concat_masks.shape)
            # concat_masks = torch.where(torch.sum(concat_masks, dim=0)>0, 1, 0)
            
            if len(concat_boxes) != 0:
                concat_boxes = torch.stack(concat_boxes, dim=0)
            else:
                concat_boxes = None

            if self.vis_mask and (b == 0) and (concat_boxes is not None):
                rgb = (x.squeeze().cpu() * 255).to(torch.uint8)
                masked_img = draw_segmentation_masks(
                    rgb, 
                    concat_masks.to(bool).squeeze().cpu(), 
                    alpha=0.6,
                )
                boxed_img = draw_bounding_boxes(
                    masked_img,
                    concat_boxes.squeeze(0).cpu(),
                    width=4,
                )
                masked_img = F.to_pil_image(masked_img)
                masked_img = np.asarray(masked_img)#[:, :, [2, 1, 0]]

                # boxed_img = draw_bounding_boxes(
                #     rgb,
                #     concat_boxes.squeeze(0).cpu(),
                #     width=1,
                # )
                boxed_img = F.to_pil_image(boxed_img)
                boxed_img = np.asarray(boxed_img)#[:, :, [2, 1, 0]]

                # segm = create_mosaic([boxed_img, masked_img], cmap="NORMAL", order="HWC")
                segm = create_mosaic([boxed_img], cmap="NORMAL", order="HWC")

                cv2.namedWindow("MaskRCNN Segmentation", cv2.WINDOW_NORMAL)
                cv2.imshow("MaskRCNN Segmentation", segm)
                manage_visualization()
    
        return concat_masks, concat_boxes