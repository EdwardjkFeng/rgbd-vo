import numpy as np
import torch
import torch.nn as nn

from geometry.geometry import compute_vertex, compute_normal, generate_xy_grid
from dataset.tum_rgbd import TUM

from omegaconf import OmegaConf

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import cv2

# Assume `normals` is a HxWx3 numpy array representing the normal map
# and its values are in the range [-1, 1].

def normal_map_to_vis(normals):
    # Remapping normals from [-1, 1] to [0, 255]
    normals_rgb = (normals + 1) * 0.5 * 255
    normals_rgb = np.clip(normals_rgb, 0, 255).astype(np.uint8)
    
    # Displaying the normal map
    # plt.imshow(normals_rgb)
    # plt.axis('off')
    # plt.show()
    return normals



if __name__ == '__main__':
    class Args():
        pass

    args = Args()
    args.conf = './config/f2f_dia.yaml'

    if args.conf:
        conf = OmegaConf.load(args.conf)

    loader = TUM(conf.data).get_dataset()

    torch_loader = DataLoader(
        loader,
        shuffle = False,
        num_workers = 4,
    )

    for batch in torch_loader:
        color0, color1, depth0, depth1, _, calib = batch['data']

        B, C, H, W = color0.shape
        px, py = generate_xy_grid(B, H, W, calib)
        
        vertex1 = compute_vertex(depth0, px, py)
        vertex2 = compute_vertex(depth1, px, py)

        normal1 = compute_normal(vertex1)
        normal2 = compute_normal(vertex2)

        # torch to numpy 
        normal1 = normal1.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        normal2 = normal2.squeeze().permute(1, 2, 0).cpu().detach().numpy()

        color0 = color0.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.imshow("RGB", cv2.cvtColor(color0, cv2.COLOR_RGB2BGR))

        n1 = normal_map_to_vis(normal1)
        n2 = normal_map_to_vis(normal2)
        cv2.namedWindow("Normal map: 1", cv2.WINDOW_NORMAL)
        cv2.imshow("Normal map: 1", cv2.cvtColor(n1, cv2.COLOR_RGB2BGR))
        cv2.namedWindow("Normal map: 2", cv2.WINDOW_NORMAL)
        cv2.imshow("Normal map: 2", cv2.cvtColor(n2, cv2.COLOR_RGB2BGR))

        cv2.waitKey(10)

    cv2.destroyAllWindows()