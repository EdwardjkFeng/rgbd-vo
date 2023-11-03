from typing import Dict, Tuple

from omegaconf import OmegaConf
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.contrib import connected_components, EdgeDetector
from kornia.morphology import closing, opening

from geometry.geometry import compute_vertex, compute_normal, generate_xy_grid
from dataset.tum_rgbd import TUM
from torch.utils.data import DataLoader
from utils.visualize import create_mosaic
from geometry.geometry import coord_grid
from experiment.run_utils import check_cuda

import matplotlib.pyplot as plt

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

def compute_edginess_map(vertex, normal, lambda_weight, tau):
    vertex_p = F.pad(vertex, pad=(1, 1, 1, 1), mode='replicate')
    normal_p = F.pad(normal, pad=(1, 1, 1, 1), mode='replicate')

    neighborhood = torch.tensor([[[-1, -1], [-1, 0], [-1, 1]],
                                 [[0, -1], [0, 0], [0, 1]],
                                 [[1, -1], [1, 0], [1, 1]]], dtype=torch.int64)
    
    neighborhood = neighborhood.view(1, 3, 3, 2).to(vertex.device)

    B, C, H, W = vertex.shape
    x, y = coord_grid(H, W, device=vertex.device)
    y = y.reshape(-1, 1, 1) + neighborhood[..., 0] + 1
    x = x.reshape(-1, 1, 1) + neighborhood[..., 1] + 1
    neighbors_v = vertex_p[..., y.view(-1, 9).long(), x.view(-1, 9).long()]
    neighbors_n = normal_p[..., y.view(-1, 9).long(), x.view(-1, 9).long()]

    v_center = vertex.view(B, C, -1)
    n_center = vertex.view(B, C, -1)

    diff_v = v_center[..., None] - neighbors_v
    print(diff_v.shape, n_center.shape)
    dist = torch.einsum('ijkl, ijkm -> ikl', diff_v, n_center[..., None])
    print(dist.shape)
    phi_d, _ = torch.abs(dist).max(dim=-1, keepdim=True)
    print(phi_d.shape)

    phi_c = torch.zeros_like(dist)
    mask = dist > 0

    phi_c[mask] = 1 - torch.einsum(
        'ijkl, ijkm -> ikl', 
        neighbors_n, 
        n_center[..., None]
    )[mask]
    print(phi_c.shape)
    phi_c, _ = phi_c.max(dim=-1, keepdim=True)

    # edginess_map = torch.where(torch.max(phi_c, phi_d) > 1.0, 0.0, 1.0)
    edginess_map = torch.where(phi_d + lambda_weight * phi_c > tau, 0.0, 1.0)
    edginess_map = edginess_map.view(B, H, W)


    # height, width = vertex.shape[:-1]
    # edginess_map = np.zeros((height, width), dtype=np.float32)

    # Define neighborhood N
    # This is a 3x3 neighborhood for simplicity; you may want to use a larger neighborhood
    # N = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # for y in range(1, height-2):
    #     for x in range(1, width-2):
    #         v = vertex[y, x]
    #         n = normals[y, x]
    #         phi_d = 0
    #         phi_c = 0
    #         for dy, dx in N:
    #             i = y + dy
    #             j = x + dx
    #             v_i = vertex[i, j]
    #             n_i = normals[i, j]
    #             dist = (v_i - v).dot(n)
    #             phi_d = max(phi_d, abs(dist))
    #             if dist < 0:
    #                 phi_c = max(phi_c, 0)
    #             else:
    #                 phi_c = max(phi_c, 1 - np.dot(n_i, n))

    #         # Check if the pixel is an edge pixel
    #         # if phi_d + lambda_weight * phi_c > tau:
    #         edginess_map[y, x] = 1.0 if max(phi_c, phi_d) > 1.0 else 0.

    return edginess_map

def segment_by_connected_components(edginess_map):
    # Use OpenCV's connectedComponents to segment the image
    num_labels, labels_im = cv2.connectedComponents(edginess_map.astype(np.uint8))
    return num_labels, labels_im


def normalize_feature(f):
    f_min = f.min()
    f_max = f.max()
    f_n = f - f_min
    return f_n / (f_max - f_min)

def create_random_labels_map(classes: int) -> dict[int, tuple[int, int, int]]:
    labels_map: Dict[int, Tuple[int, int, int]] = {}
    for i in classes:
        labels_map[i] = torch.randint(0, 255, (3,))
    labels_map[0] = torch.zeros(3)
    return labels_map


def labels_to_image(img_labels: torch.Tensor, labels_map: Dict[int, Tuple[int, int, int]]) -> torch.Tensor:
    """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
    representation for visualisation purposes."""
    assert len(img_labels.shape) == 2, img_labels.shape
    H, W = img_labels.shape
    out = torch.empty(3, H, W, dtype=torch.uint8)
    for label_id, label_val in labels_map.items():
        mask = img_labels == label_id
        for i in range(3):
            out[i].masked_fill_(mask, label_val[i])
    return out


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
        color0, color1, depth0, depth1, _, calib = check_cuda(batch['data'])

        B, C, H, W = color0.shape
        px, py = generate_xy_grid(B, H, W, calib)
        
        vertex1 = compute_vertex(depth0, px, py)
        vertex2 = compute_vertex(depth1, px, py)

        normal1 = compute_normal(vertex1)
        normal2 = compute_normal(vertex2)

        ##############################
        # torch to numpy 
        lambda_weight = 1
        tau = 0.5
        edginess_map = compute_edginess_map(vertex1, normal1, lambda_weight, tau)
        # edginess_map = edginess_map.squeeze().cpu().numpy()

        # kernel = np.ones((3, 3),np.uint8)
        # dilated_edge_map = cv2.dilate(edginess_map, kernel, iterations = 1)
        # dilated_edge_map = np.where(dilated_edge_map>0, 0., 1.)


        edge_detector = EdgeDetector().cuda()
        edges = edge_detector(color0 * 255)
        # edges = edges.squeeze().cpu().detach().numpy()
        edges = normalize_feature(edges)
        edges = torch.where(edges > 0.85, 0.0, 1.0)

        # Segment the image
        # edginess_map = edginess_map.squeeze().cpu().numpy()
        # num_labels, labels_im = segment_by_connected_components(edginess_map)
        labels_im = connected_components(edges)
        labels_im = labels_im.squeeze().cpu()
        # Covert labels to labeled image
        color_ids = torch.unique(labels_im)
        labels_map = create_random_labels_map(color_ids)
        labels_im = labels_to_image(labels_im, labels_map).permute(1, 2, 0).squeeze().numpy()
        edges = edges.squeeze().cpu().numpy()
        
        # print(labels_im.shape)

        # Visualize the labels
        ################################

        # torch to numpy 
        normal1 = normal1.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        normal2 = normal2.squeeze().permute(1, 2, 0).cpu().detach().numpy()

        color0 = color0.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.imshow("RGB", cv2.cvtColor(color0, cv2.COLOR_RGB2BGR))

        # Visualize normal and depth map
        n1 = normal_map_to_vis(normal1)
        n2 = normal_map_to_vis(normal2)

        depth0 = depth0.squeeze().cpu().numpy()
        depth1 = depth1.squeeze().cpu().numpy()

        vert_norm1 = create_mosaic([n1, depth0, edges, labels_im], cmap=['NORMAL', 'NORMAL', 'NORMAL', cv2.COLORMAP_JET], order='HWC', normalize=True)
        vert_norm2 = create_mosaic([n2, depth1], cmap=['NORMAL', 'NORMAL'], order='HWC', normalize=True)
        
        cv2.namedWindow("Normal and depth map: 1", cv2.WINDOW_NORMAL)
        cv2.imshow("Normal and depth map: 1", vert_norm1)
        cv2.namedWindow("Normal and depth map: 2", cv2.WINDOW_NORMAL)
        cv2.imshow("Normal and depth map: 2", vert_norm2)
        cv2.waitKey(10)

    cv2.destroyAllWindows()