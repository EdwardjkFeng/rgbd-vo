from typing import Dict, Tuple

from omegaconf import OmegaConf
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.contrib import connected_components, EdgeDetector
from kornia.morphology import closing, opening, dilation, erosion

from geometry.geometry import compute_vertex, compute_normal, generate_xy_grid
from dataset.tum_rgbd import TUM
from dataset.bonn_rgbd import Bonn
from dataset.dataloader import load_data
from torch.utils.data import DataLoader
from utils.visualize import create_mosaic, manage_visualization, image_to_display
from geometry.geometry import coord_grid
from models.algorithms import feature_gradient
from utils.run_utils import check_cuda

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

def compute_edginess_map(vertex, normal, lambda_weight, tau, prior_edges=None):
    device = vertex.device
    n = 3 # n has to be odd number and >= 3
    p = n//2
    vertex_p = F.pad(vertex, pad=(p, p, p, p), mode='replicate')
    normal_p = F.pad(normal, pad=(p, p, p, p), mode='replicate')

    neighborhood = torch.stack(torch.meshgrid(
        torch.arange(-p, p+1).to(device).int(),
        torch.arange(-p, p+1).to(device).int(),
        indexing="ij"
    ), dim=-1)
    
    neighborhood = neighborhood.view(1, n, n, 2).to(device)

    B, C, H, W = vertex.shape
    x, y = coord_grid(H, W, device=device)
    y = y.reshape(-1, 1, 1) + neighborhood[..., 0] + p
    x = x.reshape(-1, 1, 1) + neighborhood[..., 1] + p
    neighbors_v = vertex_p[..., y.view(-1, n*n).long(), x.view(-1, n*n).long()]
    neighbors_n = normal_p[..., y.view(-1, n*n).long(), x.view(-1, n*n).long()]

    v_center = vertex.view(B, C, -1)
    n_center = vertex.view(B, C, -1)

    diff_v = v_center[..., None] - neighbors_v
    print(diff_v.shape, n_center.shape)
    dist = torch.einsum('ijkl, ijkm -> ikl', diff_v, n_center[..., None])
    print(dist.shape)
    phi_d, _ = torch.abs(dist).max(dim=-1, keepdim=True)
    phi_d = normalize_feature(phi_d)
    print(phi_d.shape)

    phi_c = torch.zeros_like(dist)
    mask = dist >= 0

    phi_c[mask] = 1 - torch.einsum(
        'ijkl, ijkm -> ikl', 
        neighbors_n, 
        n_center[..., None]
    )[mask]
    phi_c, _ = phi_c.max(dim=-1, keepdim=True)
    print(phi_c.shape)
    # phi_c = normalize_feature(phi_c)

    # vis_disc_conv = create_mosaic(
    #     [phi_d.view(H, W), phi_c.view(H, W)],
    #     cmap=["NORMAL", "NORMAL"],
    #     order='CHW',
    # )
    vis_disc_conv = [phi_d.view(H, W), phi_c.view(H, W)]

    # edginess_map = torch.where(torch.max(phi_c, phi_d) > 1.0, 0.0, 1.0)
    phi_c, phi_d = [t.view(B, H, W) for t in [phi_c, phi_d]]
    if prior_edges is not None:
        # print(phi_d.mean(), phi_c.mean(), prior_edges.mean())
        print(phi_d.max(), phi_c.max(), prior_edges.max())
        edginess_map = torch.where(
            phi_d + lambda_weight * phi_c + 6 * prior_edges > tau, 0.0, 1.0
        )
    else: 
        edginess_map = torch.where(
            phi_d + lambda_weight * phi_c > tau, 0.0, 1.0
        )

    return edginess_map, vis_disc_conv


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


def compute_edge_sobel(img: torch.Tensor):
    dx, dy = feature_gradient(img)
    edges = torch.cat((dx, dy), dim=1)
    print(edges.shape)
    edges = edges.norm(dim=1)
    max_magnitude = edges.max()
    min_magnitude = edges.min()
    edges = (edges - min_magnitude) / (max_magnitude - min_magnitude)

    # zeros = torch.zeros_like(edges)
    # ones = torch.ones_like(edges)
    # edges = torch.where(edges > 0.1, ones, zeros)

    return edges


if __name__ == '__main__':
    class Args():
        pass

    args = Args()
    args.conf = './config/default.yaml'

    if args.conf:
        conf = OmegaConf.load(args.conf)

    # # loader = TUM(conf.data).get_dataset()
    # loader = Bonn(conf.data).get_dataset()
    if conf.data.name == "TUM_RGBD":
        # sequence = "rgbd_dataset_freiburg1_desk"
        # sequence = "rgbd_dataset_freiburg1_xyz"
        sequence = "rgbd_dataset_freiburg3_walking_static"
        conf["select_traj"] = sequence
    elif conf.data.name == "Bonn_RGBD":
        # sequence = "rgbd_bonn_balloon_tracking"
        sequence = "rgbd_bonn_person_tracking"
        # sequence = "rgbd_bonn_balloon_tracking2"
        # sequence = "rgbd_bonn_placing_nonobstructing_box"
        # sequence = "rgbd_bonn_static"
        conf.data.select_traj = sequence
    elif conf.data.name == "CoFusion":
        sequence = "car4-full"
        conf["select_traj"] = sequence

    loader = load_data(conf.data.name, conf=conf.data)
    torch_loader = DataLoader(
        loader,
        shuffle = False,
        num_workers = 4,
    )

    with torch.no_grad():
        for batch in torch_loader:
            color0, color1, depth0, depth1, _, calib = check_cuda(batch['data'])

            B, C, H, W = color0.shape
            px, py = generate_xy_grid(B, H, W, calib)
            
            vertex1 = compute_vertex(depth0, px, py)
            vertex2 = compute_vertex(depth1, px, py)

            normal1 = compute_normal(vertex1)
            normal2 = compute_normal(vertex2)

            # Sobel edges
            edges3 = compute_edge_sobel(color0)

            cv2.namedWindow("Sobel", cv2.WINDOW_NORMAL)
            cv2.imshow("Sobel", create_mosaic([edges3], cmap="NORMAL", normalize=True))
            

            # Discontinuity and concavity
            lambda_weight = 0.8
            tau = 0.5
            edginess_map1, vis_disc_conv = compute_edginess_map(
                vertex1, normal1, lambda_weight, tau, edges3,
            )
            edges1 = edginess_map1.squeeze().cpu().numpy()
            # kernel = torch.ones((3, 3)).cuda()
            # edginess_map = erosion(edginess_map[None], kernel)

            labels_im1 = connected_components(edginess_map1, num_iterations=1000)
            labels_im1 = labels_im1.squeeze().cpu()
            # Covert labels to labeled image
            color_ids1 = torch.unique(labels_im1)
            print(color_ids1.shape)
            labels_map1 = create_random_labels_map(color_ids1)
            labels_im1 = labels_to_image(labels_im1, labels_map1).permute(1, 2, 0).squeeze().numpy()


            # Edge detector
            edge_detector = EdgeDetector().cuda()
            edges = edge_detector(color0 * 255)
            # edges = edges.squeeze().cpu().detach().numpy()
            edges = normalize_feature(edges)
            edginess_map2 = torch.where(edges > 0.8, 0.0, 1.0)
            # kernel = torch.ones((3, 3)).cuda()
            # edginess_map2 = dilation(edginess_map2, kernel)

            # Segment the image
            labels_im2 = connected_components(edginess_map2, num_iterations=1000)
            labels_im2 = labels_im2.squeeze().cpu()
            # Covert labels to labeled image
            color_ids2 = torch.unique(labels_im2)
            labels_map2 = create_random_labels_map(color_ids2)
            labels_im2 = labels_to_image(labels_im2, labels_map2).permute(1, 2, 0).squeeze().numpy()
            # edges = edges.squeeze().cpu().numpy()
            edges2 = edginess_map2.squeeze().cpu().numpy()
            
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

            vert_norm = create_mosaic(
                [n1, depth0, n2, depth1],
                cmap=["NORMAL"],
                normalize=True,
            )

            edges3 = edges3.squeeze().cpu().numpy()

            # cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
            # cv2.imshow("Depth", create_mosaic([depth0], cmap="NORMAL", normalize=True))

            # cv2.namedWindow("Normal", cv2.WINDOW_NORMAL)
            # cv2.imshow("Normal", create_mosaic([n1], cmap="NORMAL", normalize=True))

            # cv2.namedWindow("Discontinuity", cv2.WINDOW_NORMAL)
            # cv2.imshow("Discontinuity", create_mosaic([vis_disc_conv[0]], cmap="NORMAL", normalize=True))

            # cv2.namedWindow("Concavity", cv2.WINDOW_NORMAL)
            # cv2.imshow("Concavity", create_mosaic([vis_disc_conv[1]], cmap="NORMAL", normalize=True))

            cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
            cv2.imshow("Edges", create_mosaic([edges1], cmap="NORMAL", normalize=True))

            cv2.namedWindow("Labels", cv2.WINDOW_NORMAL)
            cv2.imshow("Labels", create_mosaic([labels_im1], cmap=cv2.COLORMAP_JET, normalize=True))

            vert_norm1 = create_mosaic(
                [n1, edges3, depth0, vis_disc_conv[0], vis_disc_conv[1],  edges1, labels_im1, edges2, labels_im2], 
                cmap=['NORMAL', 'NORMAL', 'NORMAL', 'NORMAL', 'NORMAL', 'NORMAL', cv2.COLORMAP_JET, 'NORMAL', cv2.COLORMAP_JET], order='HWC', 
                normalize=True,
            )
            # vert_norm2 = create_mosaic([n2, depth1], cmap=['NORMAL', 'NORMAL'], order='HWC', normalize=True)
            
            # cv2.namedWindow("Normal and depth map: 1", cv2.WINDOW_NORMAL)
            # cv2.imshow("Normal and depth map: 1", vert_norm1)
            # cv2.namedWindow("Normal and depth map: 2", cv2.WINDOW_NORMAL)
            # cv2.imshow("Normal and depth map: 2", vert_norm2)
            manage_visualization()

        cv2.destroyAllWindows()