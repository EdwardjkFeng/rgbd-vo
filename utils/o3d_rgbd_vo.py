""" 
An implementation of RGBD odometry using Open3D library for comparison in the paper
# SPDX-FileCopyrightText: 2021 Binbin Xu
# SPDX-License-Identifier: BSD-3-Clause
"""

import open3d as o3d
from open3d import pipelines as o3d_pl
import numpy as np
import torch
import copy

from geometry import geometry
from utils import visualize
import cv2


class RGBDOdometry:
    def __init__(self, mode="RGBD"):
        self.odo_opt = None
        self.mode = mode
        if mode == "RGBD":
            print("Using RGB-D Odometry")
            self.odo_opt = o3d_pl.odometry.RGBDOdometryJacobianFromColorTerm()
            self.option = o3d_pl.odometry.OdometryOption(depth_min=0.01, depth_max = 5.0, depth_diff_max=0.08)
        elif mode == "ColorICP":  # TODO this part has never been used
            print("Using Hybrid RGB-D Odometry")
            self.odo_opt = o3d_pl.odometry.RGBDOdometryJacobianFromHybridTerm()
            self.option = o3d_pl.odometry.OdometryOption(depth_min=0.01, depth_max = 5.0, depth_diff_max=0.08)
        else:
            raise NotImplementedError()
        
        print(self.option)

    def set_K(self, K, width, height):
        fx, fy, cx, cy = K
        K = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        return K

    def forward(
        self,
        batch_rgb0,
        batch_rgb1,
        batch_dpt0,
        batch_dpt1,
        batch_K,
        batch_objmask0=None,
        batch_objmask1=None,
        vis_pcd=True,
        pose=None,
    ):
        assert batch_rgb0.ndim == 4
        B = batch_rgb0.shape[0]
        batch_R = []
        batch_t = []
        if batch_objmask0 is not None:
            batch_dpt0 = batch_dpt0 * batch_objmask0
        if batch_objmask1 is not None:
            batch_dpt1 = batch_dpt1 * batch_objmask1
        for i in range(B):
            rgb0 = batch_rgb0[i].permute(1, 2, 0).cpu().numpy()
            dpt0 = batch_dpt0[i].permute(1, 2, 0).cpu().numpy()
            rgb1 = batch_rgb1[i].permute(1, 2, 0).cpu().numpy()
            dpt1 = batch_dpt1[i].permute(1, 2, 0).cpu().numpy()
            K = batch_K[i].cpu().numpy().tolist()
            pose10, _ = self.track(rgb0, dpt0, rgb1, dpt1, K)
            batch_R.append(pose10[0])
            batch_t.append(pose10[1])

        batch_R = self._batch_to_tensor(batch_R).type_as(batch_K)
        batch_t = self._batch_to_tensor(batch_t).type_as(batch_K)
        return batch_R, batch_t

    def draw_registration_result(self, source, target, transformation, name="Open3D"):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp], window_name=name)

    def track(self, rgb0, dpt0, rgb1, dpt1, K, vis_pcd=True, odo_init=None):
        H, W, _ = rgb0.shape
        intrinsic = self.set_K(K, H, W)
        rgb0, dpt0, rgb1, dpt1 = self._convert_to_c_style([rgb0, dpt0, rgb1, dpt1])
        rgbd_0 = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb0),
            o3d.geometry.Image(dpt0),
            depth_scale=1,
            depth_trunc=3.0,
        )
        rgbd_1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb1),
            o3d.geometry.Image(dpt1),
            depth_scale=1,
            depth_trunc=3.0,
        )
        if odo_init is None:
            odo_init = np.identity(4)
        if vis_pcd:
            pcd_0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_0, intrinsic)
            pcd_1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_1, intrinsic)
            pcd_0.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pcd_1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


        [is_success, T_10, info] = o3d_pl.odometry.compute_rgbd_odometry(
            rgbd_0, rgbd_1, intrinsic, odo_init, self.odo_opt, self.option
        )

        trs = T_10[0:3, 3]
        # if (trs > 1).sum():  # is_success and vis_pcd:
            # print(T_10)
            # print(is_success)
            # pcd_0 = o3d.geometry.PointCloud.create_from_rgbd_image(
            #     rgbd_0, intrinsic)
            # pcd_0.transform(T_10)
            # o3d.visualization.draw_geometries([pcd_1, pcd_0])
            # self.draw_registration_result(pcd_0, pcd_1, odo_init, "init")
            # self.draw_registration_result(pcd_0, pcd_1, T_10, "aligned")

        trs = T_10[0:3, 3]
        rot = T_10[0:3, 0:3]
        pose10 = [rot, trs]

        # self.compute_residual(rgb0, dpt0, rgb1, dpt1, K, pose10)

        return pose10, is_success

    def _convert_to_c_style(self, ndarray_list):
        return [np.asarray(a, order="C") for a in ndarray_list]

    def _batch_to_tensor(self, ndarray_list):
        tensor_list = [torch.from_numpy(a) for a in ndarray_list]
        return torch.stack(tensor_list)


    def compute_residual(self, rgb0, dpt0, rgb1, dpt1, K, pose):
        rgb0 = torch.from_numpy(rgb0.transpose(2, 0, 1)[None])
        dpt0 = torch.from_numpy(dpt0.transpose(2, 0, 1)[None])
        rgb1 = torch.from_numpy(rgb1.transpose(2, 0, 1)[None])
        dpt1 = torch.from_numpy(dpt1.transpose(2, 0, 1)[None])

        B, C, H, W = rgb0.shape
        K = torch.from_numpy(np.array(K))[None]
        px, py = geometry.generate_xy_grid(B, H, W, K)
        pose = [torch.from_numpy(item)[None] for item in pose]
        u_warped, v_warped, z_warped = geometry.batch_warp_coord(px, py, dpt0, pose, K)

        rgb1_1to0 = geometry.warp_features(
            rgb1, u_warped.type_as(dpt0), v_warped.type_as(dpt0)
        )
        occ = geometry.check_occ(z_warped, dpt0, u_warped.type_as(dpt0), v_warped.type_as(dpt0))
        r_I = torch.mean((rgb1_1to0 - rgb0), dim=1).squeeze()
        r_I[occ.squeeze()] = 0

        dpt1_1to0 = geometry.warp_features(
            dpt1, u_warped.type_as(dpt0), v_warped.type_as(dpt0)
        )
        r_Z = (dpt1_1to0 - z_warped).squeeze()
        r_Z[occ.squeeze()] = 0

        v0 = geometry.compute_vertex(dpt0, px, py)
        v1 = geometry.compute_vertex(dpt1, px, py)
        R, t = pose
        warped_v1 = torch.bmm(R, v1.view(B, 3, H * W)) + t.view(B, 3, 1).expand(B, 3, H * W)
        warped_v1 = warped_v1.view(B, 3, H, W).type_as(dpt0)
        r_v0 = geometry.warp_features(v0.type_as(dpt0), u_warped.type_as(dpt0), v_warped.type_as(dpt0))
        n0 = geometry.compute_normal(v0)
        r_n0 = geometry.warp_features(n0.type_as(dpt0), u_warped.type_as(dpt0), v_warped.type_as(dpt0))
        n1 = geometry.compute_normal(v1)

        # r_p2p = torch.einsum('ijkl, ijkl -> ikl', r_v0 - warped_v1, r_n0).squeeze()
        diff = (r_v0 - warped_v1).view(3, 1, -1).permute(2, 1, 0)
        r_3dep = torch.norm(diff, dim=-1).view(H, W)
        r_3dep[occ.squeeze()] = 0
        r_n0 = r_n0.view(3, -1, 1).permute(1, 0, 2)
        r_p2p = torch.bmm(diff, r_n0).view(H, W)
        r_p2p[occ.squeeze()] = 0


        residual = visualize.create_mosaic(
            [torch.abs(r_I), torch.abs(r_Z), torch.abs(r_p2p), r_3dep],
            # cmap=["NORMAL", "NORMAL", "NORMAL", "NORMAL"], 
            cmap=[cv2.COLORMAP_JET, cv2.COLORMAP_JET, cv2.COLORMAP_JET],
            order='CHW',
            # normalize=True,
        )
        cv2.namedWindow("feature-metric residuals", cv2.WINDOW_NORMAL)
        cv2.imshow("feature-metric residuals", residual)
        cv2.waitKey(10)
