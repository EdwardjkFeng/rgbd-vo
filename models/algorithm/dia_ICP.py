"""Implementation of Combined Direct Image Alignment"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import cv2

import geometry.geometry as geometry
import geometry.projective_ops as proj_ops
from models.algorithms import feature_gradient, LM_H

from utils.timers import *
from lietorch import SE3


class DirectImageAlign(nn.Module):
    def __init__(
        self,
        combined_geom=True,
        geom_weight=0.01,
        num_iterations=5,
        mEst_func=None,
        solver_func=None,
        timers=None, 
    ) -> None:
        super().__init__()

        self.combined_geom = combined_geom
        self.lambda_ = geom_weight
        self.num_iterations = num_iterations

        self.mEstimator = mEst_func
        self.directSolver = solver_func

        self.timers = timers or NullTimer()

    def forward(self, pose, I0, I1, invD0, invD1, intrinsics, depth0=None, depth1=None):
        B, C, H, W = I0.shape
        px, py = geometry.generate_xy_grid(B, H, W,  intrinsics)

        for itr in range(self.num_iterations):
            self.timers.tic('compute warping residuals and Jacobian')
            x0, valid, Jp, Jt = proj_ops.Batch_projective_transform(
                pose, depth1, intrinsics, jacobian=True, return_depth=True
            )

            Jw = torch.matmul(Jp, Jt)

            inbound_image = (
                (x0[:, 0] > 0) & (x0[:, 0] < W - 1) & (x0[:, 1] > 0) & (x0[:, 1] < H - 1)
            )
            valid = inbound_image & valid

            valid_depth = (depth0 > 0) & (depth1 > 0)
            valid = valid[:, None] & valid_depth

            u, v, d = x0.unbind(dim=1)
            u /= W - 1
            v /= H - 1
            x0_grid = torch.stack([u, v], dim=-1)
            x0_grid = ((x0_grid - 0.5) * 2).clamp(-2, 2)

            I0_warped = f.grid_sample(
                I0, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            r_I = I1 - I0_warped # [B, C, H, W]

            # invD0_warped = f.grid_sample(
            #     invD0, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True
            # )
            # invD0_warped = f.grid_sample(
            #     depth0, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True
            # )
            # r_Z = d[:, None] - invD0_warped # [B, 1, H, W]
            # print(f"photo.: {r_I.max().data}, geo.: {r_Z.max().data}")
            
            # grad_I0 = torch.stack(feature_gradient(I0, normalize_gradient=True), dim=-1)
            dI0_dx, dI0_dy = feature_gradient(I0)
            dI0_dx = f.grid_sample(dI0_dx, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            dI0_dy = f.grid_sample(dI0_dy, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            grad_I0 = torch.stack((dI0_dx, dI0_dy), dim=-1)
            J_I = torch.einsum('ijklmn,iklnx->ijklmx', grad_I0.unsqueeze(-2), Jw) # [B, C, H, W, 1, 6]
            # print(I0[:, :, 5, 5])
            # print(1/ invD0[:, :, 5, 5])
            # print(Jw.shape)
            # print(Jw[:, 5, 5])
            # print(grad_I0[:, :, 5, 5])
            # print(J_I[:, :, 5, 5])

            # grad_invD0 = torch.stack(feature_gradient(invD0_warped, normalize_gradient=True), dim=-1)
            # dD0_dx, dD0_dy = feature_gradient(invD0)
            # dD0_dx, dD0_dy = feature_gradient(depth0)
            # dD0_dx = f.grid_sample(dD0_dx, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            # dD0_dy = f.grid_sample(dD0_dy, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            # grad_invD0 = torch.stack((dD0_dx, dD0_dy), dim=-1)
            # J_Z = torch.einsum('ijklmn,iklnx->ijklmx', grad_invD0.unsqueeze(-2), Jw) - Jt[..., [2], :][:, None] # [B, 1, H, W, 1, 6]

            # point to plane icp
            vertex0 = geometry.compute_vertex(depth0, px, py)
            vertex1 = geometry.compute_vertex(depth1, px, py)
            normal0 = geometry.compute_normal(vertex0)
            normal1 = geometry.compute_normal(vertex1)
            r_Z, J_Z, icp_occ = self.compute_ICP_residuals_jacobian(
                vertex0,
                vertex1,
                normal0,
                normal1,
                pose,
                intrinsics,
                obj_mask0=None,
                obj_mask1=None,
            )



            valid = valid.expand(-1, C+1, -1, -1).reshape(B, -1)
            J_I = J_I.reshape(B, -1, 6)
            J_Z = J_Z.reshape(B, -1, 6)
            J = torch.cat((J_I, self.lambda_ * J_Z), dim=1) * valid[..., None] # [B, HXWX(C+1), 6]

            r_I = r_I.view(B, -1)
            r_Z = -r_Z.view(B, -1)
            r = torch.cat((r_I, self.lambda_ * r_Z), dim=1) * valid # [B, HXWX(C+1)]
            self.timers.toc('compute warping residuals and Jacobian')

            self.timers.tic("Robust estimator")
            # weights_I = self.mEstimator(r_I)
            # weights_Z = self.mEstimator(r_Z)
            # weights = torch.cat((weights_I, weights_Z), dim=1)
            weights = self.mEstimator(r)
            self.timers.toc("Robust estimator")

            self.timers.tic('solve x=A^{-1}b')
            WJ = weights[..., None] * J
            JtJ =  torch.bmm(J.permute(0, 2, 1), WJ)
            Hessian = LM_H(JtJ)
            # Hessian = JtJ
            Wr = weights * r
            Rhs = torch.bmm(J.permute(0, 2, 1), Wr[..., None])
            
            pose = forward_update_pose(Hessian, Rhs, pose)
            self.timers.toc('solve x=A^{-1}b')
        
        return pose

    def compute_ICP_residuals_jacobian(self, vertex0, vertex1, normal0, normal1, pose10, K, obj_mask0=None, obj_mask1=None):
        transform = pose10.matrix()
        R, t = transform[:, :3, :3], transform[:, :3, 3]
        B, C, H, W = vertex0.shape

        rot_vertex0_to1 = torch.bmm(R, vertex1.view(B, 3, H*W))
        vertex0_to1 = rot_vertex0_to1 + t.view(B, 3, 1).expand(B, 3, H*W)
        # normal0_to1 = torch.bmm(R, normal0.view(B, 3, H * W))

        fx, fy, cx, cy = torch.split(K, 1, dim=1)
        x_, y_, s_ = torch.split(vertex0_to1, 1, dim=1)
        u_ = (x_ / s_).view(B, -1) * fx + cx
        v_ = (y_ / s_).view(B, -1) * fy + cy

        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)

        # # interpolation-version
        r_vertex1 = geometry.warp_features(vertex0, u_, v_)
        r_normal1 = geometry.warp_features(normal0, u_, v_)

        diff = vertex0_to1 - r_vertex1.view(B, 3, H * W)
        # normal_diff = (normal0_to1 * r_normal1.view(B, 3, H * W)).sum(dim=1, keepdim=True)

        # occlusion
        occ = ~inviews.view(B,1,H,W) | (diff.view(B,3,H,W).norm(p=2, dim=1, keepdim=True) > 0.1) #| \
        if obj_mask0 is not None:
            bg_mask0 = ~obj_mask0
            occ = occ | (bg_mask0.view(B, 1, H, W))
        if obj_mask1 is not None:
            obj_mask1_r = geometry.warp_features(obj_mask1.float(), u_, v_) > 0
            bg_mask1 = ~obj_mask1_r
            occ = occ | (bg_mask1.view(B, 1, H, W))

        # point-to-plane residuals
        res = (r_normal1.view(B, 3, H*W)) * diff
        res = res.sum(dim=1, keepdim=True).view(B,1,H,W)  # [B,1,H,W]
        # inverse point-to-plane jacobians
        NtC10 = torch.bmm(r_normal1.view(B,3,-1).permute(0,2,1), R)  # [B, H*W, 3]
        J_rot = torch.bmm(NtC10.view(-1,3).unsqueeze(dim=1),  #[B*H*W,1,3]
                           geometry.batch_skew(vertex1.view(B,3,-1).permute(0, 2, 1).contiguous().view(-1, 3))).squeeze()  # [B*H*W, 3]
        J_trs = -NtC10.view(-1,3)  # [B*H*W, 3]

        # compose jacobians
        J_F_p = torch.cat((J_rot, J_trs), dim=-1)  # follow the order of [rot, trs]  [B*H*W, 6]
        J_F_p = J_F_p.view(B, 1, -1, 6)  # [B, 1, HXW, 6]

        # covariance-normalized
        dpt0 = vertex1[:,2:3,:,:]
        sigma_icp = self.compute_icp_sigma(dpt_l=dpt0, normal_r=r_normal1, rot=R)
        res = res / (sigma_icp + 1e-8)
        J_F_p = J_F_p / (sigma_icp.view(B,1,H*W,1) + 1e-8)

        # follow the conversion of inversing the jacobian
        J_F_p = - J_F_p

        res[occ] = 1e-6

        return res, J_F_p, occ
    

    def compute_icp_sigma(self, dpt_l, normal_r, rot, dataset='TUM'):
        # obtain sigma
        if dataset == 'TUM':
            sigma_disp = 0.4
            sigma_xy = 5.5
            baseline = 1.2
            focal = 525.0
        else:
            raise NotImplementedError()

        B, C, H, W = normal_r.shape

        # compute sigma on depth using stereo model
        sigma_depth = torch.empty((B, 3, H, W)).type_as(dpt_l)
        sigma_depth[:, 0:2, :, :] = dpt_l / focal * sigma_xy
        sigma_depth[:, 2:3, :, :] = dpt_l * dpt_l * sigma_disp / (focal * baseline)

        J = torch.bmm(normal_r.view(B,3,H*W).transpose(1,2), rot)
        J = J.transpose(1,2).view(B,3,H,W)
        cov_icp = (J * sigma_depth * sigma_depth * J).sum(dim=1, keepdim=True)

        sigma_icp = torch.sqrt(cov_icp + 1e-8)
        return sigma_icp


def forward_update_pose(H, Rhs, pose: SE3):
    """Ues left-multiplication for the pose update in the forward compositional form. Here ksi_k o (delta_ksi)

    Args:
        H: Hessian
        Rhs: Right-hand side vector
        pose: the initial pose (forward transform inverse of xi)

    Returns:
        forward updated pose (inverse of xi)
    """
    xi = least_square_solve(H, Rhs)
    # forward compotional for SE3: delta_ksi
    pose = pose.retr(xi.squeeze(-1))
    return pose


def least_square_solve(H, Rhs):
    """x =  - H ^{-1} * Rhs

    Args:
        H: Hessian [B, 6, 6]
        Rhs: Right-hand side vector [B, 6, 1]

    Returns:
        increment (kxi) [B, 6, 1]
    """
    H_, Rhs_ = H.cpu(), Rhs.cpu()
    try:
        U = torch.linalg.cholesky(H_, upper=True)
        xi = torch.cholesky_solve(Rhs_, U, upper=True).to(H)
    except:
        inv_H = torch.inverse(H)
        xi = torch.bmm(inv_H, Rhs)
    #     # because the jacobian is also including the minus signal, it should be (J^T * J) J^T * r
    #     # xi = - xi
    # inv_H = invH(H)
    # xi = torch.bmm(inv_H, Rhs)
    return xi
    

if __name__ == "__main__":
    import numpy as np
    from omegaconf import OmegaConf
    from dataset.cofusion import CoFusion
    from torch.utils.data import DataLoader
    from utils.run_utils import check_cuda
    from scipy.spatial.transform import Rotation
    
    class Args():
        pass

    args = Args()
    args.conf = './config/default.yaml'

    if args.conf:
        conf = OmegaConf.load(args.conf)

    # loader = TUM(conf.data).get_dataset()
    loader = [CoFusion(conf.data).get_dataset()[i] for i in range(1)]

    torch_loader = DataLoader(
        loader,
        batch_size=1,
        shuffle = False,
        num_workers = 4,
    )

    dia = DirectImageAlign(False, 0.01, timers=NullTimer())
    with torch.no_grad():
        for batch in torch_loader:
            color0, color1, depth0, depth1, pose, calib = batch['data']
            print(pose.shape, depth1.shape)

            quat = Rotation.from_matrix(pose[:, :3, :3]).as_quat()
            trans = pose[:, :3, 3]
            pose_data = np.concatenate((trans, quat), axis=-1)
            T = SE3.InitFromVec(torch.tensor(pose_data).float())
            
            dia(T, color0, color1, depth0, depth1, calib)
            