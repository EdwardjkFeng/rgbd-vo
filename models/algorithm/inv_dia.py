"""Implementation of Combined Direct Image Alignment"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import cv2

import geometry.geometry as geometry
import geometry.projective_ops as proj_ops
from models.algorithms import feature_gradient, LM_H

from utils.run_utils import check_nan

from utils.timers import *
from lietorch import SE3


class InvDirectImageAlign(nn.Module):
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

        # Pre-compute jacobian
        self.timers.tic('compute pre-computable Jacobian components')

        x0, valid, Jp, Jt = proj_ops.Batch_projective_transform(
                pose, invD1, intrinsics, jacobian=True, return_depth=True
            )
        Jw = torch.matmul(Jp, Jt)

        self.timers.toc('compute pre-computable Jacobian components')

        # iteration
        for itr in range(self.num_iterations):
            self.timers.tic('compute warping residuals and Jacobian')
            x0, valid = proj_ops.Batch_projective_transform(
                    pose, invD1, intrinsics, jacobian=False, return_depth=True
                )

            inbound_image = (
                (x0[:, 0] > 0) & (x0[:, 0] < W - 1) & (x0[:, 1] > 0) & (x0[:, 1] < H - 1)
            )
            valid = inbound_image & valid

            u, v, d = x0.unbind(dim=1)
            u /= W - 1
            v /= H - 1
            x0_grid = torch.stack([u, v], dim=-1)
            x0_grid = ((x0_grid - 0.5) * 2).clamp(-2, 2)

            dI0_dx, dI0_dy = feature_gradient(I0)
            dI0_dx = f.grid_sample(dI0_dx, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            dI0_dy = f.grid_sample(dI0_dy, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            grad_I0 = torch.stack((dI0_dx, dI0_dy), dim=-1)

            dD0_dx, dD0_dy = feature_gradient(invD0)
            dD0_dx = f.grid_sample(dD0_dx, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            dD0_dy = f.grid_sample(dD0_dy, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            grad_invD0 = torch.stack((dD0_dx, dD0_dy), dim=-1)


            I0_warped = f.grid_sample(
                I0, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            r_I = I1 - I0_warped # [B, C, H, W]
            removed_area = torch.ones_like(I1) * 1e-6
            r_I = torch.where(~valid, removed_area, r_I)
            assert check_nan(r_I) == 0

            invD0_warped = f.grid_sample(
                invD0, x0_grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            r_Z = d[:, None] - invD0_warped # [B, 1, H, W]
            removed_area = torch.ones_like(invD0) * 1e-6
            r_Z = torch.where(~valid, removed_area, r_Z)
            assert check_nan(r_Z) == 0
            # print(f"photo.: {r_I.max().data}, geo.: {r_Z.max().data}")
            
            J_I = torch.einsum('ijklmn,iklnx->ijklmx', grad_I0.unsqueeze(-2), Jw) # [B, C, H, W, 1, 6]
            J_I = - J_I
            assert check_nan(J_I) == 0
            
            J_Z = torch.einsum('ijklmn,iklnx->ijklmx', grad_invD0.unsqueeze(-2), Jw) - Jt[..., [2], :][:, None] # [B, 1, H, W, 1, 6]
            J_Z = - J_Z
            assert check_nan(J_Z) == 0

            J_I = J_I.reshape(B, -1, 6)
            J_Z = J_Z.reshape(B, -1, 6)
            J = torch.cat((J_I, self.lambda_ * J_Z), dim=1) # [B, HXWX(C+1), 6]

            r_I = r_I.view(B, -1)
            r_Z = r_Z.view(B, -1)
            r = torch.cat((r_I, self.lambda_ * r_Z), dim=1) # [B, HXWX(C+1)]
            self.timers.toc('compute warping residuals and Jacobian')

            self.timers.tic("Robust estimator")
            weights = self.mEstimator(r)
            self.timers.toc("Robust estimator")

            self.timers.tic('solve x=A^{-1}b')
            WJ = weights[..., None] * J
            JtWJ =  torch.bmm(J.permute(0, 2, 1), WJ)
            Hessian = LM_H(JtWJ)
            Wr = weights * r
            Rhs = torch.bmm(J.permute(0, 2, 1), Wr[..., None])
            
            pose = inverse_update_pose(Hessian, Rhs, pose)
            self.timers.toc('solve x=A^{-1}b')
        
        return pose


def inverse_update_pose(H, Rhs, pose: SE3):
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
    pose = pose * SE3.exp(xi.squeeze(-1)).inv()
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

 
def LM_H(JtWJ):
    """Add a small diagonal damping. Without it, the training becomes quite unstable. Though no clear difference has been observed in inference without it."""
    B, L, _ = JtWJ.shape
    diagJtJ = torch.diag_embed(torch.diagonal(JtWJ, dim1=-2, dim2=-1))
    traceJtJ = torch.sum(diagJtJ, (2, 1))[:, None].expand(B, L)
    epsilon = torch.diag_embed(traceJtJ * 1e-6)
    Hessian = JtWJ + epsilon
    return Hessian     
    
    

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

    dia = InvDirectImageAlign(False, 0.01, timers=NullTimer())
    with torch.no_grad():
        for batch in torch_loader:
            color0, color1, depth0, depth1, pose, calib = batch['data']
            print(pose.shape, depth1.shape)

            quat = Rotation.from_matrix(pose[:, :3, :3]).as_quat()
            trans = pose[:, :3, 3]
            pose_data = np.concatenate((trans, quat), axis=-1)
            T = SE3.InitFromVec(torch.tensor(pose_data).float())
            
            dia(T, color0, color1, depth0, depth1, calib)
            