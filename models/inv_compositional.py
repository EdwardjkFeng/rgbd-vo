import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models

import geometry.geometry as geometry

from models.algorithms import feature_gradient

from utils.timers import NullTimer


class TrustRegionInverseCompositional(nn.Module):
    def __init__(
        self,
        max_iter=3,
        mEst_func=None,
        solver_func=None,
        timers=None,
    ) -> None:
        super().__init__()

        self.max_iterations = max_iter
        self.mEstimator = mEst_func
        self.directSolver = solver_func
        self.timers = timers if timers is not None else NullTimer()

    def forward(self, pose, x0, x1, invD0, invD1, K, wPrior=None):
        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        self.timers.tic("pre-compute Jacobian")
        J_F_p = self.__precompute_Jacobian(x0, invD0, px, py, K)
        self.timers.toc("pre-compute Jacobian")

        self.timers.tic("compute residuals")
        residuals, occ = self.__compute_warped_residual(
            pose, x0, x1, invD0, invD1, px, py, K
        )
        self.timers.toc("compute residuals")

        self.timers.tic("robust estimator")
        weights = self.mEstimator(residuals, x0, x1, wPrior)
        self.timers.toc("robust estimator")

        self.timers.tic("pre-compute JtWJ")
        # J_F_p[occ.expand(B, C, H, W).view(B, -1), :] = 1e-8
        WJ = weights.view(B, -1, 1) * J_F_p
        Jt = J_F_p.transpose(1, 2)
        JtWJ = torch.bmm(Jt, WJ)
        self.timers.toc("pre-compute JtWJ")

        # Iterative solve for increment
        for idx in range(self.max_iterations):
            self.timers.tic("solve x=A^{-1}b")
            pose = self.directSolver(
                JtWJ, Jt, weights, residuals, pose, x0, x1, invD0, invD1, K
            )
            self.timers.toc("solve x=A^{-1}b")

            self.timers.tic("compute residuals")
            residuals, occ = self.__compute_warped_residual(
                pose, x0, x1, invD0, invD1, px, py, K
            )

        return pose, weights
    
    def __precompute_Jacobian(self, x, invD, px, py, K):
        JF_x, JF_y = feature_gradient(x)
        Jx_p, Jy_p = self.__compute_Jacobian_warping(invD, px, py, K)
        JF_p = self.__compute_Jacobian_dFdp(JF_x, JF_y, Jx_p, Jy_p)
        return JF_p
    
    def __compute_Jacobian_warping(self, inv_depth, px, py, K):
        B, C, H, W = inv_depth.shape
        assert(C == 1)

        x = px.view(B, -1, 1)
        y = py.view(B, -1, 1)
        invD = inv_depth.view(B, -1, 1)

        xy = x * y
        O = torch.zeros_like(invD)

        dx_dp = torch.cat((       -xy, 1 + x*x, -y, invD,    O, -invD*x), dim=2)
        dy_dp = torch.cat((-(1 + y*y),      xy,  x,    O, invD, -invD*y), dim=2)
        
        fx, fy, _, _ = torch.split(K, 1, dim=1)
        return dx_dp*fx.view(B, 1, 1), dy_dp*fy.view(B, 1, 1)
    
    def __compute_Jacobian_dFdp(self, dF_dx, dF_dy, dx_dp, dy_dp):
        B, C, H, W = dF_dx.shape

        JF_p = (dF_dx.view(B, C, -1, 1) * dx_dp.view(B, 1, -1, 6)) + \
            (dF_dy.view(B, C, -1, 1) * dy_dp.view(B, 1, -1, 6))

        return JF_p.view(B, -1, 6)
    

    def __compute_warped_residual(self, pose, x0, x1, invD0, invD1, px, py, K):
        u_warped, v_warped, invD_warped = geometry.batch_warp_inverse_depth(
            px, py, invD0, pose, K
        )
        x1_1to0 = geometry.warp_features(x1, u_warped, v_warped)
        occ = self.__check_occ(invD_warped, invD1, u_warped, v_warped)

        residuals = x1_1to0 - x0
        
        B, C, H, W = x0.shape
        
        residuals[occ.expand(B, C, H, W)] = 0.

        return residuals, occ
    
    def __check_occ(self, inv_z_buffer, inv_z_ref, u, v, thres=1e-1):
        """ z-buffering check of occlusion 
        :param inverse depth of target frame
        :param inverse depth of reference frame
        """
        B, _, H, W = inv_z_buffer.shape

        inv_z_warped = geometry.warp_features(inv_z_ref, u, v)
        inlier = (inv_z_buffer > inv_z_warped - thres)

        valid_depth = (inv_z_buffer > 0) & (inv_z_warped > 0)

        inviews = inlier & (u > 0) & (u < W) & \
            (v > 0) & (v < H)

        return inviews.logical_not()