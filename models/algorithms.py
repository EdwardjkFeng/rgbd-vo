"""
The algorithm backbone
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as func


# import modules for geometry operations, e.g., projection, transformation
from geometry import geometry
# import tools for visualization

from utils import tools

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TrustRegionInverseComposition(nn.Module):
    """
    This is the base function of the trust-region based inverse compositional 
    algorithm.
    """
    def __init__(
        self,
        max_iter    = 3,
        mEst_func   = None,
        solver_func = None,
        timers      = None,
    ):
        super().__init__()

        self.max_iterations = max_iter
        self.mEstimator     = mEst_func
        self.directSolver   = solver_func
        self.timers         = timers


    def forward(self, T_10, F0, F1, invD0, invD1, K, wPrior=None,
                vis_res=False, obj_mask0=None, obj_mask1=None):
        """ Forward function. 

        Args:
            T10: the initial pose 
                (extrinsic of the target frame w.r.t. the reference frame)
            F0: the reference features 
            F1: the target features
            invD0: the reference inverse depth
            invD1: the target inverse depth
            K: the intrinsic parameters, [fx, fy, cx, cy]
            wPrior: (optional) provide an initial weight as input to the convolutional m-estimator. Defaults to None.
            vis_res: whether to visualize the residual maps. Defaults to False.
            obj_mask0: the reference object mask. Defaults to None.
            obj_mask1: the target object mask. Defaults to None.

        Returns:
            T_10: the estimated pose
            weights: weights predicted by the convolutional m-estimator.
        """
        
        if self.timers is None:
            self.timers = tools.NullTimer()

        B, C, H, W = F0.shape
        # pre-compute stage
        # generate image coordinate grids
        px, py = geometry.generate_xy_grid(B, H, W, K)

        # pre-compute Jacobian
        self.timers.tic('pre-compute Jacobians')
        J_F_p = self.precompute_Jacobian(invD0, F0, px, py, K) # [B, H x W, 6]
        self.timers.toc('pre-compute Jacobians')

        # compute warping residuals
        self.timers.tic('compute warping residuals')
        residuals, occ = compute_warped_residual(
            T_10, invD0, invD1, F0, F1, px, py, K, obj_mask0, obj_mask1
        ) # [B, 1, H, W]
        self.timers.toc('compute warping residuals')

        # m-estimator predict weights
        self.timers.tic('robust estimator')
        weights = self.mEstimator(residuals, F0, F1, wPrior) # [B, C, H, W]
        WJ = weights.view(B, -1, 1) * J_F_p  # [B, H x W, 6]
        self.timers.toc('robust estimator')

        # pre-computer JtWJ
        self.timers.tic('pre-compute JtWJ')
        JtWJ = torch.bmm(torch.transpose(J_F_p, 1, 2), WJ)  # [B, 6, 6]
        self.timers.toc('pre-compute JtWJ')

        # iteration
        for idx in range(self.max_iterations):
            # solve delta = A^{-1} b
            self.timers.tic('solve delta = A^{-1} b')
            T_10 = self.directSolver(
                JtWJ, torch.transpose(J_F_p, 1, 2), weights, residuals, 
                T_10, invD0, invD1, F0, F1, K, obj_mask1
            )
            self.timers.toc('solve delta = A^{-1} b')

            # compute warping residuals
            self.timers.tic('compute warping residuals')
            residuals, occ = compute_warped_residual(
                T_10, invD0, invD1, F0, F1, px, py, K, obj_mask0, obj_mask1
            ) # [B, 1, H, W]
            self.timers.toc('compute warping residuals')

            # visualize residuals
            if vis_res:
                with torch.no_grad():
                    u_warped, v_warped, inv_z_warped = \
                        geometry.batch_warp_inverse_depth(
                            px, py, invD0, T_10, K
                    )
                    F1_1to0 = geometry.warp_features(
                        F1, u_warped, v_warped
                    )
                    F_residual = display.create_mosaic(
                        [F0, F1, F1_1to0, residuals],
                        cmap=['NORMAL', 'NORMAL', 'NORMAL', cv2.COLORMAP_JET],
                        order='CHW'
                    )
                    cv2.namedWindow(
                        'feature-metric residuals', cv2.WINDOW_NORMAL
                    )
                    cv2.imshow("feature-metric residuals", F_residual)
                    cv2.waitKey(10)
        
        return T_10, weights

    def precompute_Jacobian(self, invD, F, px, py, K):
        

class TrustRegionForwardWithUncertainty(nn.Module):
    """
    Direct Dense Tracking based on forward trust region and feature-metric 
    uncertainty
    """

    def __init__(
        self,
        max_iter    = 3,
        mEst_func   = None,
        solver_func = None,
        timers      = None,
    ):
        """

        Args:
            max_iter: maximal number of iterations. Defaults to 3.
            mEst_func: M-Estimator. Defaults to None.
            solver_func: the trust-region function / network. Defaults to None.
            timers: timers object. If provided, counting timer for each step.
                Defaults to None.
        """
        super().__init__()

        self.max_iterations = max_iter
        self.mEstimator     = mEst_func
        self.directSolver   = solver_func
        self.timres         = timers

    def forward(self, T_10, F0, F1, invD0, invD1, K, sigma0, sigma1, 
                wPrior=None, vis_res=True):
        """

        Args:
            T10: the initial pose 
                (extrinsic of the target frame w.r.t. the reference frame)
            F0: the reference features 
            F1: the target features
            invD0: the reference inverse depth
            invD1: the target inverse depth
            K: the intrinsic parameters, [fx, fy, cx, cy]
            sigma0: the reference feature uncertainty
            sigma1: the target feature unceratinty
            wPrior: (optional) provide an initial weight as input to the convolutional m-estimator. Defaults to None.
            vis_res: whether to visualize the residual maps. Defaults to False.

        Returns:
            pose: the estimated pose
            weights: weights predicted by the convolutional m-estimator.
        """
        
        # Iteration

        # compute warping residuals

        # visualize feature-metric residuals

        # compute Jacobian

        # robust estimator perdicts weights

        # compute JtWJ

        # solve delta = A^{-1} b



class TrustRegionICWithUncertainty(nn.Module):
    """
    Direct Dense Tracking based on inverse compositional trust region and 
    feature-metric uncertainty.
    """

    def __init__(
        self,
        max_iter    = 3,
        mEst_func   = None,
        solver_func = None,
        timers      = None,
        uncer_prop  = False,
        combine_icp = False,
        scale_func  = None,
        remove_tru_sigma = False,
    ):
        self.max_iterations = max_iter
        self.mEstimator     = mEst_func
        self.directSolver   = solver_func
        self.timers         = timers
        self.uncer_prop     = uncer_prop
        self.combine_icp    = combine_icp
        self.scale_func     = scale_func
        self.remove_tru_sigma = remove_tru_sigma # remove truncated uncertainty


    def forward(self, T_10, F0, F1, invD0, invD1, K, sigma0, sigma1, 
                wPrior=None, vis_res=True, obj_mask0=None, obj_mask1=None):
        """

        Args:
            T_10: _description_
            F0: _description_
            F1: _description_
            invD0: _description_
            invD1: _description_
            K: _description_
            sigma0: _description_
            sigma1: _description_
            wPrior: _description_. Defaults to None.
            vis_res: _description_. Defaults to True.
            obj_mask0: _description_. Defaults to None.
            obj_mask1: _description_. Defaults to None.
        """

        # if combine icp 

        # precompute jacobian

        # iterations
        
        # compute warping function 

        # compute Jacobian

        # compute JtWJ and JtWr

        # if combine ICP, repeat the above steps

        # solve delta = A^{-1} b

        # visualize residuals 

        # update pose