"""
The algorithm backbone
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as func


# import modules for geometry operations, e.g., projection, transformation

# import tools for visualization


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


    def forward(self, T10, F0, F1, invD0, invD1, K, wPrior=None,
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
            pose: the estimated pose
            weights: weights predicted by the convolutional m-estimator.
        """
        
        # pre-compute stage

            # generate image coordinate grids

            # pre-compute Jacobian

            # compute warping residuals

            # m-estimator predict weights

            # pre-computer JtWJ

        # iteration

            # solve delta = A^{-1} b

            # compute warping residuals


            # visualize residuals



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