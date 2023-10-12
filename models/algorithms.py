"""
The algorithm backbone
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as func

# import modules for geometry operations, e.g., projection, transformation
from geometry import geometry
from models.submodules import convLayer as conv
from models.submodules import fcLayer, initialize_weights

# import utils for visualization, profiling
from utils import visualize
from utils import tools
from utils.timers import Timer, NullTimer
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
        max_iter=3,
        mEst_func=None,
        solver_func=None,
        timers=None,
    ):
        super().__init__()

        self.max_iterations = max_iter
        self.mEstimator = mEst_func
        self.directSolver = solver_func
        self.timers = timers

    def forward(
        self,
        T_10,
        F0,
        F1,
        invD0,
        invD1,
        K,
        wPrior=None,
        vis_res=False,
        obj_mask0=None,
        obj_mask1=None,
    ):
        """Forward function.

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
            self.timers = NullTimer()

        B, C, H, W = F0.shape
        # pre-compute stage
        # generate image coordinate grids
        px, py = geometry.generate_xy_grid(B, H, W, K)

        # pre-compute Jacobian
        self.timers.tic("pre-compute Jacobians")
        J_F_p = self.precompute_Jacobian(invD0, F0, px, py, K)  # [B, H x W, 6]
        self.timers.toc("pre-compute Jacobians")

        # compute warping residuals
        self.timers.tic("compute warping residuals")
        residuals, occ = compute_warped_residual(
            T_10, invD0, invD1, F0, F1, px, py, K, obj_mask0, obj_mask1
        )  # [B, 1, H, W]
        self.timers.toc("compute warping residuals")

        # m-estimator predict weights
        self.timers.tic("robust estimator")
        weights = self.mEstimator(residuals, F0, F1, wPrior)  # [B, C, H, W]
        WJ = weights.view(B, -1, 1) * J_F_p  # [B, H x W, 6]
        self.timers.toc("robust estimator")

        # pre-computer JtWJ
        self.timers.tic("pre-compute JtWJ")
        JtWJ = torch.bmm(torch.transpose(J_F_p, 1, 2), WJ)  # [B, 6, 6]
        self.timers.toc("pre-compute JtWJ")

        # iteration
        for idx in range(self.max_iterations):
            # solve delta = A^{-1} b
            self.timers.tic("solve delta = A^{-1} b")
            T_10 = self.directSolver(
                JtWJ,
                torch.transpose(J_F_p, 1, 2),
                weights,
                residuals,
                T_10,
                invD0,
                invD1,
                F0,
                F1,
                K,
                obj_mask1,
            )
            self.timers.toc("solve delta = A^{-1} b")

            # compute warping residuals
            self.timers.tic("compute warping residuals")
            residuals, occ = compute_warped_residual(
                T_10, invD0, invD1, F0, F1, px, py, K, obj_mask0, obj_mask1
            )  # [B, 1, H, W]
            self.timers.toc("compute warping residuals")

            # visualize residuals
            if vis_res:
                with torch.no_grad():
                    u_warped, v_warped, _ = geometry.batch_warp_inverse_depth(
                        px, py, invD0, T_10, K
                    )
                    F1_1to0 = geometry.warp_features(F1, u_warped, v_warped)
                    F_residual = visualize.create_mosaic(
                        [F0, F1, F1_1to0, residuals],
                        cmap=["NORMAL", "NORMAL", "NORMAL", cv2.COLORMAP_JET],
                        order="CHW",
                    )
                    cv2.namedWindow("feature-metric residuals", cv2.WINDOW_NORMAL)
                    cv2.imshow("feature-metric residuals", F_residual)
                    visualize.manage_visualization()

        return T_10, weights

    def precompute_Jacobian(self, invD, F, px, py, K):
        """Pre-compute the image Jacobian on the reference frame

        Args:
            invD: invD, template Depth
            F: template feature
            px: normalized image coordinate in cols (x)
            py: noramlized image coordinate in rows (y)
            K: the intrinsic parameters, [fx, fy, cx, cy]

        Returns:
            precomputed image (feature map) Jacobian on template
        """
        JF_x, JF_y = feature_gradient(F)  # [B, 1, H, W], [B, 1, H, W]
        Jx_p, Jy_p = compute_Jacobian_warping(
            invD, K, px, py
        )  # [B, HxW, 6], [B, HxW, 6]
        J_F_p = compute_Jacobian_dFdp(JF_x, JF_y, Jx_p, Jy_p)  # [B, HxW, 6]

        return J_F_p

    def forward_residuals(
        self,
        pose,
        F0,
        F1,
        invD0,
        invD1,
        K,
        wPrior=None,
        vis_res=False,
        obj_mask0=None,
        obj_mask1=None,
    ):
        B, C, H, W = F0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        residuals, occ = compute_warped_residual(
            pose,
            invD0,
            invD1,
            F0,
            F1,
            px,
            py,
            K,
            obj_mask0=obj_mask0,
            obj_mask1=obj_mask1,
        )  # [B, 1, H, W]

        # weighting via learned robust cost function
        weights = self.mEstimator(residuals, F0, F1, wPrior)  # [B, C, H, W]
        residuals = weights * residuals

        loss = compute_avg_loss([residuals], occ)

        return loss


class TrustRegionForwardWithUncertainty(nn.Module):
    """
    Direct Dense Tracking based on forward trust region and feature-metric
    uncertainty
    """

    def __init__(
        self,
        max_iter=3,
        mEst_func=None,
        solver_func=None,
        timers=None,
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
        self.mEstimator = mEst_func
        self.directSolver = solver_func
        self.timres = timers

    def forward(
        self, T_10, F0, F1, invD0, invD1, K, sigma0, sigma1, wPrior=None, vis_res=True
    ):
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
        max_iter=3,
        mEst_func=None,
        solver_func=None,
        timers=None,
        uncer_prop=False,
        combine_icp=False,
        scale_func=None,
        remove_tru_sigma=False,
    ):
        self.max_iterations = max_iter
        self.mEstimator = mEst_func
        self.directSolver = solver_func
        self.timers = timers
        self.uncer_prop = uncer_prop
        self.combine_icp = combine_icp
        self.scale_func = scale_func
        self.remove_tru_sigma = remove_tru_sigma  # remove truncated uncertainty

    def forward(
        self,
        T_10,
        F0,
        F1,
        invD0,
        invD1,
        K,
        sigma0,
        sigma1,
        wPrior=None,
        vis_res=True,
        obj_mask0=None,
        obj_mask1=None,
    ):
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


class DeepRobustEstimator(nn.Module):
    """The M-estimator

    When use estimator_type = 'MultiScale2w', it is the proposed convolutional M-estimator
    """

    def __init__(self, estimator_type):
        super(DeepRobustEstimator, self).__init__()

        if estimator_type == "MultiScale2w":
            self.D = 4
        elif estimator_type == "None":
            self.mEst_func = self.__constant_weight
            self.D = -1
        else:
            raise NotImplementedError()

        if self.D > 0:
            self.net = nn.Sequential(
                conv(True, self.D, 16, 3, dilation=1),
                conv(True, 16, 32, 3, dilation=2),
                conv(True, 32, 64, 3, dilation=4),
                conv(True, 64, 1, 3, dilation=1),
                nn.Sigmoid(),
            )
            initialize_weights(self.net)
        else:
            self.net = None

    def forward(self, residual, x0, x1, ws=None):
        """
        :param residual, the residual map
        :param x0, the feature map of the template
        :param x1, the feature map of the image
        :param ws, the initial weighted residual
        """
        if self.D == 1:  # use residual only
            context = residual.abs()
            w = self.net(context)
        elif self.D == 4:
            B, C, H, W = residual.shape
            wl = func.interpolate(ws, (H, W), mode="bilinear", align_corners=True)
            context = torch.cat((residual.abs(), x0, x1, wl), dim=1)
            w = self.net(context)
        elif self.D < 0:
            w = self.mEst_func(residual)

        return w

    def __weight_Huber(self, x, alpha=0.02):
        """weight function of Huber loss:
        refer to P. 24 w(x) at
        https://members.loria.fr/moberger/Enseignement/Master2/Documents/ZhangIVC-97-01.pdf

        Note this current implementation is not differentiable.
        """
        abs_x = torch.abs(x)
        linear_mask = abs_x > alpha
        w = torch.ones(x.shape).type_as(x)

        if linear_mask.sum().item() > 0:
            w[linear_mask] = alpha / abs_x[linear_mask]
        return w

    def __constant_weight(self, x):
        """mimic the standard least-square when weighting function is constant"""
        return torch.ones(x.shape).type_as(x)


class DirectSolverNet(nn.Module):
    # the enum types for direct solver
    SOLVER_NO_DAMPING = 0
    SOLVER_RESIDUAL_VOLUME = 1

    def __init__(self, solver_type, samples=10, direction="inverse") -> None:
        super().__init__()
        self.direction = direction
        if solver_type == "Direct-Nodamping":
            self.net = None
            self.type = self.SOLVER_NO_DAMPING
        elif solver_type == "Direct-ResVol":
            # flattened JtJ and JtR (number of samples, currently fixed at 10)
            self.samples = samples
            self.net = self.__deep_damping_regressor(D=6 * 6 + 6 * samples)
            self.type = self.SOLVER_RESIDUAL_VOLUME
            initialize_weights(self.net)
        else:
            raise NotImplementedError()

    def forward(
        self, JtJ, Jt, weights, R, pose0, invD0, invD1, x0, x1, K, obj_mask1=None
    ):
        """
        :param JtJ, the approximated Hessian JtJ, [B, 6, 6]
        :param Jt, the trasposed Jacobian, [B, 6, CXHXW]
        :param weights, the weight matrix, [B, C, H, W]
        :param R, the residual, [B, C, H, W]
        :param pose0, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param x0, the template feature map, [B, C, H, W]
        :param x1, the image feature map, [B, C, H, W]
        :param K, the intrinsic parameters

        -----------
        :return updated pose
        """

        B = JtJ.shape[0]

        wR = (weights * R).view(B, -1, 1)  # [B, CXHXW, 1]
        JtR = torch.bmm(Jt, wR)  # [B, 6, 1]

        if self.type == self.SOLVER_NO_DAMPING:
            # Add a small diagonal damping. Without it, the training becomes quite unstable
            # Do not see a clear difference by removing the damping in inference though
            Hessian = LM_H(JtJ)
        elif self.type == self.SOLVER_RESIDUAL_VOLUME:
            Hessian = self.__regularize_residual_volume(
                JtJ,
                Jt,
                JtR,
                weights,
                pose0,
                invD0,
                invD1,
                x0,
                x1,
                K,
                sample_range=self.samples,
                obj_mask1=obj_mask1,
            )
        else:
            raise NotImplementedError()

        if self.direction == "forward":
            updated_pose = forward_update_pose(Hessian, JtR, pose0)
        elif self.direction == "inverse":
            updated_pose = inverse_update_pose(Hessian, JtR, pose0)
        else:
            raise NotImplementedError(
                "pose updated should be either forward or inverse"
            )
        return updated_pose

    def __regularize_residual_volume(
        self,
        JtJ,
        Jt,
        JtR,
        weights,
        pose,
        invD0,
        invD1,
        x0,
        x1,
        K,
        sample_range,
        obj_mask1=None,
    ):
        """regularize the approximate with residual volume

        :param JtJ, the approximated Hessian JtJ
        :param Jt, the trasposed Jacobian
        :param JtR, the Right-hand size residual
        :param weights, the weight matrix
        :param pose, the initial estimated pose
        :param invD0, the template inverse depth map
        :param invD1, the image inverse depth map
        :param K, the intrinsic parameters
        :param x0, the template feature map
        :param x1, the image feature map
        :param sample_range, the numerb of samples

        ---------------
        :return the damped Hessian matrix
        """
        # the following current support only single scale
        JtR_volumes = []

        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        diag_mask = torch.eye(6).view(1, 6, 6).type_as(JtJ)
        diagJtJ = diag_mask * JtJ
        traceJtJ = torch.sum(diagJtJ, (2, 1))
        epsilon = (traceJtJ * 1e-6).view(B, 1, 1) * diag_mask
        n = sample_range
        lambdas = torch.logspace(-5, 5, n).type_as(JtJ)

        for s in range(n):
            # the epsilon is to prevent the matrix to be too ill-conditioned
            D = lambdas[s] * diagJtJ + epsilon
            Hessian = JtJ + D
            pose_s = inverse_update_pose(Hessian, JtR, pose)

            res_s, _ = compute_warped_residual(
                pose_s, invD0, invD1, x0, x1, px, py, K, obj_mask1=obj_mask1
            )
            JtR_s = torch.bmm(Jt, (weights * res_s).view(B, -1, 1))
            JtR_volumes.append(JtR_s)

        JtR_flat = torch.cat(tuple(JtR_volumes), dim=2).view(B, -1)
        JtJ_flat = JtJ.view(B, -1)
        damp_est = self.net(torch.cat((JtR_flat, JtJ_flat), dim=1))
        R = diag_mask * damp_est.view(B, 6, 1) + epsilon  # also lift-up

        return JtJ + R

    def __deep_damping_regressor(D):
        """Output a damping vector at each dimension"""
        net = nn.Sequential(
            fcLayer(in_planes=D, out_planes=128, bias=True),
            fcLayer(in_planes=128, out_planes=256, bias=True),
            fcLayer(in_planes=256, out_planes=6, bias=True),
        )  # the last ReLU makes sure every predicted value is positive
        return net


def compute_warped_residual(
    pose10, invD0, invD1, F0, F1, px, py, K, obj_mask0=None, obj_mask1=None
):
    """Compute the residual error of warped target image w.r.t. the reference feature map.

    Args:
        pose10: the forward warping pose from the reference camera to the target frame. Note that warping from the target frame to the reference frame is the inverse of this operation.
        invD0: reference inverse depth
        invD1: target inverse depth
        F0: reference feature image
        F1: target feature image
        px: pixel x map
        py: pixel y map
        K: intrinsic calibraiton
        obj_mask0: reference object mask. Defaults to None.
        obj_mask1: target object mask. Defaults to None.

    Returns:
        residual (of reference image), and occlusion information
    """
    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
        px, py, invD0, pose10, K
    )
    F1_1to0 = geometry.warp_features(F1, u_warped, v_warped)
    crd = torch.cat((u_warped, v_warped), dim=1)
    occ = geometry.check_occ(inv_z_warped, invD1, crd)

    residuals = F1_1to0 - F0

    B, C, H, W = F0.shape

    # determine whehter the object is in-view
    if obj_mask0 is not None:
        bg_mask0 = -obj_mask0
        occ = occ | (bg_mask0.view(B, 1, H, W))
    if obj_mask1 is not None:
        # determine wehter the object is in-view
        obj_mask1_r = geometry.warp_features(obj_mask1.float(), u_warped, v_warped) > 0
        bg_mask1 = ~obj_mask1_r
        occ = occ | (bg_mask1.view(B, 1, H, W))

    residuals[occ.expand(B, C, H, W)] = 1e-3

    return residuals, occ


def feature_gradient(F, normalize_gradient=True):
    """Calcualte the gradient on the feature space using Sobel operator

    Args:
        img: input image
        normalize_gradient: whether to normalized the gradient. Defaults to True.
    """
    B, C, H, W = F.shape

    sobel_x = (
        torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(F)
    )
    sobel_y = (
        torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).to(F)
    )

    F_pad = func.pad(F.view(-1, 1, H, W), (1, 1, 1, 1), mode="replicate")
    dF_dx = func.conv2d(F_pad, sobel_x, stride=1, padding=0)
    dF_dy = func.conv2d(F_pad, sobel_y, stride=1, padding=0)

    if normalize_gradient:
        mag = torch.sqrt((dF_dx**2) + (dF_dy**2) + 1e-8)
        dF_dx = dF_dx / mag
        dF_dy = dF_dy / mag

    return dF_dx.view(B, C, H, W), dF_dy.view(B, C, H, W)


def compute_Jacobian_dFdp(JF_x, JF_y, Jx_p, Jy_p):
    """chained gradient of feature map w.r.t. the pose

    Args:
        JF_x: Jacobian of the feature map in x direction [B, C, H, W]
        JF_y: Jacobian of the feature map in y direction [B, C, H, W]
        Jx_p: Jacobian of the x map to manifold p [B, HxW, 6]
        Jy_p: Jacobian of the y map to manifold p [B, HxW, 6]

    Returns:
        the feature map Jacobian in x, y direction, [B, 2, 6]
    """
    B, C, H, W = JF_x.shape

    # precompute J_F_p, JtWJ
    JF_p = JF_x.view(B, C, -1, 1) * Jx_p.view(B, 1, -1, 6) + JF_y.view(
        B, C, -1, 1
    ) * Jy_p.view(B, 1, -1, 6)

    return JF_p.view(B, -1, 6)


def compute_Jacobian_warping(invD, K, px, py, pose=None):
    """Compute teh Jacobian matrix of the warped (x, y) w.r.t. the inverse depth (linearized at origin)

    Args:
        invD: the inverse depth
        K: the intrinsic calibration
        px: the pixel x map
        py: the pixel y map
        pose: the relative transformation. Defaults to None.

    Returns:
        the warping Jacobian in x, y direction
    """
    B, C, H, W = invD.shape
    assert C == 1

    if pose is not None:
        x_y_invz = torch.cat((px, py, invD), dim=1)
        R, t = pose
        warped = torch.bmm(R, x_y_invz.view(B, 2, H * W)) + t.view(B, 3, 1).expand(
            B, 3, H * W
        )
        px, py, invD = warped.split(1, dim=1)

    x = px.view(B, -1, 1)
    y = py.view(B, -1, 1)
    invz = invD.view(B, -1, 1)

    xy = x * y
    o = torch.zeros((B, H * W, 1)).to(invD)

    dx_dp = torch.cat((-xy, 1 + x**2, -y, invz, o, invz * x), dim=2)
    dy_dp = torch.cat((-1 - y**2, xy, x, o, invz, -invz * y), dim=2)

    fx, fy, cx, cy = torch.unbind(K, dim=1)
    return dx_dp * fx.view(B, 1, 1), dy_dp * fy.view(B, 1, 1)


def compute_avg_loss(x_list: list, invalid_area):
    assert isinstance(x_list, list)
    valid_res_list = []
    for x in x_list:
        removed_area = torch.zeros_like(x)
        valid_res = torch.where(invalid_area, removed_area, x)
        valid_res_list.append(valid_res)
    B, C, H, W = invalid_area.shape
    valid_pixel_num = H * W - invalid_area.sum(dim=[2, 3]).squeeze()

    loss_t = torch.zeros_like(invalid_area).float()
    for valid_res in valid_res_list:
        loss_t += (valid_res**2).sum(dim=1, keepdim=True)
    loss_sum = loss_t.sum(dim=[2, 3], keepdim=True).squeeze()
    avg_loss = loss_sum / valid_pixel_num
    # print("avg", loss_sum**0.5/valid_pixel_num)
    return avg_loss


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
        # because the jacobian is also including the minus signal, it should be (J^T * J) J^T * r
        # xi = - xi
    return xi


def inverse_update_pose(H, Rhs, pose):
    """Use left-multiplication for the pose update in the inverse compositional form. Here kxi (se3) is formulated in the order of [rot, trs]

    Args:
        H: Hessian
        Rhs: Right-hand side vector
        pose: the initial pose (forward transform inverse of xi)

    Returns:
        forward updated pose (inverse of xi)
    """
    xi = least_square_solve(H, Rhs)
    # simplifed inverse compositional for SE3: (delta_ksi)^-1
    d_R = geometry.batch_twist2Mat(-xi[:, :3].view(-1, 3))
    d_t = -torch.bmm(d_R, xi[:, 3:])

    R, t = pose
    pose = geometry.batch_Rt_compose(R, t, d_R, d_t)
    return pose


def forward_update_pose(H, Rhs, pose):
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
    d_R = geometry.batch_twist2Mat(xi[:, :3].view(-1, 3))
    d_t = xi[:, 3:]

    R, t = pose
    pose = geometry.batch_Rt_compose(R, t, d_R, d_t)
    return pose


def LM_H(JtWJ):
    """Add a small diagonal damping. Without it, the training becomes quite unstable. Though no clear difference has been observed in inference without it."""
    B, L, _ = JtWJ.shape
    diagJtJ = torch.diag_embed(torch.diagonal(JtWJ, dim1=-2, dim2=-1))
    traceJtJ = torch.sum(diagJtJ, (2, 1))[:, None].expand(B, L)
    epsilon = torch.diag_embed(traceJtJ * 1e-6)
    Hessian = JtWJ + epsilon
    return Hessian
