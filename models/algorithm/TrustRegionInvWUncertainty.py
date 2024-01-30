"""Implementation of Combined Direct Image Alignment"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import cv2

import geometry.geometry as geometry
import geometry.projective_ops as proj_ops
from models.algorithms import feature_gradient, LM_H

import utils.visualize as display
from utils.timers import *
from utils.run_utils import check_nan
from lietorch import SE3


class TrustRegionInverseWUncertainty(nn.Module):
    """
    Dirct dense tracking based on trust region and feature-metric uncertainty
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
    ) -> None:
        super().__init__()

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
        pose10,
        x0,
        x1,
        invD0,
        invD1,
        K,
        sigma0,
        sigma1,
        wPrior=None,
        depth0=None,
        depth1=None,
        vis_res=True,
        obj_mask0=None,
        obj_mask1=None,
    ):
        """
        :param pose10, the initial pose
            (extrinsic of the target frame w.r.t. the referenc frame)
        :param x0, the template features
        :param x1, the image features
        :param invD0, the template inverse depth
        :param invD1, the image inverse depth
        :param K, the intrinsic parameters, [fx, fy, cx, cy]
        :param wPrior (optional), provide an initial weight as input to the convolutional scaler
        """
        assert sigma0 is not None and sigma1 is not None

        B, C, H, W = x0.shape
        px, py = geometry.generate_xy_grid(B, H, W, K)

        if self.combine_icp:
            assert depth0 is not None and depth1 is not None
            if self.timers:
                self.timers.tic("compute vertex and normal")
            vertex0 = geometry.compute_vertex(depth0, px, py)
            vertex1 = geometry.compute_vertex(depth1, px, py)
            normal0 = geometry.compute_normal(vertex0)
            normal1 = geometry.compute_normal(vertex1)
            if self.timers:
                self.timers.toc("compute vertex and normal")

        if self.timers:
            self.timers.tic("compute pre-computable Jacobian components")
        grad_f0, grad_sigma0, Jx_p, Jy_p = self.precompute_jacobian_components(
            invD0, x0, sigma0, px, py, K
        )
        if self.timers:
            self.timers.toc("compute pre-computable Jacobian components")

        #
        # if self.timers: self.timers.tic('robust estimator')
        # weights = self.mEstimator(weighted_res, x0, x1, wPrior)  # [B, C, H, W]
        # if self.timers: self.timers.toc('robust estimator')
        w_icp = None
        for idx in range(self.max_iterations):
            if self.timers:
                self.timers.tic("compute warping residuals")
            weighted_res, res, sigma, occ = compute_inverse_residuals(
                pose10,
                invD0,
                invD1,
                x0,
                x1,
                sigma0,
                sigma1,
                px,
                py,
                K,
                obj_mask0=obj_mask0,
                obj_mask1=obj_mask1,
                remove_truncate_uncertainty=self.remove_tru_sigma,
            )  # [B,C,H,W]
            if self.timers:
                self.timers.toc("compute warping residuals")

            # compute batch-wise average residual
            # print("feat:", compute_avg_res(weighted_res, occ))

            if self.timers:
                self.timers.tic("compose Jacobian components")
            J_F_p, _, _ = self.compose_inverse_jacobians(
                res, sigma, sigma0, grad_f0, grad_sigma0, Jx_p, Jy_p
            )  # [B,C,HXW,6]
            if self.timers:
                self.timers.toc("compose Jacobian components")

            if self.timers:
                self.timers.tic("compute JtWJ and JtR")
            # wJ = weights.view(B, -1, 1) * J_F_p  # [B,CXHXW,6]
            JtWJ = self.compute_JtJ(J_F_p)  # [B, 6, 6]
            JtR = self.compute_Jtr(J_F_p, weighted_res)  # [B, 6, 1]
            if self.timers:
                self.timers.toc("compute JtWJ and JtR")

            if self.combine_icp:
                if self.timers:
                    self.timers.tic("compute ICP residuals and jacobians")
                icp_residuals, icp_J, icp_occ = self.compute_ICP_residuals_jacobian(
                    vertex0,
                    vertex1,
                    normal0,
                    normal1,
                    pose10,
                    K,
                    obj_mask0=obj_mask0,
                    obj_mask1=obj_mask1,
                )
                if self.timers:
                    self.timers.toc("compute ICP residuals and jacobians")

                # use the scale computed at the first iteration
                # @TODO: test if we should also scale feature residuals
                if self.timers:
                    self.timers.tic("compute scale function")
                if idx == 0 or w_icp is None:
                    w_icp = self.scale_func(
                        icp_residuals, weighted_res, wPrior
                    )  # [B,1,H,W]
                icp_residuals = w_icp * icp_residuals
                icp_J = w_icp.view(B, 1, H * W, 1) * icp_J
                if self.timers:
                    self.timers.toc("compute scale function")
                # print("icp:", compute_avg_res(icp_residuals, icp_occ))

                if self.timers:
                    self.timers.tic("compute ICP JtWJ and JtR")
                icp_JtWJ = self.compute_JtJ(icp_J)  # [B, 6, 6]
                icp_JtR = self.compute_Jtr(icp_J, icp_residuals)
                JtWJ = JtWJ + icp_JtWJ
                JtR = JtR + icp_JtR
                if self.timers:
                    self.timers.toc("compute ICP JtWJ and JtR")

            if self.timers:
                self.timers.tic("solve x=A^{-1}b")
            pose10 = self.GN_solver(JtWJ, JtR, pose10)
            if self.timers:
                self.timers.toc("solve x=A^{-1}b")

            if vis_res:
                with torch.no_grad():
                    (
                        u_warped,
                        v_warped,
                        inv_z_warped,
                    ) = geometry.batch_warp_inverse_depth(px, py, invD0, pose10, K)
                    x1_1to0 = geometry.warp_features(x1, u_warped, v_warped)
                    feat_0 = display.visualize_feature_channels(x0, order="CHW")
                    feat_1 = display.visualize_feature_channels(x1, order="CHW")
                    feat_1_0 = display.visualize_feature_channels(x1_1to0, order="CHW")
                    feat_res = display.visualize_feature_channels(
                        weighted_res, order="CHW"
                    )

                    feat_residual = display.create_mosaic(
                        [feat_0, feat_1, feat_1_0, feat_res],
                        cmap=[
                            cv2.COLORMAP_JET,
                            cv2.COLORMAP_JET,
                            cv2.COLORMAP_JET,
                            cv2.COLORMAP_JET,
                        ],
                        order="CHW",
                    )
                    cv2.namedWindow("feature-metric residuals", cv2.WINDOW_NORMAL)
                    cv2.imshow("feature-metric residuals", feat_residual)
                    cv2.waitKey(10)
        # print("--->")
        if self.combine_icp:
            weights = w_icp
        else:
            weights = torch.ones(weighted_res.shape).type_as(weighted_res)
        if self.uncer_prop:
            # compute sigma_ksi: 6X6
            inv_sigma_ksi = JtWJ
            # Hessian = lev_mar_H(JtWJ)
            # sigma_ksi = invH(Hessian)
            return pose10, weights, inv_sigma_ksi
        else:
            return pose10, weights


    def compute_JtJ(self, J: torch.Tensor):
        # J in dimension of (B, C, HW, y)
        B, C, HW, y = J.shape
        J_reshape = J.permute(0, 2, 1, 3).contiguous() # [B, HW, C, 6]
        J_reshape = J_reshape.view(-1, C, y) # [BHW, C, 6]
        JtJ = torch.bmm(torch.transpose(J_reshape, 1, 2), J_reshape) # [BHW, 6, 6]
        JtJ = JtJ.view(B, HW, y, y)
        JtJ = JtJ.sum(dim=1)
        return JtJ # [B, 6, 6]
    
    def compute_Jtr(self, J: torch.Tensor, res: torch.Tensor):
        # J in the dimension of (B, C, HW, y)
        # res in the dimension of [B, C, H, W]
        B,C,H,W = res.shape
        res = res.view(B, C, H*W, 1).permute(0,2,1,3).contiguous()  # [B, HW, C, 1]
        res = res.view(-1,C,1)  # [B*HW, C, 1]
        J_reshape = J.permute(0, 2, 1, 3).contiguous()  # [B, HW, C, 6]
        J_reshape = J_reshape.view(-1, C, 6)  # [B*HW, C, 6]

        Jtr = torch.bmm(torch.transpose(J_reshape, 1, 2), res)  # [B*HW, 6, 1]
        Jtr = Jtr.view(B, H*W, 6, 1)
        Jtr = Jtr.sum(dim=1)
        return Jtr  # [B, 6, 1]
    
    def GN_solver(self, JtJ, Jtr, pose0):
        B = JtJ.shape[0]

        # Add a small diagonal damping. Without it, the training becomes quite unstable
        # Do not see a clear difference by removing the damping in inference though
        Hessian = lev_mar_H(JtJ)

        updated_pose = inverse_update_pose(Hessian, Jtr, pose0)

        return updated_pose
    
    def precompute_jacobian_components(self, invD0, f0, sigma0, px, py, K, grad_interp=False, crd0=None):
        if not grad_interp:
            # inverse: no need for interpolation in gradients: (linearized at origin)
            f0_gradx, f0_grady = feature_gradient(f0)
            sigma0_gradx, sigma0_grady = feature_gradient(sigma0)
        else:
            # gradients of bilinear interpolation
            if crd0 is None:
                _, _, H, W = f0.shape
                crd0 = geometry.gen_coordinate_tensors(W, H).unsqueeze(dim=0).double()
            f0_gradx, f0_grady = geometry.grad_bilinear_interpolation(crd0, f0, replace_nan_as_eps=True)
            sigma0_gradx, sigma0_grady = geometry.grad_bilinear_interpolation(crd0, sigma0, replace_nan_as_eps=True)

        grad_f0 = torch.stack((f0_gradx, f0_grady), dim=2)
        grad_sigma0 = torch.stack((sigma0_gradx, sigma0_grady), dim=2)
        Jx_p, Jy_p = compute_jacobian_warping(invD0, K, px, py)  # [B, HXW, 6], [B, HXW, 6]

        return grad_f0, grad_sigma0, Jx_p, Jy_p
    
    def compose_inverse_jacobians(self, res, sigma, sigma0, grad_f0, grad_sigma0, Jx_p, Jy_p):
        B, C, H, W = sigma0.shape
        res = res.unsqueeze(dim=2)
        sigma = sigma.unsqueeze(dim=2)
        sigma0 = sigma0.unsqueeze(dim=2)
        J_res_crd = - grad_f0 / sigma - res * (sigma0 * grad_sigma0 / (sigma ** 3))
        J_res_x, J_res_y = J_res_crd.split(1, dim=2)
        J_res_x.squeeze_(dim=2)
        J_res_y.squeeze_(dim=2)


        J_res_p = compute_jacobian_dIdp(J_res_x, J_res_y, Jx_p, Jy_p)  # [B, CXHXW, 6]
        J_res_rot, J_res_trs = J_res_p.view(B, C, H, W, 6).split(3, dim=-1)  # [B, C, H, W, 3]
        J_res_trs = J_res_trs.permute(0, 1, 4, 2, 3)  # [B, C, 3, H, W]
        J_res_rot = J_res_rot.permute(0, 1, 4, 2, 3)  # [B, C, 3, H, W]
        # follow the conversion of inverse the jacobian
        J_res_p = - J_res_p
        # separate channel and batch and pixel number
        J_res_p = J_res_p.view(B, C, -1, 6)  # [B, C, HXW, 6]
        assert check_nan(J_res_p) == 0
        return J_res_p, J_res_trs, J_res_rot
    
    def compute_ICP_residuals_jacobian(self, vertex0, vertex1, normal0, normal1, pose10, K, obj_mask0=None, obj_mask1=None):
        R, t = pose10
        B, C, H, W = vertex0.shape

        rot_vertex0_to1 = torch.bmm(R, vertex0.view(B, 3, H*W))
        vertex0_to1 = rot_vertex0_to1 + t.view(B, 3, 1).expand(B, 3, H*W)
        # normal0_to1 = torch.bmm(R, normal0.view(B, 3, H * W))

        fx, fy, cx, cy = torch.split(K, 1, dim=1)
        x_, y_, s_ = torch.split(vertex0_to1, 1, dim=1)
        u_ = (x_ / s_).view(B, -1) * fx + cx
        v_ = (y_ / s_).view(B, -1) * fy + cy

        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)

        # # interpolation-version
        r_vertex1 = geometry.warp_features(vertex1, u_, v_)
        r_normal1 = geometry.warp_features(normal1, u_, v_)

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
                           geometry.batch_skew(vertex0.view(B,3,-1).permute(0, 2, 1).contiguous().view(-1, 3))).squeeze()  # [B*H*W, 3]
        J_trs = -NtC10.view(-1,3)  # [B*H*W, 3]

        # compose jacobians
        J_F_p = torch.cat((J_rot, J_trs), dim=-1)  # follow the order of [rot, trs]  [B*H*W, 6]
        J_F_p = J_F_p.view(B, 1, -1, 6)  # [B, 1, HXW, 6]

        # covariance-normalized
        dpt0 = vertex0[:,2:3,:,:]
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


def compute_inverse_residuals(pose_10, invD0, invD1, x0, x1, sigma0, sigma1, px, py, K,
                              obj_mask0=None, obj_mask1=None, remove_truncate_uncertainty=False):
    u_warped, v_warped, inv_z_warped = geometry.batch_warp_inverse_depth(
        px, py, invD0, pose_10, K)
    warped_crd = torch.cat((u_warped, v_warped), dim=1)
    occ = geometry.check_occ(inv_z_warped, invD1, warped_crd)

    B, C, H, W = x0.shape
    if obj_mask0 is not None:
        bg_mask0 = ~obj_mask0
        occ = occ | (bg_mask0.view(B, 1, H, W))
    if obj_mask1 is not None:
        # determine whether the object is in-view
        u, v = warped_crd.split(1, dim=1)
        obj_mask1_r = geometry.warp_features(obj_mask1.float(), u, v)>0
        bg_mask1 = ~obj_mask1_r
        occ = occ | (bg_mask1.view(B, 1, H, W))

    (weighted_res, res, sigma,
     f_r, sigma_r, occ) = compose_residuals(warped_crd, occ, x0,
                                            x1, sigma0, sigma1,
                                            eps=1e-6,
                                            remove_tru_sigma=remove_truncate_uncertainty)
    return weighted_res, res, sigma, occ


def compose_residuals(warped_crd, invalid_mask, f0, f1, sigma0, sigma1,
                      eps=1e-6, perturbed_crd0=None, remove_tru_sigma=False):
    u, v = warped_crd.split(1, dim=1)
    [f_r, sigma_r] = [geometry.warp_features(img, u, v) for img in [f1, sigma1]]

    # if crd0 is perturbed, mainly for unit test
    if perturbed_crd0 is not None:
        u0, v0 = perturbed_crd0.split(1, dim=1)
        [f0, sigma0] = [geometry.warp_features(img, u0, v0) for img in [f0, sigma0]]
    res = f_r - f0
    # sigma = torch.sqrt(sigma_r.pow(2) + sigma0.pow(2) + eps) + eps
    sigma = torch.sqrt(sigma_r.pow(2) + sigma0.pow(2))
    weighted_res = res / sigma

    # handle the truncated uncertainty areas
    if remove_tru_sigma:
        sigma_tru = (sigma_r == sigma_r.min()) | (sigma_r == sigma_r.max()) | \
                    (sigma0 == sigma0.min()) | (sigma0 == sigma0.max())
        sigma_tru = sigma_tru[:, 0:1, :, :]  # a dirty way to handle 1-dim uncertainty
        invalid_mask = invalid_mask | sigma_tru

    # handle invalidity
    removed_area = torch.ones_like(f_r[0]) * eps
    weighted_res = torch.where(invalid_mask, removed_area, weighted_res)
    # res = torch.where(invalid_mask, removed_area, res)
    # sigma = torch.where(invalid_mask, removed_area, sigma)

    # assert check_nan(sigma) == 0
    assert check_nan(weighted_res) == 0
    return weighted_res, res, sigma, f_r, sigma_r, invalid_mask


def lev_mar_H(JtWJ):
    # Add a small diagonal damping. Without it, the training becomes quite unstable
    # Do not see a clear difference by removing the damping in inference though
    B, _, _ = JtWJ.shape
    diag_mask = torch.eye(6).view(1, 6, 6).type_as(JtWJ)
    diagJtJ = diag_mask * JtWJ
    traceJtJ = torch.sum(diagJtJ, (2, 1))
    epsilon = (traceJtJ * 1e-6).view(B, 1, 1) * diag_mask
    Hessian = JtWJ + epsilon
    return Hessian


def inverse_update_pose(H, Rhs, pose):
    """ Ues left-multiplication for the pose update 
    in the inverse compositional form
    refer to equation (10) in the paper 
    here ksi (se3) is formulated in the order of [rot, trs]
    :param the (approximated) Hessian matrix
    :param Right-hand side vector
    :param the initial pose (forward transform inverse of xi)
    ---------
    :return the forward updated pose (inverse of xi)
    """

    xi = least_square_solve(H, Rhs)
    # simplifed inverse compositional for SE3: (delta_ksi)^-1
    d_R = geometry.batch_twist2Mat(-xi[:, :3].view(-1,3))
    d_t = -torch.bmm(d_R, xi[:, 3:])

    R, t = pose
    pose = geometry.batch_Rt_compose(R, t, d_R, d_t) 
    return pose


def least_square_solve(H, Rhs):
    """
    x =  - H ^{-1} * Rhs
    importantly: use pytorch inverse to have a differential inverse operation
    :param H: Hessian
    :type H: [B, 6, 6]
    :param  Rhs: Right-hand side vector
    :type Rhs: [B, 6, 1]
    :return: solved ksi
    :rtype:  [B, 6, 1]
    """
    inv_H = invH(H)  # [B, 6, 6] square matrix
    xi = torch.bmm(inv_H, Rhs)
    # because the jacobian is also including the minus signal, it should be (J^T * J) J^T * r
    # xi = - xi
    return xi


def invH(H):
    """ Generate (H+damp)^{-1}, with predicted damping values
    :param approximate Hessian matrix JtWJ
    -----------
    :return the inverse of Hessian
    """
    # GPU is much slower for matrix inverse when the size is small (compare to CPU)
    # works (50x faster) than inversing the dense matrix in GPU
    if H.is_cuda:
        # invH = bpinv((H).cpu()).cuda()
        # invH = torch.inverse(H)
        invH = torch.inverse(H.cpu()).cuda()
    else:
        invH = torch.inverse(H)
    return invH


def compute_jacobian_warping(p_invdepth, K, px, py, pose=None):
    """ Compute the Jacobian matrix of the warped (x,y) w.r.t. the inverse depth
    (linearized at origin)
    :param p_invdepth the input inverse depth
    :param the intrinsic calibration
    :param the pixel x map
    :param the pixel y map
     ------------
    :return the warping jacobian in x, y direction
    """
    B, C, H, W = p_invdepth.size()
    assert(C == 1)

    if pose is not None:
        x_y_invz = torch.cat((px, py, p_invdepth), dim=1)
        R, t = pose
        warped = torch.bmm(R, x_y_invz.view(B, 3, H * W)) + \
                 t.view(B, 3, 1).expand(B, 3, H * W)
        px, py, p_invdepth = warped.split(1, dim=1)

    x = px.view(B, -1, 1)
    y = py.view(B, -1, 1)
    invd = p_invdepth.view(B, -1, 1)

    xy = x * y
    O = torch.zeros((B, H*W, 1)).type_as(p_invdepth)

    # This is cascaded Jacobian functions of the warping function
    # Refer to the supplementary materials for math documentation
    dx_dp = torch.cat((-xy,     1+x**2, -y, invd, O, -invd*x), dim=2)
    dy_dp = torch.cat((-1-y**2, xy,     x, O, invd, -invd*y), dim=2)

    fx, fy, cx, cy = torch.split(K, 1, dim=1)
    return dx_dp*fx.view(B,1,1), dy_dp*fy.view(B,1,1)


def compute_jacobian_dIdp(Jf_x, Jf_y, Jx_p, Jy_p):
    """ chained gradient of image w.r.t. the pose
    :param the Jacobian of the feature map in x direction  [B, C, H, W]
    :param the Jacobian of the feature map in y direction  [B, C, H, W]
    :param the Jacobian of the x map to manifold p  [B, HXW, 6]
    :param the Jacobian of the y map to manifold p  [B, HXW, 6]
    ------------
    :return the image jacobian in x, y, direction, Bx2x6 each
    """
    B, C, H, W = Jf_x.shape

    # precompute J_F_p, JtWJ
    Jf_p = Jf_x.view(B,C,-1,1) * Jx_p.view(B,1,-1,6) + \
        Jf_y.view(B,C,-1,1) * Jy_p.view(B,1,-1,6)
    
    return Jf_p.view(B,-1,6)