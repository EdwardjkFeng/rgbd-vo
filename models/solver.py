import torch
import torch.nn as nn

from submodules import fcLayer
from algorithms import *


class DirectSolverNet(nn.Module):

    # the enum types for direct solver
    SOLVER_NO_DAMPING       = 0
    SOLVER_RESIDUAL_VOLUME  = 1

    def __init__(self, solver_type, samples=10, direction='inverse'):
        super(DirectSolverNet, self).__init__()
        self.direction = direction  # 'inverse' or 'forward'
        if solver_type == 'Direct-Nodamping':
            self.net = None
            self.type = self.SOLVER_NO_DAMPING
        elif solver_type == 'Direct-ResVol':
            # flattened JtJ and JtR (number of samples, currently fixed at 10)
            self.samples = samples
            self.net = deep_damping_regressor(D=6*6+6*samples)
            self.type = self.SOLVER_RESIDUAL_VOLUME
            initialize_weights(self.net)
        else: 
            raise NotImplementedError()

    def forward(self, JtJ, Jt, weights, R, pose0, invD0, invD1, x0, x1, K, obj_mask1=None):
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

        wR = (weights * R).view(B, -1, 1)   # [B, CXHXW, 1]
        JtR = torch.bmm(Jt, wR)  # [B, 6, 1]

        if self.type == self.SOLVER_NO_DAMPING:
            # Add a small diagonal damping. Without it, the training becomes quite unstable
            # Do not see a clear difference by removing the damping in inference though
            Hessian = LM_H(JtJ)
        elif self.type == self.SOLVER_RESIDUAL_VOLUME:
            Hessian = self.__regularize_residual_volume(JtJ, Jt, JtR, weights,
                pose0, invD0, invD1, x0, x1, K, sample_range=self.samples, obj_mask1=obj_mask1)
        else:
            raise NotImplementedError()

        if self.direction == 'forward':
            updated_pose = forward_update_pose(Hessian, JtR, pose0)
        elif self.direction == 'inverse':
            updated_pose = inverse_update_pose(Hessian, JtR, pose0)
        else:
            raise NotImplementedError('pose updated should be either forward or inverse')
        return updated_pose

    def __regularize_residual_volume(self, JtJ, Jt, JtR, weights, pose,
        invD0, invD1, x0, x1, K, sample_range, obj_mask1=None):
        """ regularize the approximate with residual volume

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

        diag_mask = torch.eye(6).view(1,6,6).type_as(JtJ)
        diagJtJ = diag_mask * JtJ
        traceJtJ = torch.sum(diagJtJ, (2,1))
        epsilon = (traceJtJ * 1e-6).view(B,1,1) * diag_mask
        n = sample_range
        lambdas = torch.logspace(-5, 5, n).type_as(JtJ)

        for s in range(n):
            # the epsilon is to prevent the matrix to be too ill-conditioned
            D = lambdas[s] * diagJtJ + epsilon
            Hessian = JtJ + D
            pose_s = inverse_update_pose(Hessian, JtR, pose)

            res_s,_= compute_warped_residual(pose_s, invD0, invD1, x0, x1, px, py, K, obj_mask1=obj_mask1)
            JtR_s = torch.bmm(Jt, (weights * res_s).view(B,-1,1))
            JtR_volumes.append(JtR_s)

        JtR_flat = torch.cat(tuple(JtR_volumes), dim=2).view(B,-1)
        JtJ_flat = JtJ.view(B,-1)
        damp_est = self.net(torch.cat((JtR_flat, JtJ_flat), dim=1))
        R = diag_mask * damp_est.view(B,6,1) + epsilon # also lift-up

        return JtJ + R
    
def deep_damping_regressor(D):
    """ Output a damping vector at each dimension
    """
    net = nn.Sequential(
        fcLayer(in_planes=D,   out_planes=128, bias=True),
        fcLayer(in_planes=128, out_planes=256, bias=True),
        fcLayer(in_planes=256, out_planes=6, bias=True)
    ) # the last ReLU makes sure every predicted value is positive
    return net