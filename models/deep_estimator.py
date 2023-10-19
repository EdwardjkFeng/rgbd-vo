"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from models.submodules import convLayer as conv, initialize_weights


class DeepRobustEstimator(nn.Module):
    """The Deep M-Estimator"""
    def __init__(
        self,
        estimator_type: str = 'Residual2w',
        uncertainty_type: str = 'laplacian',
        in_channel: int = 1,
        absolute_residual: bool = False,
    ):
        super().__init__()

        if estimator_type == 'Residual2w':
            assert in_channel == 1

        self.estimator_type = estimator_type
        self.uncertainty_type = uncertainty_type
        self.C = in_channel
        self.absolute_residual = absolute_residual

        if estimator_type == 'None':
            self.net = self.__constant_weight
        elif estimator_type == 'Huber':
            self.net = self.__weight_Huber
        elif estimator_type in ['Residual2w', 'MultiScale2w']:
            self.net = nn.Sequential(
                conv(True, self.C, 16, 3, dilation=1),
                conv(True, 16, 32, 3, dilation=2),
                conv(True, 32, 64, 3, dilation=4),
                conv(True, 64, 1,  3, dilation=1),
            )
            initialize_weights(self.net)
        else:
            raise NotImplementedError()
        
    def forward(self, residual, F0=None, F1=None, ws=None):
        if self.estimator_type == 'Residual2w':
            context = residual.abs() if self.absolute_residual else residual
            w = self.net(context)
        elif self.estimator_type == 'MultiScale2w':
            B, C, H, W = residual.shape
            wl = func.interpolate(
                ws, (H, W), mode='bilinear', align_corners=True
            )

            if self.absolute_residual:
                context = torch.cat((residual.abs(), F0, F1, wl), dim=1)  
            else:
                context = torch.cat((residual, F0, F1, wl), dim=1)
            w = self.net(context)

            if self.uncertainty_type == 'sigmoid':
                w = torch.sigmoid(self.net(context))
            elif self.uncertainty_type == 'laplacian':
                w = torch.exp(torch.clamp(w, min=-3, max=3))
        elif self.estimator_type == 'Huber' or self.estimator_type == 'None':
            w = self.net(residual)
        
        return w

    def __weight_Huber(self, x, alpha=1.345):
        """ weight function of Huber loss:
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
        """ mimic the standard least-square when weighting function is constant
        """
        return torch.ones_like(x)