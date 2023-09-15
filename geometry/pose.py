""" SE3 pose """

from typing import Union, Tuple, List, Dict, NamedTuple
import torch
import numpy as np

from .utils import autocast, TensorWrapper, so3exp_map, skew_symmetric


class Pose(TensorWrapper):
    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        """Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        """
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)
    
    @classmethod
    @autocast
    def from_axis_angle_t(cls, axis_angle: torch.Tensor, t: torch.Tensor):
        """Pose from an axis-angle rotation vector and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            axis_angle: axis-angle rotation vector with shape (..., 3).
            t: translation vector with shape (..., 3).
        """
        assert axis_angle.shape[-1] == 3
        assert t.shape[-1] == 3
        assert axis_angle[:-1] == t.shape[:-1]
        return cls.from_Rt(so3exp_map(axis_angle), t)
    
    @classmethod
    def from_matrix(cls, T: torch.Tensor):
        """Pose from an SE(3) transformation matrix.

        Args:
            T: transformation matrix with shape (..., 4, 4).
        """
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)
    
    @classmethod
    def from_colmap(cls, image: NamedTuple):
        """Pose from a COLMAP Image."""
        return cls.from_Rt(image.qvec2rotmat(), image.tvec())
    
    @property
    def R(self) -> torch.Tensor:
        '''Underlying rotation matrix with shape (..., 3, 3).'''
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1]+(3, 3))
    
    @property
    def t(self) -> torch.Tensor:
        '''Underlying translation vector with shape (..., 3).'''
        return self._data[..., -3:]
    

    def inv(self) -> 'Pose':
        '''Invert an SE(3) pose.'''
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)
    
    def compose(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C.compose(T_A2B) -> T_A2C.'''
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)
    
    @autocast
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points.

        Args:
            p3d: 3D points, numpy.ndarray or torch.Tensor with shape (..., 3)

        Returns:
            transformed 3D points, torch.Tensor with shape (..., 3).
        """
        assert p3d.shape[-1] == 3
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)
    
    def __mul__(self, p3d: torch.Tensor) -> torch.Tensor:
        '''Transform a set of 3D points: T_A2B * p3d_A -> p3d_B.'''
        return self.transform(p3d)
    
    def __matmul__(self, other: 'Pose') -> 'Pose':
        '''Chain two SE(3) poses: T_B2C @ T_A2B -> T_A2C.'''
        return self.compose(other)
    
    @autocast
    def J_transform(self, p3d_out: torch.Tensor):
        J_t = torch.diag_embed(torch.ones_like(p3d_out))
        J_rot = -skew_symmetric(p3d_out)
        J = torch.cat([J_t, J_rot], dim=-1)
        return J # N x 3 x 6
    
    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()
    
    def magnitude(self) -> Tuple[torch.Tensor]:
        """Magnitude of the SE(3) transformation.

        Returns:
            dr: rotation angle in degrees.
            dt: translation distance in meters.
        """
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1) / 2, -1, 1)
        dr = torch.acos(cos).abs() / torch.pi * 180
        dt = torch.norm(self.t, dim=-1)
        return dr, dt
    
    def __repr__(self) -> str:
        return f'Pose: {self.shape} {self.dtype} {self.device} \n[[{self._data[0]}, {self._data[1]}, {self._data[2]}, {self._data[-3]}],\n [{self._data[3]}, {self._data[4]}, {self._data[5]}, {self._data[-2]}],\n [{self._data[6]}, {self._data[7]}, {self._data[8]}, {self._data[-1]}],]'