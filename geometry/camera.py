"""Pinhole camera"""

from typing import Union, Tuple, Dict, NamedTuple
import torch
import numpy as np

from .utils import autocast, TensorWrapper


class Camera(TensorWrapper):
    eps = 1e-3

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] in (6, 8, 10)
        super().__init__(data)

    @classmethod
    def create_camera(cls, camera: Union[Dict, NamedTuple]):
        '''Camera from dictionary (or COLMAP tuple).
        The origin (0, 0) is the center of the top-left pixel.
        This is different from COLMAP.
        '''
        if isinstance(camera, tuple):
            camera = camera._asdict()
        
        model = camera['model']
        params = camera['params']

        if model in ['OPENCV', 'PINHOLE']:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif model in ['SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL']:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if model == 'SIMPLE_RADIAL':
                params = np.r_[params, 0.]
        else:
            raise NotImplementedError(model)
        
        data = np.r_[camera['width'], camera['height'],
                     fx, fy, cx, cy, params] # TODO why -0.5?
        
        return cls(data)
        
    @property
    def size(self) -> torch.Tensor:
        '''Size (width height) of the images, with shape (..., 2).'''
        return self._data[..., :2]
    
    @property
    def f(self) -> torch.Tensor:
        '''Focal lengths (fx, fy) with shape (..., 2).'''
        return self._data[..., 2:4]
    
    @property
    def c(self) -> torch.Tensor:
        '''Principal points (cx, cy) with shape (.., 2).'''
        return self._data[..., 4:6]
    
    @property
    def p(self) -> torch.Tensor:
        '''Distortion parameters, with shape (..., {0, 2, 4}).'''
        if self._data.shape[-1] > 6:
            return self._data[..., 6:]
        else:
            print('No distortion coefficients available in the model.')
            return None
    
    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        '''Update the camera parameters after resizing an image.'''
        if isinstance(scales, (int, float)):
            scales = (scales, scales)
        s = self._data.new_tensor(scales)
        # About the strange 0.5 offset https://github.com/JakobEngel/dso?search=1
        data = torch.cat([self.size * s, 
                          self.f * s,
                          (self.c + 0.5) * s - 0.5,
                          self.p], -1)
        return self.__class__(data)
    
    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        '''Update the camera parameters after cropping an image.'''
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)
        data = torch.cat([size,
                          self.f,
                          self.c - left_top,
                          self.p], -1)
        return self.__class__(data)
    
    @autocast
    def in_image(self, p2d: torch.Tensor):
        '''Check if 2D points are within the image boundaries.'''
        assert p2d.shape[-1] == 2
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1), -1))
        return valid
    
    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Project 3D points into the camera plane and check for visibility.'''
        z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)
        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        return p2d, valid
    
    def J_porject(self, p3d: torch.Tensor):
        x, y, z = p3d.unbind(dim=-1)
        zero = torch.zeros_like(z)
        z = z.clamp(min=self.eps)
        inv_z = 1/z
        J = torch.stack([
            inv_z,  zero, -x*inv_z*inv_z,
            zero,  inv_z, -y*inv_z*inv_z
        ], dim=-1)
        J = J.reshape(p3d.shape[:-1] + (2, 3))
        return J # N x 2 x 3

    ###########################################################################
    # TODO currently not implemented
    ###########################################################################
    @autocast
    def undistort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Undistort normalized 2D coordinates and check for validity of
           the distortion model
        '''
        assert pts.shape[-1] == 2
        return self.undistort_points(pts, self.p)
    

    def undistort_points(self, pts, dist_coeff):
        """Undistort normalized 2D coordinates and check for validity of
           the distortion model

        Args:
            pts: normaalized 2D coordinates
            dist_coeff: distortion coefficients
        """
        pass

    def J_undistort(self, pts: torch.Tensor):
        return self.J_undistort_points(pts, self.p) # N x 2 x 2
    
    def J_undistort_points(self, pts, dist_coeff):
        pass

    ###########################################################################
    ###########################################################################

    @autocast
    def denoramlize(self, p2d: torch.Tensor) -> torch.Tensor:
        '''Convert normalized 2D coordinates into pixel coordinates.'''
        return p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)
    
    def J_denormalize(self):
        return torch.diag_embed(self.f).unsqueeze(-3) # 1 x 2 x 2
    
    @autocast
    def world_to_image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        '''Transform 3D points into 2D pixel coordinates.'''
        p2d, visiable = self.project(p3d)
        
        if not self.p is None:
            p2d, mask = self.undistort(p2d)
        
        p2d = self.denoramlize(p2d)
        valid = visiable & mask & self.in_image(p2d) \
            if not self.p is None else valid & self.in_image(p2d)
        
        return p2d, valid
    
    def J_world_to_image(self, p3d: torch.Tensor):
        p2d_distorted, valid = self.project(p3d)
        if not self.p is None:
            J = self.J_denormalize() @ self.J_undistort(p2d_distorted)
        else: 
            J = self.J_denormalize()
        
        J = J @ self.J_project(p3d)
        return J, valid
    
    def __repr__(self) -> str:
        return f'Camera: {self.shape} {self.dtype} {self.device} {self._data}'