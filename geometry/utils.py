import functools
import inspect
from typing import List
import torch
import numpy as np


def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
        if they are numpy arrays. Use the device and dtype of the wrapper.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device('cpu')
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)
    
        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)
    
    return wrap


class TensorWrapper:
    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]
    
    @property
    def device(self):
        return self._data.device
    
    @property
    def dtype(self):
        return self._data.dtype
    
    def __getitem__(self, index):
        return self.__class__(self._data[index])
    
    def __setitem__(self, index, item):
        self._data[index] = item.data
    
    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))
    
    def cpu(self):
        return self.__class__(self._data.cpu())
    
    def cuda(self):
        return self.__class__(self._data.cuda())
    
    def pin_memory(self):
        return self.__class__(self._data.pin_memory())
    
    def float(self):
        return self.__class__(self._data.float())
    
    def double(self):
        return self.__class__(self._data.double())
    
    def detach(self):
        return self.__class__(self._data.detach())
    
    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return cls.stack(*args, **kwargs)
        else:
            return NotImplemented


def so3exp_map(w, eps: float = 1e-7) -> torch.Tensor:
    """Compute rotation matrices from batched twists.

    Args:
        w: batched 3D axis-angle vectors of shape (..., 3).
        eps: . Defaults to 1e-7.

    Returns:
        A batch of rotation matrices of shape (..., 3, 3).
    """
    theta = w.norm(p=2, dim=-1, keepdim=True)
    small = theta < eps
    div = torch.where(small, torch.ones_like(theta), theta)
    W = skew_symmetric(w / div)
    theta = theta[..., None] # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res) # first-order Taylor approx
    return torch.eye(3).to(W) + res
    

def skew_symmetric(v: torch.Tensor):
    '''Create a skew-symmetric matrix from a 
       (batched) vector of size (..., 3).
    '''
    o = torch.zeros_like(v[..., 0])
    x, y, z = v.unbind(dim=-1)
    M = torch.stack([ o,  -z,  y,
                      z,   o, -x,
                     -y,   x,  o], dim=-1).reshape(v.shape[:-1]+(3, 3))
    return M


def coord_grid(H: int, W: int, **kwarg):
    """ Create grid for pixel coordinates """
    y, x = torch.meshgrid(
        torch.arange(H).to(**kwarg).float(),
        torch.arange(W).to(**kwarg).float(),
        indexing='ij'
    )
    return torch.stack([x, y], dim=-1)


def generate_xy_grid(B, H, W, K):
    """ Generate a batch a image grid from image space to world space
        px = (u - cx) / fx
        py = (v - cy) / fy

    Args:
        B: batch size
        H: height
        W: width
        K: camera intrinsic array [fx, fy, cx, cy]
    
    Returns:
        px: u-coordinate as grid of shape (B, 1, H, W)
        py: v-coordinate as grid of shape (B, 1, H, W)
    """
    fx, fy, cx, cy = K.split(1, dim=1)
    u, v = coord_grid(H, W).unbind(dim=-1)
    if B > 1:
        u = u.view(1, 1, H, W).repeat(B, 1, 1, 1)
        v = v.view(1, 1, H, W).repeat(B, 1, 1, 1)
    px = ((u.view(B, -1) - cx) / fx).view(B, 1, H, W)
    py = ((u.view(B, -1) - cy) / fy).view(B, 1, H, W)
    return px, py