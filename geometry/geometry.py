import numpy as np
import torch
import torch.nn as nn


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
    theta = theta[..., None]  # ... x 1 x 1
    res = W * torch.sin(theta) + (W @ W) * (1 - torch.cos(theta))
    res = torch.where(small[..., None], W, res)  # first-order Taylor approx
    return torch.eye(3).to(W) + res


def skew_symmetric(v: torch.Tensor):
    """Create a skew-symmetric matrix from a
    (batched) vector of size (..., 3).
    """
    o = torch.zeros_like(v[..., 0])
    x, y, z = v.unbind(dim=-1)
    M = torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(
        v.shape[:-1] + (3, 3)
    )
    return M


def coord_grid(H: int, W: int, **kwarg):
    """Create grid for pixel coordinates"""
    y, x = torch.meshgrid(
        torch.arange(H).to(**kwarg).float(),
        torch.arange(W).to(**kwarg).float(),
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1)


def generate_xy_grid(B, H, W, K):
    """Generate a batch a image grid from image space to world space
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
    u, v = coord_grid(H, W, device=K.device).unbind(dim=-1)
    if B > 1:
        u = u.view(1, 1, H, W).repeat(B, 1, 1, 1)
        v = v.view(1, 1, H, W).repeat(B, 1, 1, 1)
    px = ((u.view(B, -1) - cx) / fx).view(B, 1, H, W)
    py = ((u.view(B, -1) - cy) / fy).view(B, 1, H, W)
    return px, py


def batch_twist2Mat(twist):
    """The exponential map from so3 to SO3

        Calculate the rotation matrix using Rodrigues' Rotation Formula
        http://electroncastle.com/wp/?p=39 
        or Ethan Eade's lie group note:
        http://ethaneade.com/lie.pdf equation (13)-(15)

    Args:
        twist: twist/axis angle [B, 3] \in \so3 space 

    Returns:
        Rotation matrix Bx3x3 \in \SO3 space
    """
    B = twist.size()[0]
    theta = twist.norm(p=2, dim=1).view(B, 1)
    w_so3 = twist / theta.expand(B, 3)
    W = batch_skew(w_so3)
    return torch.eye(3).repeat(B,1,1).type_as(W) \
        + W*torch.sin(theta.view(B,1,1)) \
        + W.bmm(W)*(1-torch.cos(theta).view(B,1,1))


def batch_Rt_compose(d_R, d_t, R0, t0):
    """Compose operator of R, t: [d_R*R | d_R*t + d_t] 
        We use left-mulitplication rule here. 

        function tested in 'test_geometry.py'

    Args:
        d_R: rotation incremental [B, 3, 3]
        d_t: translation incremental [B, 3]
        R0: initial rotation [B, 3, 3]
        t0: initial translation [B, 3]
    """
    R1 = torch.bmm(d_R, R0)
    t1 = d_R.bmm(t0.view(-1, 3, 1)) + d_t.view(-1, 3, 1)
    return R1, t1.view(-1, 3)

def batch_skew(w):
    """Generate a batch of skew-symmetric matrices

    Args:
        w: vector [B, 3]
    """

    B, D = w.size()
    assert D == 3
    o = torch.zeros(B).to(w)
    w0, w1, w2 = w.unbind(dim=-1)
    w_hat = torch.stack((o, -w2, w1, w2, o, -w0, -w1, w0, o), dim=1).view(B, 3, 3)
    return w_hat


def batch_warp_inverse_depth(px, py, invD0, pose10, K):
    """Compute the warping grid w.r.t. the SE3 transform given the inverse depth

    Args:
        px: the x coordinate map
        py: the y coordinate map
        invD0: the inverse depth in frame 0
        pose10: the 3D transform in SE3
        K: the intrinsics

    Returns:
        u: projected u coordinate in image space [B, 1, H, W]
        v: projected v coordinate in image space [B, 1, H, W]
        inv_z: projected inverse depth [B, 1, H, W]
    """

    [R, t] = pose10
    B, _, H, W = px.shape

    I = torch.ones((B, 1, H, W)).type_as(invD0)
    u_hom = torch.cat((px, py, I), dim=1)

    warped = torch.bmm(R, u_hom.view(B, 3, H * W)) + t.view(B, 3, 1).expand(B, 3, H * W) * invD0.view(B, 1, H * W).expand(B, 3, H * W)

    x_, y_, s_ = torch.split(warped, 1, dim=1)
    fx, fy, cx, cy = torch.split(K, 1, dim=1)

    u_ = (x_ / s_).view(B, -1) * fx + cx
    v_ = (y_ / s_).view(B, -1) * fy + cy

    inv_z_ = invD0 / s_.view(B, 1, H, W)

    return u_.view(B, 1, H, W), v_.view(B, 1, H, W), inv_z_


def check_occ(inv_z_buffer, inv_z_ref, crd, thres=1e-1, depth_valid=None):
    """z-buffering check of occlusion

    Args:
        inv_z_buffer: inverse depth of target frame
        inv_z_ref: inverse depth of reference frame
        crd: coordinate map
        thres: thrseshold. Defaults to 1e-1.
        depth_valid: valid depth mask. Defaults to None.

    Returns:
        occluded mask
    """

    B, _, H, W = inv_z_buffer.shape
    u, v = crd.split(1, dim=1)
    inv_z_warped = warp_features(inv_z_ref, u, v)

    inlier = inv_z_buffer > inv_z_warped - thres
    inviews = inlier & (u > 0) & (u < W - 1) & (v > 0) & (v < H - 1)
    if depth_valid is not None:
        inviews = inviews & depth_valid

    return inviews.logical_not()


def warp_features(F, u, v):
    """Warp the feature map (F) w.r.t. the grid (u, v)"""

    B, _, H, W = F.shape

    u_norm = u / ((W - 1) / 2) - 1
    v_norm = v / ((H - 1) / 2) - 1
    uv_grid = torch.cat((u_norm.view(B, H, W, 1), v_norm.view(B, H, W, 1)), dim=3)
    F_warped = nn.functional.grid_sample(
        F, uv_grid, align_corners=True, mode="bilinear", padding_mode="border"
    )
    return F_warped


def batch_transform_xyz(xyz_tensor, R, t, get_Jacobian=True):
    """Transform the point cloud w.r.t. the transformation matrix

    Args:
        xyz_tensor: point cloud coordinate as xyz tensor [B, 3, H, W]
        R: rotation matrix [B, 3, 3]
        t: translation vector [B, 3]
        get_Jacobian: whether to return the Jacobian. Defaults to True.
    """

    B, C, H, W = xyz_tensor.shape
    t_tensor = t.contiguous().view(B, 3, 1).repeat(1, 1, H * W)
    p_tesnor = xyz_tensor.contiguous().view(B, C, H * W)
    xyz_t_tensor = torch.baddbmm(t_tensor, R, p_tesnor)

    if get_Jacobian:
        # return both the transformed tensor and its Jacobian matrix
        rotated_tensor = (
            torch.bmm(R, p_tesnor).permute(0, 2, 1).contiguous().view(-1, 3)
        )
        J_R = batch_skew(rotated_tensor)  # [BxHxW, 3, 3]
        J_t = -1 * torch.eye(3).view(1, 3, 3).expand(B * H * W, 3, 3).to(J_R)
        J = torch.cat((J_t, J_R), dim=-1)  # [BxHxW, 3, 6]
        return xyz_t_tensor.view(B, C, H, W), J
    else:
        return xyz_t_tensor.view(B, C, H, W)


def batch_inverse_project(depth, K):
    """Inverse project pixels (u, v) to a point cloud given intrinsic

    Args:
        depth: depth [B, H, W]
        K: calibration is torch array composed of [fx, fy, cx, cy]

    Returns:
        xyz tensor (batch of point cloud) [B, 3, H, W]
    """

    if depth.dim() == 3:
        B, H, W = depth.size()
    else:
        B, _, H, W = depth.size()

    x, y = generate_xy_grid(B, H, W, K)
    z = depth.view(B, 1, H, W)
    return torch.cat((x * z, y * z, z), dim=1)


def batch_create_transform(trs, rot):
    if isinstance(trs, np.ndarray):
        top = np.hstack((rot, trs.reshape(3, 1)))
        bot = np.hstack((np.zeros([1, 3]), np.ones([1, 1])))
        out = np.vstack((top, bot))

    elif torch.is_tensor(trs):
        B = trs.shape[0] # B x 3
        top = torch.cat((rot, torch.unsqueeze(trs, 2)), dim=2) # [B, 3, 4]
        bot = torch.cat((torch.zeros(B, 1, 3), torch.ones(B, 1, 1)), 2).to(top)
        out = torch.cat((top, bot), dim=1)
    return out


def batch_mat2Rt(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    if T.ndim == 3:
        return R, t
    else:
        return R[None], t[None]