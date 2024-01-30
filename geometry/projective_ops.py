import torch
import torch.nn.functional as f

from lietorch import SE3
from geometry import geometry

MIN_DEPTH = 0.1

def coord_grid(H: int, W: int, B=None, **kwarg):
    """Create grid for pixel coordinates"""
    y, x = torch.meshgrid(
        torch.arange(H).to(**kwarg).float(),
        torch.arange(W).to(**kwarg).float(),
        indexing="ij",
    )
    
    if B is not None:
        x = x.view(1, 1, H, W).contiguous().expand(B, -1, -1, -1)
        y = y.view(1, 1, H, W).contiguous().expand(B, -1, -1, -1)
    return x, y


def generate_xy_grid(B, H, W, intrinsics):
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
    fx, fy, cx, cy = intrinsics.split(1, dim=1)
    u, v = coord_grid(H, W, B, device=intrinsics.device)

    px = ((u.view(B, -1) - cx) / fx).view(B, 1, H, W)
    py = ((v.view(B, -1) - cy) / fy).view(B, 1, H, W)

    return px, py


def batch_inv_proj(intrinsics, invD, jacobian=False):
    """ Pinhole camera inverse porjection. 
    
    Return:
        pts: homogeneous point cloud coord with inverse depth [B, H, W, 4]"""
    B, _, H, W = invD.shape

    px, py = generate_xy_grid(B, H, W, intrinsics)

    i = torch.ones_like(invD)
    # pts = torch.cat([px, py, i, invD], dim=1)#.permute(0, 2, 3, 1) # [B, H, W, 4]
    # pts = pts * torch.where(invD > 0, 1.0/invD, 0.0)
    pts = torch.cat([px * invD, py * invD, invD, i], dim=1)
    pts = pts.permute(0, 2, 3, 1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[:, -1, ...] = 1.0 # [B, H, W, 4]
        return pts, J

    return pts, None


def bacth_transform(P1: torch.Tensor, T21: SE3, jacobian=False):
    """Transform point coordinates with T from 1->2.

    Args:
        P1: point cloud coordinate as xyz tensor [B, 4, H, W]
        T21: transformation
        Jacobian: whether to return Jacobian. Defaults to False.

    Return:
        P2: point cloud coordinate in j coordinate system [B, 4, H, W]
        Jt: Jacobian of Transformation w.r.t. twist
    """
    B, H, W, _ = P1.shape # [B, H, W, 4]
    # print(P1.shape, T21[:, None, None].shape)
    P2 = T21[:, None, None].act(P1)

    if jacobian:
        # print(P2.shape)
        X, Y, Z, d = P2.unbind(dim=-1)
        o = torch.zeros_like(d)

        Jt = torch.stack([
            d,  o,  o,  o,  Z, -Y,
            o,  d,  o, -Z,  o,  X, 
            o,  o,  d,  Y, -X,  o,
            o,  o,  o,  o,  o,  o,
        ], dim=-1).view(B, H, W, 4, 6)

        return P2, Jt
    
    return P2, None
        

def batch_proj(P, intrinsics, jacobian=False, return_depth=False):
    """Pinhole camera projection.
    
    Return:
        p: pixel coordinate
    """
    fx, fy, cx, cy = intrinsics[..., None].split(1, dim=1)
    X, Y, Z, D = P.unbind(dim=-1) # [B, H, W]

    # Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)
    # d = 1.0 / Z
    # d = torch.where(Z > MIN_DEPTH, torch.ones_like(Z), 1.0 / Z)
    d = 1.0 / Z

    # print(X.shape, d.shape, fx.shape)
    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        # coords = torch.stack([x, y, D*d], dim=1)
        coords = torch.stack([x, y, Z/D], dim=1)
    else:
        coords = torch.stack([x, y], dim=1)

    if jacobian:
        B, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
        ], dim=-1).view(B, H, W, 2, 4)

        return coords, proj_jac

    return coords, None


def Batch_projective_transform(pose, invD1, intrinsics, jacobian=False, return_depth=False):
    """ Map points from 1->0. """

    # inverse project (pinhole)
    X1, Jz = batch_inv_proj(intrinsics, invD1, jacobian)

    # transform
    X0, Jt = bacth_transform(X1, pose, jacobian)

    # project
    x0, Jp = batch_proj(X0, intrinsics, jacobian, return_depth)

    # exclude points too close to camera 
    valid = ((X0[..., 2] > MIN_DEPTH) & (X1[..., 2] > MIN_DEPTH))
    
    if jacobian:
        # J = torch.matmul(Jp, Jt)
        return x0, valid, Jp, Jt
    
    return x0, valid


# ICP
def compute_ICP_residuals_Jacobian(vertex0, vertex1, normal0, normal1, T21, intrinsics):
    B, C, H, W = vertex0.shape

    vertex0 = vertex0.permute(0, 2, 3, 1)
    vertex0_to1 = T21[:, None, None].act(vertex0)
    
    fx, fy, cx, cy = intrinsics.split(1, dim=1)
    # print(vertex0_to1.shape)
    x_, y_, s_ = vertex0_to1.split(1, dim=-1)
    u_ = (x_ / s_).view(B, -1) * fx + cx
    v_ = (y_ / s_).view(B, -1) * fy + cy

    w_vertex1 = geometry.warp_features(vertex1, u_, v_)
    w_normal1 = geometry.warp_features(normal1, u_, v_)
    diff = vertex0_to1.permute(0, 3, 1, 2) - w_vertex1

    res = (w_normal1.view(B, 3, H*W)) * diff.view(B, 3, H*W)
    res = res.sum(dim=1, keepdim=True).view(B, -1)

    R = T21.matrix()[:, :3, :3]
    NtC10 = torch.bmm(w_normal1.view(B,3,-1).permute(0,2,1), R)  # [B, H*W, 3]
    J_rot = torch.bmm(NtC10.view(-1,3).unsqueeze(dim=1),  #[B*H*W,1,3]
                        geometry.batch_skew(vertex0.view(-1, 3))).squeeze()  # [B*H*W, 3]
    J_trs = -NtC10.view(-1,3)  # [B*H*W, 3]

    # compose jacobians
    J_F_p = torch.cat((J_rot, J_trs), dim=-1)  # follow the order of [rot, trs]  [B*H*W, 6]
    J_F_p = J_F_p.view(B, -1, 6)  # [B, 1, HXW, 6]

    return res, J_F_p
