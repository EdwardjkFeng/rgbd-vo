""" Coordinate transformation """

import torch
from lietorch import SE3


def cam1_to_cam2(cam1_pose: SE3, cam2_pose: SE3):
    r""" Compute transformation from cam1 to cam2 coordinate system
    :math:`T_cam2_cam1 = (T_w_cam2)^(-1) * T_w_cam1`


    Args:
        cam1_pose: transfromation from cam1 to world coordinate system, T_w_cam1
        cam2_pose: transformation from cam2 to world coordinate system, T_w_cam2
    """

    return cam2_pose.inv() * cam1_pose
