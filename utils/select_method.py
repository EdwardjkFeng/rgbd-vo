"""
A wrapper to select different methods for comparison
"""

import torch
from models.LeastSquareTracking import LeastSquareTracking

from utils.timers import Timers


def select_method(method_name, options):
    assert method_name in ['RGB', 'RGB+ICP']
    if method_name == 'RGB':
        print('==>Load RGB method')
        rgb_tracker = LeastSquareTracking(
            encoder_name='RGB',
            combine_ICP=False,
            feature_channel=1,
            uncertainty_channel=1,
            # feature_extract='conv',
            uncertainty_type='None',
            scaler='None',
            direction=options.direction,
            max_iter_per_pyr=options.max_iter_per_pyr,
            mEst_type='None',
            solver_type='Direct-Nodamping',
            init_pose_type='identity',
            options=options,
            # timers=Timers(),
        )
        if torch.cuda.is_available(): rgb_tracker.cuda()
        rgb_tracker.eval()
        return rgb_tracker
    
    if method_name == "RGB+ICP":
        print("==>Load RGB+ICP method")
        rgbd_tracker = LeastSquareTracking(
            encoder_name="RGB",
            combine_ICP=True,
            uncertainty_channel=1,
            uncertainty_type="None",
            scaler="None",
            direction="inverse",
            max_iter_per_pyr=options.max_iter_per_pyr,
            mEst_type="None",
            solver_type="Direct-Nodamping",
            init_pose_type='identity',
            remove_tru_sigma=False,
            scale_scaler=0.2,
            options=options,
        )
        if torch.cuda.is_available(): rgbd_tracker.cuda()
        rgbd_tracker.eval()
        return rgbd_tracker