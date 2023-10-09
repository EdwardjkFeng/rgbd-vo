"""
A wrapper to select different methods for comparison
"""

import torch
from models.LeastSquareTracking import LeastSquareTracking


def select_method(method_name, options):
    assert method_name in ['RGB']
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
            direction='inverse',
            max_iter_per_pyr=options.max_iter_per_pyr,
            mEst_type='None',
            solver_type='Direct-Nodamping',
            init_pose_type='identity',
            options=options,
        )
        if torch.cuda.is_available(): rgb_tracker.cuda()
        rgb_tracker.eval()
        return rgb_tracker