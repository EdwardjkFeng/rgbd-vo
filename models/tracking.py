import os.path as osp
import torch
import torch.nn as nn
from lietorch import SE3
import numpy as np
import cv2

from models.submodules import color_normalize  # todo

from models.algorithm.dia import DirectImageAlign
from models.algorithm.inv_dia import InvDirectImageAlign

from models.pyramid import ImagePyramids, FeaturePyramid
# from models.algorithms import DirectSolverNet
from models.deep_estimator import DeepRobustEstimator
# from models.segmentation import Segmentor  # Provide segmentation model

from utils import visualize
from utils.timers import NullTimer
from utils.logger import check_directory


class LeastSquareTracking(nn.Module):
    # all enum types
    NONE                 = -1
    RGB                  =  0
    CONV_RGBD            =  1

    def __init__(
        self,
        encoder_name="RGB",
        uncertainty_type="identity",
        pyramid_level=4,
        max_iter_per_pyr=10,
        mEst_type=None,
        solver_type=None,
        tr_samples=10,
        direction="inverse",
        options=None,
        vis_res=False,
        no_weight_sharing=False,
        init_pose_type=None,
        feature_channel=None,
        uncertainty_channel=None,
        remove_tru_sigma=None,
        feature_extract=None,
        combine_ICP=None,
        scaler=None,
        scale_scaler=None,
        timers=None,
    ) -> None:
        super().__init__()
        self.scales = [0, 1, 2, 3]
        self.construct_image_pyramids = ImagePyramids(self.scales, pool='avg')
        self.construct_depth_pyramids = ImagePyramids(self.scales, pool='max')

        self.timers = timers or NullTimer()
        self.direction = direction

        # self.mEst_type = None
        # self.vis_res = True
        
        """ =============================================================== """
        """              use option to transfer parameters                  """
        """ =============================================================== """
        # used in forward fuction
        self.vis_feat_uncer = options.vis_feat
        self.train_uncer_prop = options.train_uncer_prop
        
        # feature & uncertainty pyramid
        feature_extract = feature_extract or options.feature_extract
        self.uncertrainty_type = uncertainty_type
        feature_channel = feature_channel or options.feature_channel
        uncertainty_channel = uncertainty_channel or options.uncertainty_channel

        # scaling function
        scale_type = scaler or options.scaler

        # optimization solving function
        remove_tru_sigma = remove_tru_sigma or options.remove_tru_sigma

        # pose initialization function
        init_pose_type = init_pose_type or options.init_pose
        self.predict_init_pose = False if init_pose_type == "identity" else True
        self.train_init_pose = options.train_init_pose
        init_pose_scale = options.scale_init_pose
        init_pose_multi_hypo = options.multi_hypo # TODO experiment
        res_input_for_init_pose = options.res_input # feed residual into posenet
        self.checkpoint = options.checkpoint # chosse a checkpoint model to test

        """ =============================================================== """
        """             Initialize the Deep Feature Extractor               """
        """ =============================================================== """
        if encoder_name == "RGB":
            print("The network will use raw image for direct image alignment.")
            self.encoder = None
            self.encoder_type = self.RGB
            context_dim = 1
        elif encoder_name == "ConvRGBD": # original 
            print("The network extract features from RGB, using feature map for direct image aligment.")
            context_dim = 4
            self.encoder = FeaturePyramid(
                C=context_dim,
                feature_channel=feature_channel,
                feature_extractor=feature_extract,
                uncertainty_channel=uncertainty_channel,
                uncertainty_type=uncertainty_type,
            )
            self.encoder_type = self.CONV_RGBD
        else:
            raise NotImplementedError()
        
        """ =============================================================== """
        """             Initialize the Robuset Estimator                    """
        """ =============================================================== """
        # TODO
        mEst_type = "None"
        self.mEst_func = DeepRobustEstimator(mEst_type)
        mEst_funcs = [self.mEst_func, self.mEst_func, self.mEst_func, self.mEst_func]

        """ =============================================================== """
        """             Initialize the Trust-Region Damping                 """
        """ =============================================================== """
        # TODO
        self.solver_func = None
        solver_funcs = []

        """ =============================================================== """
        """             Initialize the Trust-Region Method                  """
        """ =============================================================== """
        self.track_type = None
        self.trackers = []
        self.pyramid_level = pyramid_level

        if self.direction == "forward":
            self.track_type = "Forward_DIA"
            for p in range(pyramid_level):
                tracker_l = DirectImageAlign(
                    num_iterations = max_iter_per_pyr, 
                    mEst_func= mEst_funcs[p],
                    timers = timers,
                )
                self.trackers.append(tracker_l)

        elif self.direction == "inverse":
            self.track_type = "Inverse_DIA"      
            for p in range(pyramid_level):
                tracker_l = InvDirectImageAlign(
                    num_iterations = max_iter_per_pyr, 
                    mEst_func= mEst_funcs[p],
                    timers = timers,
                )
                self.trackers.append(tracker_l)
        
        print("{:=^80}".format(f" Using {self.track_type} "))

        """ =============================================================== """
        """             Initialize Pose Predictor Network                   """
        """ =============================================================== """
        # TODO
        if self.predict_init_pose:
            if self.train_init_pose:
                print("=> Add predicted pose to the output pose (joint train the init pose)")
            print("=> # init pose hypothesis:", init_pose_multi_hypo)
            if init_pose_type == 'sfm_net':
                self.pose_predictor = SFMPoseNet(scale_motion=init_pose_scale,
                                                 multi_hypo=init_pose_multi_hypo,
                                                 res_input=res_input_for_init_pose)
            elif init_pose_type == 'dense_net':
                self.pose_predictor = PoseNet(scale_motion=init_pose_scale,
                                              multi_hypo=init_pose_multi_hypo,
                                              res_input=res_input_for_init_pose)
            else:
                raise NotImplementedError('unsupported pose predictor network')
            

    def forward(self, img0, img1, depth0, depth1, intrinsics, pose=None, logger=None, iteration=0, vis=False, obj_mask0=None, obj_mask1=None, index=None):
        preprocessed_data = self.__preprocess(img0, img1, depth0, depth1, pose_init=None, obj_mask0=obj_mask0, obj_mask1=obj_mask1)

        (I0, I1, F0, F1, sigma0, sigma1, dpt0_pyr, dpt1_pyr, invD0, invD1, obj_mask0_pyr, obj_mask1_pyr, pose_init) = preprocessed_data

        poses_to_train = [[], []] # '[translation, rotation] caution different from original
        sigma_xi = []

        # initial pose prediction
        if self.predict_init_pose and self.train_init_pose:
            [t0, R0] = pose_init
            poses_to_train[0].append(R0)
            poses_to_train[1].append(t0)
            if self.train_uncer_prop:
                B = invD0[0].shape[0] # TODO make sure dim is correct
                sigma_xi.append(torch.eye(6).view(1, 6, 6).type_as(R0).repeat(B, 1, 1))
            
        # the prior of the mask
        prior_W = torch.ones(invD0[3].shape).type_as(invD0[3]) * 0.001

        self.timers.tic("trust-region update")

        # # Coarse-to-fine trust-region update
        for p in range(self.pyramid_level-1, -1, -1):
            intrinsics_l = intrinsics / (1 << p)
            if self.track_type in ["Forward_DIA", "Inverse_DIA"]:
                output = self.trackers[p](pose_init, F0[p], F1[p], invD0[p],invD1[p], intrinsics_l, dpt0_pyr[p], dpt1_pyr[p])
            else:
                raise NotImplementedError()
            
            transform_matrix = output.matrix()
            poses_to_train[0].append(transform_matrix[:, :3, 2:3]) # translation
            poses_to_train[1].append(transform_matrix[:, :3, :3]) # rotation
            pose_init = output

            if self.train_uncer_prop:
                pass # TODO

        out_pose = output.inv().matrix()
        output = (out_pose[:, :3, :3], out_pose[:, :3, 3])
        
        # # Trust-region update on level 3
        # intrinsics3 = intrinsics / (1 << 3)
        # if self.track_type == "Forward_DIA":
        #     output3 = self.trackers[3](pose_init, F0[3], F1[3], invD0[3], invD1[3], intrinsics3)
        # else:
        #     raise NotImplementedError()
        
        # pose3 = output3
        # pose3_matrix = pose3.matrix()
        # poses_to_train[0].append(pose3_matrix[:, :3, 2:3]) # translation
        # poses_to_train[1].append(pose3_matrix[:, :3, :3]) # rotation matrix

        # if self.train_uncer_prop:
        #     pass # TODO

        # # Trust-region update on level 2
        # intrinsics2 = intrinsics / (1 << 2)
        # if self.track_type == "Forward_DIA":
        #     output2 = self.trackers[2](pose3, F0[2], F1[2], invD0[2], invD1[2], intrinsics2)
        # else:
        #     raise NotImplementedError()
        
        # pose2 = output2
        # pose2_matrix = pose2.matrix()
        # poses_to_train[0].append(pose2_matrix[:, :3, 2:3]) # translation
        # poses_to_train[1].append(pose2_matrix[:, :3, :3]) # rotation matrix

        # if self.train_uncer_prop:
        #     pass # TODO

        # # Trust-region update on level 1
        # intrinsics1 = intrinsics / (1 << 1)
        # if self.track_type == "Forward_DIA":
        #     output1 = self.trackers[1](pose2, F0[1], F1[1], invD0[1], invD1[1], intrinsics1)
        # else:
        #     raise NotImplementedError()
        
        # pose1 = output1
        # pose1_matrix = pose1.matrix()
        # poses_to_train[0].append(pose1_matrix[:, :3, 2:3]) # translation
        # poses_to_train[1].append(pose1_matrix[:, :3, :3]) # rotation matrix

        # if self.train_uncer_prop:
        #     pass # TODO
        
        # # Trust-region update on level 0
        # if self.track_type == "Forward_DIA":
        #     output0 = self.trackers[0](pose1, F0[0], F1[0], invD0[0], invD1[0], intrinsics)
        # else:
        #     raise NotImplementedError()
        
        # pose0 = output0.inv()
        # pose0_matrix = pose0.matrix()
        # poses_to_train[0].append(pose0_matrix[:, :3, 2:3]) # translation
        # poses_to_train[1].append(pose0_matrix[:, :3, :3]) # rotation matrix

        # output = (pose0_matrix[:, :3, :3], pose0_matrix[:, :3, 3])

        # if self.train_uncer_prop:
        #     pass # TODO
        
        self.timers.toc("trust-region update")

        with torch.no_grad():
            # visualization
            pass

        if self.training:
            pyr_t = torch.stack(tuple(poses_to_train[0]), dim=1)
            pyr_R = torch.stack(tuple(poses_to_train[1]), dim=1)
            if self.train_uncer_prop:
                pass
            else:
                return pyr_R, pyr_t
        else:
            return output
        

    def __preprocess(self, img0, img1, depth0, depth1, pose_init=None, obj_mask0=None, obj_mask1=None):
        self.timers.tic("extract features")
        # pre-processing all the data, all the invalid inputs depth are set to 0
        # invD0 = torch.clamp(1.0 / depth0, 0, 10)
        # invD1 = torch.clamp(1.0 / depth1, 0, 10)
        invD0 = torch.where(depth0 > 0, 1.0/depth0, depth0)
        invD1 = torch.where(depth1 > 0, 1.0/depth1, depth1)
        # invD0[invD0 == invD0.min()] = 0
        # invD1[invD1 == invD1.min()] = 0
        # invD0[invD0 == invD0.max()] = 0
        # invD1[invD1 == invD1.max()] = 0


        # I0 = color_normalize(img0)
        # I1 = color_normalize(img1)
        I0 = img0
        I1 = img1

        x0, sigma0, orig_x0 = self.__encode_features(I0, invD0, I1, invD1)
        x1, sigma1, orig_x1 = self.__encode_features(I1, invD1, I0, invD0)
        inv_d0 = self.construct_depth_pyramids(invD0)
        inv_d1 = self.construct_depth_pyramids(invD1)


        dpt0_pyr = self.construct_depth_pyramids(depth0)
        dpt1_pyr = self.construct_depth_pyramids(depth1)
        # dpt0_pyr = [None] * len(self.scales)
        # dpt1_pyr = [None] * len(self.scales)

        if obj_mask0 is not None:
            obj_mask0_pyr = self.construct_image_pyramids(obj_mask0)
        else:
            obj_mask0_pyr = [None] * len(self.scales)
        if obj_mask1 is not None:
            obj_mask1_pyr = self.construct_image_pyramids(obj_mask1)
        else:
            obj_mask1_pyr = [None] * len(self.scales)

        self.timers.toc("extract features")


        # init pose
        if pose_init is None:
            if self.predict_init_pose:
                t0, R0 = self.pose_predictor(orig_x0[3], orig_x1[3])
                tq = None # TODO
            else:
                B = I0.shape[0]
                tq = torch.zeros(B, 7, dtype=torch.float).to(I0)
            pose_init = SE3.exp(tq)
        
        return (I0, I1, x0, x1, sigma0, sigma1, dpt0_pyr, dpt1_pyr, inv_d0, inv_d1, obj_mask0_pyr, obj_mask1_pyr, pose_init)
    

    def __encode_features(self, img0, invD0, img1, invD1):
        """ Get encoded features. """
        if self.encoder_type == self.RGB:
            I = self.__color3to1(img0)
            x = self.construct_image_pyramids(I)
            sigma = [torch.ones_like(a) for a in x]
            origin_x = x
        elif self.encoder_type == self.CONV_RGBD:
            m = torch.cat((img0, invD0), dim=1)
            x, sigma, origin_x = self.encoder.forward(m)
        else:
            raise NotImplementedError()
        
        return x, sigma, origin_x
    
    def __color3to1(self, img):
        """ Return a gray-scale image. """
        B, _, H, W = img.shape
        return (img[:,0] * 0.114 + img[:, 1] * 0.587 + img[:, 2] * 0.299).view(B,1,H,W)