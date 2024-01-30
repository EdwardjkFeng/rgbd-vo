""" 
Scripts to run keyframe visual odometry on a sequence of images
"""

# import standard library
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import argparse
import os.path as osp

# import third party
import cv2
from cv2 import imread
import numpy as np
import open3d as o3d
from copy import deepcopy
import torch

import config
from geometry.geometry import batch_create_transform, batch_mat2Rt
from geometry import geometry
from utils.select_method import select_method
from utils.run_utils import check_cuda
from dataset.dataloader import load_data
from utils.logger import check_directory

from utils.o3d_rgbd_vo import RGBDOdometry
from utils.o3d_icp_vo import ICP_Odometry
from utils import visualize
from utils.tools import save_trajectory

from kornia.morphology import dilation
from dataset.cofusion import CoFusion

from models.segmentation import MaskRCNN, SamPrompter
from torchvision.transforms.functional import resize


R_y_180 = np.eye(4, dtype=float)
R_y_180[0, 0] = R_y_180[2, 2] = -1.0

RED = np.array([1.0, 0, 0]).reshape(1, 3)
GREEN = np.array([0, 1.0, 0]).reshape(1, 3)
BLUE = np.array([0, 0, 1.0]).reshape(1, 3)


class MyScene:
    def __init__(self):
        self.dataloader = None
        self.network = None
        self.index = 0
        self.video_id = None
        self.last_pose = None
        self.last_gt_pose = None
        self.camera_transfrom = None
        self.is_gt_tracking = False
        self.init = False
        self.is_play = True
        self.vo_type = None
        self.two_view = False
        self.options = None

        self.est_poses = []  # list of timestamp, estimated poses pair

    def reset(self):
        self.init = True


def camera_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ R_y_180


def run_vo(scene: "MyScene"):
    if not scene.is_play:
        return

    dataset = scene.dataloader
    options = scene.options

    for i in range(len(dataset)):
        scene.index = i

        if scene.vo_type == "incremental":
            batch = dataset[scene.index]
        else:
            batch = dataset.get_keypair(scene.index)
        color0, color1, depth0, depth1, GT_Rt, intrins = check_cuda(batch["data"])
        name = batch["name"]

        # masks0, masks1 ,object_indices0, object_indices1, object_transform = check_cuda(batch["object_info"])


        # if scene.filter_moving_objects:
        #     # color0 = CoFusion.filter_moving_objects(color0, masks0)
        #     depth0 = CoFusion.filter_moving_objects(depth0, masks0)
        #     # color1 = CoFusion.filter_moving_objects(color1, masks1)
        #     depth1 = CoFusion.filter_moving_objects(depth1, masks1)

        # if scene.index > 1:
        #     return

        scene_id = name["seq"]

        # Reset scene for new scene.
        if scene_id != scene.video_id:
            scene.reset()
            # scene.init_idx = scene.index
            scene.video_id = scene_id
        else:
            scene.init = False

        # print(len(dataset.cam_pose_seq[0]), len(dataset))
        GT_WC = dataset.cam_pose_seq[0][scene.index]  # ground truth camera pose
        depth_file = dataset.depth_seq[0][scene.index]

        print
        if not options.save_img:
            # half resolution
            rgb = color1.permute((1, 2, 0)).cpu().numpy()

            # depth = (
            #     imread(depth_file, cv2.IMREAD_ANYDEPTH).astype(np.float32)
            #     / scene.scale_factor
            # )
            # depth = cv2.resize(
            #     depth,
            #     None,
            #     fx=dataset.fx_s,
            #     fy=dataset.fy_s,
            #     interpolation=cv2.INTER_NEAREST,
            # )
            depth = depth1.squeeze().cpu().numpy()
            valid_depth = (depth > 0.5) & (depth < 5.0)
            depth = np.where(valid_depth, depth, 0.0)
            # print(depth.shape)
            # depth = depth1.squeeze().cpu().numpy()
            K = {
                "fx": intrins[0].item(),
                "fy": intrins[1].item(),
                "ux": intrins[2].item(),
                "uy": intrins[3].item(),
            }
        else:
            # original resolution for demo
            rgb = imread(dataset.image_seq[0][scene.index])
            depth = imread(depth_file).astype(np.float32) / 5e3
            calib = np.asarray(dataset.calib[0], dtype=np.float32)
            K = {"fx": calib[0], "fy": calib[1], "ux": calib[2], "uy": calib[3]}

            # save input rgb and depth images
            img_index_png = str(scene.index).zfill(5) + ".png"
            if options.dataset == "VaryLighting":
                output_folder = osp.join(dataset.seq_names[0], "kf_vo", options.vo)
            else:
                output_folder = os.path.join("/home/binbin/Pictures", "kf_vo", options.vo)

            rgb_img = osp.join(output_folder, "rgb", img_index_png)
            depth_img = osp.join(output_folder, "depth", img_index_png)
            check_directory(rgb_img)
            check_directory(depth_img)
            cv2.imwrite(rgb_img, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(depth_img, depth)

        ########################################################################
        ###                              Tracking                            ###
        ########################################################################
        if scene.init:
            if GT_WC is not None:
                T_WC = GT_WC
            else:
                T_WC = np.array(
                    [
                        [0.0, -1.0, 0.0, -0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                # T_WC = np.eye(4)
            scene.T_WC = T_WC
            scene.est_poses.append((dataset.timestamp[0][scene.index], scene.T_WC))
            if scene.vo_type == "keyframe":
                scene.T_WK = T_WC
        else:
            if GT_WC is not None:
                pose_gt = batch_mat2Rt(GT_Rt)
            with torch.no_grad():
                color0 = color0.unsqueeze(dim=0)
                color1 = color1.unsqueeze(dim=0)
                depth0 = depth0.unsqueeze(dim=0)
                depth1 = depth1.unsqueeze(dim=0)
                intrins = intrins.unsqueeze(dim=0)

                if options.save_img:
                    output = scene.network.forward(
                        color0, color1, depth0, depth1, intrins, index=scene.index
                    )
                else:
                    output = scene.network.forward(color0, color1, depth0, depth1, intrins, pose=None)

            R, t = output
            if scene.is_gt_tracking:
                T_WC = GT_WC
                scene.T_WC = T_WC

                if scene.last_GT_WC is not None:
                    gt_rel_pose = np.dot(np.linalg.inv(GT_WC), scene.last_GT_WC).astype(
                        np.float32
                    )
            else:
                if scene.vo_type == "incremental":
                    T_CR = batch_create_transform(t, R)
                    # T_CR = GT_Rt
                    T_CR = T_CR.squeeze(dim=0).cpu().numpy()
                    T_WC = np.dot(scene.T_WC, np.linalg.inv(T_CR)).astype(np.float32)
                    # T_WC = np.dot(scene.T_WC, T_CR)
                elif scene.vo_type == "keyframe":
                    T_CK = batch_create_transform(t, R)
                    # T_CK = GT_Rt
                    T_CK = T_CK.squeeze(dim=0).cpu().numpy()
                    T_WC = np.dot(scene.T_WK, np.linalg.inv(T_CK)).astype(np.float32)

                    # print large drift in keyframe tracking,
                    # just for noticing a possible tracking failure, not usd later
                    T_CC = np.dot(np.linalg.inv(T_WC), scene.T_WC).astype(np.float32)
                    trs_drift = np.copy(T_CC[0:3, 3:4]).transpose()
                    if np.linalg.norm(trs_drift) > 0.02:
                        print(depth_file)
                else:
                    raise NotImplementedError()
                scene.T_WC = T_WC
                scene.est_poses.append((dataset.timestamp[0][scene.index], scene.T_WC))

            scene.last_GT_WC = np.copy(GT_WC)
            gt_trs = np.copy(GT_WC[0:3, 3:4]).transpose()
            trs = np.copy(T_WC[0:3, 3:4]).transpose()

            if scene.last_pose is not None:
                lines = np.array([[0, 1]])
                # Visualize ground truth trajectory
                gt_poses_seg = np.concatenate((scene.last_gt_pose, gt_trs), axis=0)

            scene.last_pose = trs
            scene.last_gt_pose = gt_trs

        if dataset.realscene:
            if options.save_img:
                if scene.vo_type == "keyframe":
                    T_see = np.array(
                        [
                            [1.000, 0.000, 0.000, 0.2],
                            [0.000, 0.866, 0.500, -0.7],
                            [0.000, -0.500, 0.866, -0.8],
                            [0.000, 0.000, 0.000, 1.0],
                        ]
                    )

                    scene.camera_transform = camera_transform(np.matmul(scene.T_WK, T_see))

            else:
                # adjust which transformation use to set the see pose
                if scene.vo_type == "keyframe":
                    T_see = np.array(
                        [
                            [1.000, 0.000, 0.000, 0.2],
                            [0.000, 0.866, 0.500, -0.7],
                            [0.000, -0.500, 0.866, -0.8],
                            [0.000, 0.000, 0.000, 1.0],
                        ]
                    )

                    scene.camera_transform = camera_transform(np.matmul(scene.T_WK, T_see))
        else:
            # scene.camera.transform = T_WC @ tf.translation_matrix([0, 0, 2.5])
            scene.camera_transform = T_WC @ np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 2.5],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

        # if scene.index == scene.init_idx + 1:
        #     input()
        # print(scene.index)
        sys.stdout.write("Tracking %s: %d frames  \r" % (scene.video_id, scene.index))
        sys.stdout.flush()
        scene.index += 1  # scene.track_config['frame_step']
        # print("<=================================")
        if options.save_img:
            return


def main(options):
    # keyframes = [int(x) for x in options.keyframes.split(',')]
    # if options.dataset in ['BundleFusion', 'TUM_RGBD']:
    #     obj_has_mask = False
    # else:
    #     obj_has_mask = True

    # eval_loaders = create_eval_loaders(options, options.eval_set,
    #                                    [1,], total_batch_size, options.trajectory)

    scene = MyScene()
    # tracker = select_method(options.vo, options)
    tracker = RGBDOdometry("RGBD")
    # tracker = ICP_Odometry("ColorICP")
    # tracker = ICP_Odometry("Point2Point") # gradslam
    scene.network = tracker

    scene.index = 0  # config['start_frame']  # starting frame e.g. 60
    scene.video_id = None
    scene.last_pose = None
    scene.last_gt_pose = None
    scene.last_GT_WC = None
    # scene.maskrcnn = MaskRCNN(target_labels=[1])
    # scene.samprompter = SamPrompter()
    # scene.generate_mask = False
    # scene.filter_moving_objects = options.filter_moving_objects
    scene.compute_res = False
    scene.is_gt_tracking = options.gt_tracker
    scene.init = False  # True only for the first frame
    scene.is_play = True  # immediately start playing when called
    scene.vo_type = options.vo_type
    scene.two_view = options.two_view
    scene.options = options

    # Load data
    if options.dataset == "TUM_RGBD":
        # sequence = "rgbd_dataset_freiburg1_desk"
        # sequence = "rgbd_dataset_freiburg1_xyz"
        # sequence = "rgbd_dataset_freiburg3_walking_rpy"
        from dataset.tum_rgbd import tum_sequences_dict
        sequences = []
        for ks, conf in tum_sequences_dict().items():
            seq_names = conf['seq']
            sequences += seq_names
    elif options.dataset == "Bonn_RGBD":
        # sequences = ["rgbd_bonn_balloon_tracking"]
        # sequence = "rgbd_bonn_person_tracking2"
        # sequences = ["rgbd_bonn_synchronous"]
        # sequence = "rgbd_bonn_placing_nonobstructing_box"
        # sequence = "rgbd_bonn_static"
        from dataset.bonn_rgbd import bonn_sequences_dict
        sequences = []
        for ks, conf in bonn_sequences_dict().items():
            seq_names = conf['seq']
            sequences += seq_names
    elif options.dataset == "CoFusion":
        sequence = "room4-full"

    for sequence in sequences:
        print(sequence)
        try:
            np_loader = load_data(options.dataset, keyframes=[1], load_type='full', select_trajectory=sequence, truncate_depth=True, options=options)

            # scene.visualizer = vis
            scene.dataloader = np_loader
            scene.dataloader.realscene = True

            scene.scale_factor = (
                5e3 if scene.options.dataset in ["TUM_RGBD", "Bonn_RGBD"] else 1
            )
            run_vo(scene)
            # scene.network.timers.print()
            save_trajectory(options.dataset + "/" + sequence, scene.est_poses, "rgbdvo.txt")
        except:
            print("Fail:", sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the network")
    config.add_basics_config(parser)
    config.add_test_basics_config(parser)
    config.add_tracking_config(parser)
    config.add_vo_config(parser)

    options = parser.parse_args()
    # options.filter_moving_objects = True
    # to save visualization: --save_img and --vis_feat
    print("---------------------------------------")
    main(options)
