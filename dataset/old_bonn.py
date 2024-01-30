"""Bonn dataset"""
import sys, os
import random
import pickle
from pathlib import Path
from omegaconf import OmegaConf
from typing import Optional, Union

import cv2
import numpy as np
import logging
from tqdm import tqdm

import torch.utils.data as data
from transforms3d import quaternions

from .base_dataset import BaseDataset
from .utils import *
from .dataset_utils import *
from dataset import logger
from utils.tools import print_conf


"""
The following scripts use the directory structure as:

root
└── rgbd_bonn_balloon
    ├── depth
    ├── depth.txt
    ├── groundtruth.txt
    ├── rgb
    └── rgb.txt
"""


def bonn_sequences_dict():
    """The sequence dictionary of Bonn RGBD dataset
    https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/

    The calibration parameters refer to:
    https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/
    """
    bonn_dict = bonn_trainval_dict()
    bonn_test = bonn_test_dict()

    for scene in bonn_dict.keys():
        bonn_dict[scene]["seq"] += bonn_test[scene]["seq"]

    return bonn_dict


def bonn_trainval_dict():
    return {
        "scene": {
            "calib": [542.822841, 542.576870, 315.593520, 237.756098],
            "seq": [
                "rgbd_bonn_crowd",
                "rgbd_bonn_kidnapping_box2",
                "rgbd_bonn_moving_nonobstructing_box",
                "rgbd_bonn_placing_obstructing_box",
                "rgbd_bonn_static",
                "rgbd_bonn_balloon",
                "rgbd_bonn_placing_nonobstructing_box3",
                "rgbd_bonn_moving_nonobstructing_box2",
                "rgbd_bonn_balloon_tracking",
                "rgbd_bonn_moving_obstructing_box",
                # "rgbd_bonn_person_tracking2",
                "rgbd_bonn_removing_nonobstructing_box",
                "rgbd_bonn_placing_nonobstructing_box2",
                "rgbd_bonn_synchronous",
                # "rgbd_bonn_crowd3",
                "rgbd_bonn_removing_obstructing_box",
                # "rgbd_bonn_placing_nonobstructing_box",
                # "rgbd_bonn_balloon2",
                "rgbd_bonn_crowd2",
                "rgbd_bonn_synchronous2",
                "rgbd_bonn_removing_nonobstructing_box2",
                "rgbd_bonn_static_close_far",
                "rgbd_bonn_moving_obstructing_box2",
                "rgbd_bonn_kidnapping_box",
                "rgbd_bonn_person_tracking",
                # "rgbd_bonn_balloon_tracking2",
            ],
        },
    }


def bonn_test_dict():
    return {
        "scene": {
            "calib": [542.822841, 542.576870, 315.593520, 237.756098],
            "seq": [
                "rgbd_bonn_person_tracking2",
                "rgbd_bonn_crowd3",
                "rgbd_bonn_placing_nonobstructing_box",
                "rgbd_bonn_balloon2",
                "rgbd_bonn_balloon_tracking2",
            ],
        },
        "default": {
            "calib": [542.822841, 542.576870, 315.593520, 237.756098],
            "seq": [
                "None",
            ],
        },
    }


class Bonn(BaseDataset):
    default_conf = {
        "name": "Bonn_RGBD",
        "num_workers": 8,
        "train_batch_size": 1,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "dataset_dir": "Bonn_RGBD_Dataset/",
        "select_traj": "rgbd_bonn_balloon",
        "category": "test",
        "keyframes": [1],
        "truncate_depth": True,
        "grayscale": False,
        "resize": 0.25,
    }

    def _init(self, conf):
        print_conf(self.conf)

    def get_dataset(self):
        return _Dataset(self.conf)


class _Dataset(data.Dataset):
    def __init__(self, conf) -> None:
        super().__init__()
        self.root = Path(conf.dataset_dir)
        self.conf = conf

        self.image_seq = []  # list(seq) or list(frame) of string (rbg image path)
        self.timestamp = []  # empty
        self.depth_seq = []  # list(seq) of list(frame) of string (depth image path)
        self.invalid_seq = []  # empty
        self.cam_pose_seq = []  # list(seq) of list(frame) of 4 x 4 ndarray
        self.calib = []  # list(seq) of list(intrinsics: fx, fy, cx, cy)
        self.seq_names = []  # list(seq) or string(seq name)

        self.ids = 0
        self.seq_acc_ids = [0]
        self.keyframes = self.conf.keyframes

        if self.conf.category in ["test", "full"]:
            self.__load_test()
        elif self.conf.category in ["train", "validation"]:
            self.__load_train_val()
        else:
            raise NotImplementedError()

        self.truncate_depth = self.conf.truncate_depth

        logger.info(
            f"{self.conf.name} dataloader for {self.conf.category} using keyframe {self.keyframes}: {self.ids} valid frames."
        )

    def __load_train_val(self):
        bonn_data = bonn_trainval_dict()
        for seq_name in bonn_data["scene"]["seq"]:
            seq_path = seq_name
            if self.conf.select_traj is not None:
                if seq_path != self.conf.select_traj:
                    continue

            self.calib.append(bonn_data["scene"]["calib"])
            datacache_root = osp.join(osp.dirname(__file__), f"cache/{self.conf.name}")
            sync_traj_file = osp.join(datacache_root, seq_path, "sync_trajectory.pkl")
            if not osp.isfile(sync_traj_file):
                logger.info(
                    f"Synchronized trajectory file {sync_traj_file} has not been generated."
                )
                logging.info("Generate it now ...")
                write_sync_trajectory(self.root, seq_name, dataset=self.conf.name, max_diff=0.1)

            with open(sync_traj_file, "rb") as f:
                trainval = pickle.load(f)
                total_num = len(trainval)
                train_ids, val_ids = self.__gen_trainval_index(total_num)
                if self.conf.category == "train":
                    ids = train_ids
                else:
                    ids = val_ids

                images = [trainval[idx][1] for idx in ids]
                depths = [trainval[idx][2] for idx in ids]
                poses = [self.__tq2mat(trainval[idx][0]) for idx in ids]  # extrinsic
                timestamp = [osp.splitext(osp.basename(image))[0] for image in images]

                self.timestamp.append(timestamp)
                self.image_seq.append(images)
                self.depth_seq.append(depths)
                self.cam_pose_seq.append(poses)
                self.seq_names.append(seq_path)
                self.ids += max(0, len(images) - max(self.keyframes))
                self.seq_acc_ids.append(self.ids)

    def __gen_trainval_index(self, seq_len, ratio=0.1):
        s = int((1 - ratio) // ratio)
        indices = np.arange(seq_len)
        print(seq_len)
        val_ids = indices[s :: (s + 1)]
        train_ids = np.setdiff1d(indices, val_ids)
        print(len(train_ids), len(val_ids))
        return train_ids, val_ids

    def __load_test(self):
        bonn_data = bonn_sequences_dict()
        assert len(self.keyframes) == 1
        kf = self.conf.keyframes[0]
        self.keyframes = [1]

        for ks, scene in bonn_data.items():
            for seq_name in scene["seq"]:
                seq_path = seq_name

                if self.conf.select_traj is not None:
                    if seq_path != self.conf.select_traj:
                        continue

                self.calib.append(scene["calib"])

                # load or generate synchronized trajectory file
                datacache_root = osp.join(
                    osp.dirname(__file__), f"cache/{self.conf.name}"
                )
                sync_traj_file = osp.join(
                    datacache_root, seq_path, "sync_trajectory.pkl"
                )
                if not osp.isfile(sync_traj_file):
                    logger.info(
                        f"Synchronized trajectory file {sync_traj_file} has not been generated."
                    )
                    logging.info("Generate it now ...")
                    write_sync_trajectory(self.root, seq_name, dataset=self.conf.name, max_diff=0.1)

                with open(sync_traj_file, "rb") as f:
                    frames = pickle.load(f)
                    total_num = len(frames)

                    images = [frames[idx][1] for idx in range(0, total_num, kf)]
                    depths = [frames[idx][2] for idx in range(0, total_num, kf)]
                    poses = [  # extrinsic
                        self.__tq2mat(frames[idx][0]) for idx in range(0, total_num, kf)
                    ]
                    timestamp = [
                        osp.splitext(osp.basename(image))[0] for image in images
                    ]

                    self.timestamp.append(timestamp)
                    self.image_seq.append(images)
                    self.depth_seq.append(depths)
                    self.cam_pose_seq.append(poses)
                    self.seq_names.append(seq_path)
                    self.ids += max(0, len(images) - 1)
                    self.seq_acc_ids.append(self.ids)

        if len(self.image_seq) == 0:
            logger.warn(
                "The specified trajectory {:} is not in the test set."
                "\nTry to support this customized dataset".format(self.conf.select_traj)
            )
            if osp.exists(osp.join(self.root, self.conf.select_traj)):
                self.calib.append(bonn_sequences_dict()["default"]["calib"])
                sync_traj_file = osp.join(self.root, self.conf.select_traj)
                seq_name = seq_path = osp.basename(self.conf.select_traj)
                if not osp.isfile(sync_traj_file):
                    logger.info(
                        "The synchronized trajctory file {:} has not been generated.".format(
                            seq_path
                        )
                    )
                    logger.info("Generate it now ...")
                    write_sync_trajectory(self.root, self.conf.select_traj, max_diff=0.1)

                with open(sync_traj_file, "rb") as p:
                    frames = pickle.load(p)
                    total_num = len(frames)

                    timestamp = [
                        osp.splitext(osp.basename(image))[0] for image in images
                    ]
                    images = [frames[idx][1] for idx in range(total_num)]
                    depths = [frames[idx][2] for idx in range(total_num)]
                    poses = [  # extrinsic
                        self.__tq2mat(frames[idx][0] for idx in range(0, total_num, kf))
                    ]

                    self.timestamp.append(timestamp)
                    self.image_seq.append(images)
                    self.depth_seq.append(depths)
                    self.cam_pose_seq.append(poses)
                    self.seq_names.append(seq_path)
                    self.ids += max(0, len(images) - 1)
                    self.seq_acc_ids.append(self.ids)
            else:
                raise Exception(
                    "The specified trajectory is not in the test set nor a supported customized dataset"
                )

    def __getitem__(self, idx):
        seq_idx = max(np.searchsorted(self.seq_acc_ids, idx + 1) - 1, 0)
        frame_idx = idx - self.seq_acc_ids[seq_idx]

        this_idx = frame_idx
        next_idx = frame_idx + random.choice(self.keyframes)

        color0, scale = self.__load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1, _ = self.__load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        depth0 = self.__load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self.__load_depth_tensor(self.depth_seq[seq_idx][next_idx])

        # normalize the coordinate
        calib = np.asarray(self.calib[seq_idx], dtype=np.float32)
        self.fx_s, self.fy_s = scale
        calib[0] *= self.fx_s
        calib[1] *= self.fy_s
        calib[2] *= self.fx_s
        calib[3] *= self.fy_s

        cam_pose0 = self.cam_pose_seq[seq_idx][this_idx]
        cam_pose1 = self.cam_pose_seq[seq_idx][this_idx]
        transform = np.dot(np.linalg.inv(cam_pose1), cam_pose0).astype(np.float32)

        name = {"seq": self.seq_names[seq_idx], "frame0": this_idx, "frame1": next_idx}

        # camera_info
        camera_info = {
            "height": color0.shape[0],
            "width": color0.shape[1],
            "fx": calib[0],
            "fy": calib[1],
            "ux": calib[2],
            "uy": calib[3],
        }

        data = {
            "name": name,
            "data": [color0, color1, depth0, depth1, transform, calib],
            "camera_info": camera_info,
        }
        return data

    def __len__(self):
        return self.ids

    def __load_rgb_tensor(self, path):
        """Load the rgb image."""
        image = read_image(path, self.conf.grayscale) / 255.0
        image, scale = resize(image, self.conf.resize, interp="linear")
        image = np.transpose(image, (2, 0, 1))  # channel first convention
        return image, scale

    def __load_depth_tensor(self, path):
        """Load depth:
        The depth images are scaled by a factor of 5000, i.e., a pixel value of 5000 in the depth image corresponds to a distance of 1 meter from the camera, 10000 to 2 meter distance, etc. A pixel value of 0 means missing value/no data.
        """
        # depth = read_image(path) / 5e3
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5e3
        depth, _ = resize(depth, self.conf.resize, interp="nearest")
        if self.conf.truncate_depth:
            # the accurate range of kinect depth
            valid_depth = (depth > 0.5) & (depth < 5.0)
            depth = np.where(valid_depth, depth, 0.0)
        return depth[None, :]  # channel first convention


    def __tq2mat(self, tq):
        """Transform translation-quaternion (tq) to (4x4) matrix. Adapted from
        https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/compute_global_transformation.py
        
        According to the documentation of the dataset, it's necessary to transform
        them to the same coordinate frame. To convert a model from the reference
        frame of the sensor to the one of the ground truth
        """
        tq = np.array(tq)
        T = np.eye(4)
        T[:3, :3] = quaternions.quat2mat(np.roll(tq[3:], 1))
        T[:3, 3] = tq[:3]

        T_ros = np.matrix(
            "-1 0 0 0;\
            0 0 1 0;\
            0 1 0 0;\
            0 0 0 1"
        )
        T_m = np.matrix(
            "1.0157    0.1828   -0.2389    0.0113;\
            0.0009   -0.8431   -0.6413   -0.0098;\
            -0.3009    0.6147   -0.8085    0.0111;\
            0         0         0    1.0000"
        )
        return T_ros * T * T_ros * T_m
