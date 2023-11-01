"""
Data loader fot TUM RBGD benchmark
"""


import sys, os

sys.path.append("..")
import os.path as osp
import random
import pickle
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import logging
import torch.utils.data as data
from transforms3d import quaternions

import cv2
from tqdm import tqdm

from .base_dataset import BaseDataset
from .utils import *
from .dataset_utils import *
from dataset import logger
from utils.tools import print_conf


""" 
The following scripts use the directory structure as:

root
└── rgbd_dataset_freiburg1_desk
    ├── accelerometer.txt
    ├── depth
    ├── depth.txt
    ├── groundtruth.txt
    ├── rgb
    └── rgb.txt
"""


def tum_sequences_dict():
    """the sequence dictionary of TUM dataset
    https://vision.in.tum.de/data/datasets/rgbd-dataset/download

    The calibration parameters refers to:
    https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    """
    return {
        "fr1": {
            "calib": [525.0, 525.0, 319.5, 239.5],
            "seq": [
                "rgbd_dataset_freiburg1_360",
                "rgbd_dataset_freiburg1_desk",
                "rgbd_dataset_freiburg1_desk2",
                "rgbd_dataset_freiburg1_floor",
                "rgbd_dataset_freiburg1_room",
                "rgbd_dataset_freiburg1_xyz",
                "rgbd_dataset_freiburg1_rpy",
                "rgbd_dataset_freiburg1_plant",
                "rgbd_dataset_freiburg1_teddy",
            ],
        },
        "fr2": {
            "calib": [525.0, 525.0, 319.5, 239.5],
            "seq": [
                "rgbd_dataset_freiburg2_desk",
                "rgbd_dataset_freiburg2_360_hemisphere",
                "rgbd_dataset_freiburg2_large_no_loop",
                "rgbd_dataset_freiburg2_large_with_loop",
                "rgbd_dataset_freiburg2_pioneer_360",
                "rgbd_dataset_freiburg2_pioneer_slam",
                "rgbd_dataset_freiburg2_pioneer_slam2",
                "rgbd_dataset_freiburg2_pioneer_slam3",
                "rgbd_dataset_freiburg2_xyz",
                "rgbd_dataset_freiburg2_rpy",
                "rgbd_dataset_freiburg2_coke",
                "rgbd_dataset_freiburg2_dishes",
                "rgbd_dataset_freiburg2_flowerbouquet_brownbackground",
                "rgbd_dataset_freiburg2_metallic_sphere2",
                "rgbd_dataset_freiburg2_flowerbouquet",
                "rgbd_dataset_freiburg2_360_kidnap",
                "rgbd_dataset_freiburg2_desk_with_person",
            ],
        },
        "fr3": {
            "calib": [525.0, 525.0, 319.5, 239.5],
            "seq": [
                "rgbd_dataset_freiburg3_cabinet",
                "rgbd_dataset_freiburg3_nostructure_notexture_far",
                "rgbd_dataset_freiburg3_nostructure_notexture_near_withloop",
                "rgbd_dataset_freiburg3_nostructure_texture_far",
                "rgbd_dataset_freiburg3_nostructure_texture_near_withloop",
                "rgbd_dataset_freiburg3_structure_notexture_near",
                "rgbd_dataset_freiburg3_structure_texture_far",
                "rgbd_dataset_freiburg3_structure_texture_near",
                "rgbd_dataset_freiburg3_teddy",
                "rgbd_dataset_freiburg3_walking_halfsphere",
                "rgbd_dataset_freiburg3_walking_rpy",
                "rgbd_dataset_freiburg3_sitting_rpy",
                "rgbd_dataset_freiburg3_sitting_static",
                "rgbd_dataset_freiburg3_sitting_xyz",
                "rgbd_dataset_freiburg3_walking_static",  # dynamic scene
                "rgbd_dataset_freiburg3_walking_xyz",  # dynamic scene
                "rgbd_dataset_freiburg3_long_office_household",
            ],
        },
        "default": {
            "calib": [525.0, 525.0, 319.5, 239.5],
            "seq": [
                "None",  # anything not list here
            ],
        },
    }


class TUM(BaseDataset):
    default_conf = {
        "name": "TUM",
        'num_workers': 8,
        'train_batch_size': 1,
        'val_batch_size': 1,
        'test_batch_size': 1,
        "dataset_dir": "TUM_RGBD_Dataset/",
        "select_traj": "rgbd_dataset_freiburg1_desk",
        "category": "test",
        "keyframes": [1],
        "truncate_depth": True,
        "grayscale": False,
        "resize": 0.25,
        "add_val_dataset": False,
    }

    def _init(self, conf):
        print_conf(self.conf)

    def get_dataset(self):
        return _Dataset(self.conf)

# DATA_PATH = '/home/jingkun/Dataset/'

class _Dataset(data.Dataset):
    def __init__(self, conf):
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

        if self.conf.category == "test":
            self._load_test()
        elif self.conf.category in ["train", "validation"]:
            self._load_train_val(
                self.root,
                select_traj=self.conf.select_traj,
                add_val_dataset=self.conf.add_val_dataset,
            )
        else:
            raise NotImplementedError()

        self.truncate_depth = self.conf.truncate_depth

        logger.info(
            f"TUM dataloader for {self.conf.category} using keyframe {self.keyframes}: {self.ids} valid frames."
        )

    def _load_test(self):
        tum_data = tum_sequences_dict()
        assert len(self.keyframes) == 1
        kf = self.conf.keyframes[0]
        self.keyframes = [1]

        for ks, scene in tum_data.items():
            for seq_name in scene["seq"]:
                seq_path = seq_name

                if self.conf.select_traj is not None:
                    if seq_path != self.conf.select_traj:
                        continue

                self.calib.append(scene["calib"])

                # load or generate synchronized trajectory file
                datacache_root = osp.join(osp.dirname(__file__), 'cache/tum_rgbd')
                sync_traj_file = osp.join(datacache_root, seq_path, "sync_trajectory.pkl")
                if not osp.isfile(sync_traj_file):
                    logger.info(
                        f"Synchronized trajectory file {sync_traj_file} has not been generated."
                    )
                    logging.info("Generate it now ...")
                    write_sync_trajectory(self.root, seq_name, dataset='tum_rgbd')

                with open(sync_traj_file, "rb") as f:
                    frames = pickle.load(f)
                    total_num = len(frames)

                    images = [frames[idx][1] for idx in range(0, total_num, kf)]
                    depths = [frames[idx][2] for idx in range(0, total_num, kf)]
                    poses = [  # extrinsic
                        tq2mat(frames[idx][0]) for idx in range(0, total_num, kf)
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
                self.calib.append(tum_sequences_dict()["default"]["calib"])
                sync_traj_file = osp.join(self.root, self.conf.select_traj)
                seq_name = seq_path = osp.basename(self.conf.select_traj)
                if not osp.isfile(sync_traj_file):
                    logger.info(
                        "The synchronized trajctory file {:} has not been generated.".format(
                            seq_path
                        )
                    )
                    logger.info("Generate it now ...")
                    write_sync_trajectory(self.root, self.conf.select_traj)

                with open(sync_traj_file, "rb") as p:
                    frames = pickle.load(p)
                    total_num = len(frames)

                    timestamp = [
                        osp.splitext(osp.basename(image))[0] for image in images
                    ]
                    images = [frames[idx][1] for idx in range(total_num)]
                    depths = [frames[idx][2] for idx in range(total_num)]
                    poses = [  # extrinsic
                        tq2mat(frames[idx][0] for idx in range(0, total_num, kf))
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

        color0, scale = self._load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1, _ = self._load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        depth0 = self._load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self._load_depth_tensor(self.depth_seq[seq_idx][next_idx])

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

    def _load_rgb_tensor(self, path):
        """Load the rgb image."""
        image = read_image(path, self.conf.grayscale) / 255.0
        image, scale = resize(image, self.conf.resize, interp="linear")
        image = np.transpose(image, (2, 0, 1))  # channel first convention
        return image, scale

    def _load_depth_tensor(self, path):
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



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str)
    args = parser.parse_args()

    if args.conf:
        conf = OmegaConf.load(args.conf)

    loader = TUM(conf.data).get_dataset()
    import torchvision.utils as torch_utils

    torch_loader = data.DataLoader(loader, batch_size=16, shuffle=False, num_workers=4)

    for batch in torch_loader:
        item = batch
        color0, color1, depth0, depth1, transform, calib = item["data"]
        B, C, H, W = color0.shape

        bcolor0_img = torch_utils.make_grid(color0, nrow=4)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(bcolor0_img.numpy().transpose(1, 2, 0))
        plt.show()
