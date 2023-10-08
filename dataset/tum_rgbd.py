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
from .setting import DATA_PATH
from .utils import *
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
        "dataset_dir": "TUM/",
        "select_traj": "rgbd_dataset_freiburg1_desk",
        "category": "test",
        "keyframes": [1],
        "truncate_depth": True,
        "grayscale": False,
        "resize": None,
        "add_val_dataset": False,
    }

    def _init(self, conf):
        pass

    def get_dataset(self):
        return _Dataset(self.conf)


class _Dataset(data.Dataset):
    def __init__(self, conf):
        super().__init__()
        self.root = Path(DATA_PATH, conf.dataset_dir)
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
                datacache_root = osp.join(osp.dirname(__file__), 'cache')
                sync_traj_file = osp.join(datacache_root, seq_path, "sync_trajectory.pkl")
                if not osp.isfile(sync_traj_file):
                    logger.info(
                        f"Synchronized trajectory file {sync_traj_file} has not been generated."
                    )
                    logging.info("Generate it now ...")
                    write_sync_trajectory(self.root, seq_name)

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


""" =============================================================== """
"""      Some utilities to work with TUM RGB-D data                 """
""" =============================================================== """


def tq2mat(tq):
    """Transform translation-quaternion (tq) to (4x4) matrix."""
    tq = np.array(tq)
    T = np.eye(4)
    T[:3, :3] = quaternions.quat2mat(np.roll(tq[3:], 1))
    T[:3, 3] = tq[:3]
    return T


def write_sync_trajectory(local_dir, subject_name):
    """Generate synchronized trajectories.

    Args:
        local_dir: the root of the directory
        subject_name:
    """

    rgb_file = osp.join(local_dir, subject_name, "rgb.txt")
    depth_file = osp.join(local_dir, subject_name, "depth.txt")
    pose_file = osp.join(local_dir, subject_name, "groundtruth.txt")

    rgb_list = read_file_list(rgb_file)
    depth_list = read_file_list(depth_file)
    pose_list = read_file_list(pose_file)

    matches = associate_three(
        rgb_list, depth_list, pose_list, offset=0.0, max_difference=0.02
    )

    trajectory_info = []
    for a, b, c in matches:
        pose = [float(x) for x in pose_list[c]]
        rgb_file = osp.join(local_dir, subject_name, rgb_list[a][0])
        depth_file = osp.join(local_dir, subject_name, depth_list[b][0])
        trajectory_info.append([pose, rgb_file, depth_file])

    datacache_root = osp.join(osp.dirname(__file__), 'cache')
    dataset_path = osp.join(datacache_root, subject_name, 'sync_trajectory.pkl')

    if not osp.isdir(osp.join(datacache_root, subject_name)):
        os.makedirs(osp.join(datacache_root, subject_name))

    with open(dataset_path, "wb") as output:
        pickle.dump(trajectory_info, output)

    txt_path = osp.join(datacache_root, subject_name, "sync_trajectory.txt")
    pickle2txt(dataset_path, txt_path)


def pickle2txt(pickle_file, txt_file):
    """Write the pickle_file into a txt_file."""
    with open(pickle_file, "rb") as pkl_file:
        traj = pickle.load(pkl_file)

    with open(txt_file, "w") as f:
        for frame in traj:
            f.write(" ".join(["%f " % x for x in frame[0]]))
            f.write(frame[1] + " ")
            f.write(frame[2] + "\n")


"""
The following utility files are provided by TUM RGBD dataset benchmark

Refer: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
"""


def read_file_list(filename):
    """Read a trajectroy from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3 ..." is the 3D position and 3D orientation associated to this timestamp.

    Args:
        filename: file name.

    Returns:
        dict: dictonary of (stamp, data) tuples
    """

    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [
        [v.strip() for v in line.split(" ") if v.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)


def associate(first_list, second_list, offset, max_difference):
    """Associate two dictionaries of (stamp, data). As the time stamps never match exactly, we aim to find the closest match for every input tuple.

    Args:
        first_list: first dictionary of (stamp, data) tuples
        second_list: second dictionary of (stamp, data) tuples
        offset: time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference: search radius for candidate generation

    Returns:
        matches: list of matched tuples ((stamp1, data1), (stamp2, data2))
    """

    first_keys = list(first_list)
    second_keys = list(second_list)
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            # first_keys.remove(a)
            # second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


def associate_three(first_list, second_list, third_list, offset, max_difference):
    """Associate two dictionaries of (stamp, data). As the time stamps never match exactly, we aim to find the cloeset match for every input tuple.

    Args:
        first_list: first dict of (stamp, data) tuples (default to be rgb)
        second_list: second dict of (stamp, data) tuplse (default to be depth)
        third_list: third dict of (stamp, data) tuples (default to be pose)
        offset: time offset between dictionaries (e.g., to model the delay between the sensors)
        max_difference: search radius for candidate generation
    """

    first_keys = list(first_list)
    second_keys = list(second_list)
    third_keys = list(third_list)

    # find the potential matches in (rgb, depth)
    matches_ab = associate(first_list, second_list, offset, max_difference)

    # find the potential matches in (rgb, depth, pose)
    potential_matches = [
        (abs(a - (c + offset)), abs(b - (c + offset)), a, b, c)
        for (a, b) in matches_ab
        for c in third_keys
        if abs(a - (c + offset)) < max_difference
        and abs(b - (c + offset)) < max_difference
    ]

    potential_matches.sort()
    matches_abc = []
    for diff_rbg, diff_depth, a, b, c in potential_matches:
        if a in first_keys and b in second_keys and c in third_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            third_keys.remove(c)
            matches_abc.append((a, b, c))
    matches_abc.sort()
    return matches_abc


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
