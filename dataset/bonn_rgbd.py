""" Bonn dataset"""
import sys, os, random
import pickle

import numpy as np
import os.path as osp
import torch.utils.data as data

# from imageio import imread
import cv2
from tqdm import tqdm
from transforms3d import quaternions
from cv2 import resize, INTER_NEAREST, imread

from dataset import logger
from .dataset_utils import *


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

def bonn_trainval_static_dict():
    return {
        "scene": {
            "calib": [542.822841, 542.576870, 315.593520, 237.756098],
            "seq": [
                "rgbd_bonn_static",
                "rgbd_bonn_static_close_far",
            ],
        },
    }

def bonn_trainval_dynamic_dict():
    return {
        "scene": {
            "calib": [542.822841, 542.576870, 315.593520, 237.756098],
            "seq": [
                "rgbd_bonn_crowd",
                "rgbd_bonn_kidnapping_box2",
                "rgbd_bonn_moving_nonobstructing_box",
                "rgbd_bonn_placing_obstructing_box",
                # "rgbd_bonn_static",
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
                # "rgbd_bonn_static_close_far",
                "rgbd_bonn_moving_obstructing_box2",
                "rgbd_bonn_kidnapping_box",
                "rgbd_bonn_person_tracking",
                # "rgbd_bonn_balloon_tracking2",
            ],
        },
    }


class Bonn(data.Dataset):

    def __init__(self, basedir = '', category='train', setup=None, 
                 select_traj=None, keyframes=[1], data_transform=None, 
                 image_resize=0.25, truncate_depth=True) -> None:
        super().__init__()

        self.name = 'Bonn_RGBD'

        self.image_seq = []  # list(seq) or list(frame) of string (rbg image path)
        self.timestamp = []  # empty
        self.depth_seq = []  # list(seq) of list(frame) of string (depth image path)
        self.invalid_seq = []  # empty
        self.cam_pose_seq = []  # list(seq) of list(frame) of 4 x 4 ndarray
        self.calib = []  # list(seq) of list(intrinsics: fx, fy, cx, cy)
        self.seq_names = []  # list(seq) or string(seq name)

        self.ids = 0
        self.seq_acc_ids = [0]
        self.keyframes = keyframes

        self.transforms = data_transform
        self.fx_s = image_resize
        self.fy_s = image_resize
        self.truncate_depth = truncate_depth

        if category in ["test", "full"]:
            self.__load_test(basedir, category, select_traj)
        elif category in ["train", "validation"]:
            self.__load_train_val(basedir, category, setup, 
                                  select_traj=select_traj)
        else:
            raise NotImplementedError()

        logger.info(
            "{:} dataloader for {:} using keyframe {:}: {:} valid frames.".format(self.name, category, keyframes, self.ids)
        )

    def __load_train_val(self, root, category, setup=None, select_traj=None):

        if setup == 'static':
            bonn_data = bonn_trainval_static_dict()
        elif setup == 'dynamic':
            bonn_data = bonn_trainval_dynamic_dict()
        else:
            bonn_data = bonn_trainval_dict()

        for seq_name in bonn_data["scene"]["seq"]:
            seq_path = seq_name
            if select_traj is not None:
                if seq_path != select_traj:
                    continue

            self.calib.append(bonn_data["scene"]["calib"])
            datacache_root = osp.join(osp.dirname(__file__), f"cache/{self.name}")
            sync_traj_file = osp.join(datacache_root, seq_path, "sync_trajectory.pkl")
            if not osp.isfile(sync_traj_file):
                logger.info(
                    f"Synchronized trajectory file {sync_traj_file} has not been generated."
                )
                logger.info("Generate it now ...")
                write_sync_trajectory(root, seq_name, dataset=self.name, max_diff=0.02)

            with open(sync_traj_file, "rb") as f:
                trainval = pickle.load(f)
                total_num = len(trainval)
                train_ids, val_ids = self.__gen_trainval_index(total_num)
                if category == "train":
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

    def __load_test(self, root, category='test', select_traj=None):

        if category == 'full':
            bonn_data = bonn_sequences_dict()
        else:
            bonn_data = bonn_test_dict()
        assert len(self.keyframes) == 1
        kf = self.keyframes[0]
        self.keyframes = [1]

        for ks, scene in bonn_data.items():
            for seq_name in scene["seq"]:
                seq_path = seq_name
                if seq_name == 'None':
                    continue

                if select_traj is not None and seq_path != select_traj:
                    continue

                self.calib.append(scene["calib"])

                # load or generate synchronized trajectory file
                datacache_root = osp.join(
                    osp.dirname(__file__), f"cache/{self.name}"
                )
                sync_traj_file = osp.join(
                    datacache_root, seq_path, "sync_trajectory.pkl"
                )
                if not osp.isfile(sync_traj_file):
                    logger.info(
                        f"Synchronized trajectory file {sync_traj_file} has not been generated."
                    )
                    logger.info("Generate it now ...")
                    write_sync_trajectory(root, seq_name, dataset=self.name, max_diff=0.02)

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
                "\nTry to support this customized dataset".format(select_traj)
            )
            if osp.exists(osp.join(root, select_traj)):
                self.calib.append(bonn_sequences_dict()["default"]["calib"])
                sync_traj_file = osp.join(root, select_traj)
                seq_name = seq_path = osp.basename(select_traj)
                if not osp.isfile(sync_traj_file):
                    logger.info(
                        "The synchronized trajctory file {:} has not been generated.".format(
                            seq_path
                        )
                    )
                    logger.info("Generate it now ...")
                    write_sync_trajectory(root, select_traj, max_diff=0.1)

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

    def get_keypair(self, index, kf_idx=0):
        # pair in the way like [[1, 3], [1, 5], [1, 7],...[1, N]]
        seq_idx = max(np.searchsorted(self.seq_acc_ids, index + 1) - 1, 0)
        frame_idx = index - self.seq_acc_ids[seq_idx]

        this_idx = kf_idx
        next_idx = frame_idx

        color0 = self.__load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1 = self.__load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        depth0 = self.__load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self.__load_depth_tensor(self.depth_seq[seq_idx][next_idx])

        if self.transforms:
            color0, color1 = self.transforms([color0, color1])

            # normalize the coordinate
        calib = np.asarray(self.calib[seq_idx], dtype=np.float32)
        calib[0] *= self.fx_s
        calib[1] *= self.fy_s
        calib[2] *= self.fx_s
        calib[3] *= self.fy_s

        cam_pose0 = self.cam_pose_seq[seq_idx][this_idx]
        cam_pose1 = self.cam_pose_seq[seq_idx][next_idx]
        transform = np.dot(np.linalg.inv(cam_pose1), cam_pose0).astype(np.float32)

        name = {'seq': self.seq_names[seq_idx],
                'frame0': this_idx,
                'frame1': next_idx}

        # camera_info = dict()
        camera_info = {"height": color0.shape[0],
                       "width": color0.shape[1],
                       "fx": calib[0],
                       "fy": calib[1],
                       "ux": calib[2],
                       "uy": calib[3]}
        
        data = {
            "name": name,
            "data": [color0, color1, depth0, depth1, transform, calib],
            "camera_info": camera_info,
        }
        return data

    def __getitem__(self, idx):
        seq_idx = max(np.searchsorted(self.seq_acc_ids, idx + 1) - 1, 0)
        frame_idx = idx - self.seq_acc_ids[seq_idx]

        this_idx = frame_idx
        next_idx = frame_idx + random.choice(self.keyframes)

        color0 = self.__load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1= self.__load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        depth0 = self.__load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self.__load_depth_tensor(self.depth_seq[seq_idx][next_idx])

        if self.transforms:
            color0, color1 = self.transforms([color0, color1])

        # normalize the coordinate
        calib = np.asarray(self.calib[seq_idx], dtype=np.float32)
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
        """ Load the rgb image. """
        image = imread(path)[:, :, [2, 1, 0]]
        image = image.astype(np.float32) / 255.0
        image = resize(image, None, fx=self.fx_s, fy=self.fy_s)
        return image

    def __load_depth_tensor(self, path):
        """Load depth:
        The depth images are scaled by a factor of 5000, i.e., a pixel value of 5000 in the depth image corresponds to a distance of 1 meter from the camera, 10000 to 2 meter distance, etc. A pixel value of 0 means missing value/no data.
        """
        # depth = read_image(path) / 5e3
        depth = imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5e3
        depth = resize(depth, None, fx=self.fx_s, fy=self.fy_s, interpolation=INTER_NEAREST)
        if self.truncate_depth:
            valid_depth = (depth > 0.5) & (depth < 5.0) # the accurate range of kinect depth
            depth = np.where(valid_depth, depth, 0.0)
        return depth[None, :]  


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


if __name__ == '__main__': 
    import torchvision.utils as torch_utils
    from dataset.dataloader import image_transforms
    # seqs = []
    # seq_names = bonn_trainval_static_dict()['scene']['seq']
    # seq_names = [name[10:] for name in seq_names]
    # seqs += (seq_names)
    # print(seqs)

    data_transform = image_transforms(['color_augment', 'numpy2torch'])

    loader = Bonn(basedir='/home/jingkun/Dataset/Bonn_RGBD_Dataset', category='test', setup=None, keyframes=[1], data_transform=data_transform)
 
    torch_loader = data.DataLoader(loader, batch_size=16, 
        shuffle=False, num_workers=4)

    for batch in torch_loader: 
        name = batch["name"]
        color0, color1, depth0, depth1, transform, K = batch["data"]
        B, C, H, W = color0.shape
        print(color0.shape)

        bcolor0_img = torch_utils.make_grid(color0, nrow=4)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(bcolor0_img.numpy().transpose(1,2,0))
        plt.show()