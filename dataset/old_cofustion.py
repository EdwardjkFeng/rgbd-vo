"""
Co-Fusion Dataset
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import os.path as osp
import glob
from natsort import natsorted
import random
import pickle
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import logging
import torch.utils.data as data

import cv2
import tqdm

from .base_dataset import BaseDataset
from .utils import *
from .dataset_utils import *
from dataset import logger
from utils.tools import print_conf

"""
The following scripts use the dirctory structure as:
root
└── room4-full
    ├── calibration.txt
    ├── colour
    │   ├── Color0001.png
    │   ├── Color0002.png
    ...
    ├── depth_noise
    │   ├── Depth0001.exr
    │   ├── Depth0002.exr
    ...
    ├── depth_original
    │   ├── Depth0001.exr
    │   ├── Depth0002.exr
    ...
    ├── mask_colour
    │   ├── Mask0001.png
    │   ├── Mask0002.png
    ...
    ├── mask_id
    │   ├── Mask0001.png
    │   ├── Mask0002.png
    ...
    └── trajectories
        ├── gt-cam-0.txt
        ├── gt-car-2.txt
        ...
"""


def cofusion_sequences_dict():
    return {
        "room4-full": {"calib": [360, 360, 320, 240]},
        "car4-full": {"calib": [564.3, 564.3, 480, 270]},
    }


class CoFusion(BaseDataset):
    default_conf = {
        "name": "CoFusion",
        "num_workers": 8,
        "train_batch_size": 1,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "dataset_dir": "CoFusion/",
        "select_traj": "room4-full",
        "category": "test",
        "keyframes": [1],
        "truncate_depth": True,
        "noisy_depth": True,
        "load_object_data": True,
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
        self.object_mask_seq = []
        self.invalid_seq = []  # empty
        self.cam_pose_seq = []  # list(seq) of list(frame) of 4 x 4 ndarray
        self.object_pose_seq = []
        self.object_vis_idx = []
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
        raise NotImplementedError()

    def __load_test(self):
        cofusion_data = cofusion_sequences_dict()
        assert len(self.keyframes) == 1
        kf = self.conf.keyframes[0]
        self.keyframes = [1]

        for seq_name, seq_config in cofusion_data.items():
            seq_path = seq_name

            if self.conf.select_traj is not None:
                if seq_path != self.conf.select_traj:
                    continue

            self.calib.append(seq_config["calib"])

            images = self.__read_file_list(seq_name, "colour")
            total_num = len(images)
            images = [images[idx] for idx in range(0, total_num, kf)]

            if self.conf.noisy_depth:
                depth_dir = "depth_noise"
            else:
                depth_dir = "depth_original"
            depths = [
                self.__read_file_list(seq_name, depth_dir)[idx]
                for idx in range(0, total_num, kf)
            ]

            poses_dict = self.__read_pose_list(
                osp.join(self.root, seq_name, "trajectories/gt-cam-0.txt")
            )
            poses = [
                tq2mat([v for v in list(poses_dict.values())][idx])
                for idx in range(0, total_num, kf)
            ]
            
            timestamp = [
                [k for k in list(poses_dict.keys())][idx]
                for idx in range(0, total_num, kf)
            ]

            if self.conf.load_object_data:
                object_masks = self.__read_file_list(seq_name, "mask_id")
                object_masks = [object_masks[idx] for idx in range(0, total_num, kf)]

                poses_files = glob.glob(osp.join(self.root, seq_name, "trajectories/*.txt"))
                poses_files.remove(osp.join(self.root, seq_name, "trajectories/gt-cam-0.txt"))

                # load object poses
                object_idx_pose_pair = {}
                for f in poses_files:
                    object_index = int(f.split("/")[-1].split("-")[-1].split(".")[0])
                    object_pose_dict = self.__read_pose_list(f)
                    object_idx_pose_pair.update({object_index: object_pose_dict})
                
                object_poses = []
                # object_indices = []
                for i in range(0, total_num, kf):
                    t = timestamp[i]
                    object_poses_t = {}
                    for obj in object_idx_pose_pair.keys():
                        if t in object_idx_pose_pair[obj].keys():
                            object_poses_t.update(
                                {obj: tq2mat(object_idx_pose_pair[obj][t])}
                            )
                    object_poses.append(object_poses_t)

                # object_indices = [list(pair.keys()) for pair in object_poses]

            self.timestamp.append(timestamp)
            self.image_seq.append(images)
            self.depth_seq.append(depths)
            self.cam_pose_seq.append(poses)
            self.object_mask_seq.append(object_masks)
            self.object_pose_seq.append(object_poses)
            # self.object_vis_idx.append(object_indices)
            self.seq_names.append(seq_name)
            self.ids += max(0, len(images) - 1)
            self.seq_acc_ids.append(self.ids)

    def __getitem__(self, idx):
        seq_idx = max(np.searchsorted(self.seq_acc_ids, idx + 1) - 1, 0)
        frame_idx = idx - self.seq_acc_ids[seq_idx]

        this_idx = frame_idx
        next_idx = frame_idx + random.choice(self.keyframes)

        color0, scale = self.__load_rgb_tensor(self.image_seq[seq_idx][this_idx])
        color1, _ = self.__load_rgb_tensor(self.image_seq[seq_idx][next_idx])

        depth0 = self.__load_depth_tensor(self.depth_seq[seq_idx][this_idx])
        depth1 = self.__load_depth_tensor(self.depth_seq[seq_idx][next_idx])

        # object mask
        masks0 = self.__load_mask_tensor(self.object_mask_seq[seq_idx][this_idx])
        masks1 = self.__load_mask_tensor(self.object_mask_seq[seq_idx][next_idx])

        # object indices
        object_indices0 = np.delete(np.unique(masks0), 0)
        object_indices1 = np.delete(np.unique(masks1), 0)

        # object transform
        vis_obj = np.union1d(object_indices0, object_indices1)
        object_transform = {}
        for obj in vis_obj:
            try:
                obj_pose0 = self.object_pose_seq[seq_idx][this_idx][obj]
                obj_pose1 = self.object_pose_seq[seq_idx][next_idx][obj]
                object_transform.update(
                    {obj: np.dot(np.linalg.inv(obj_pose1), obj_pose0).astype(np.float32)}
                )
            except:
                object_transform.update(
                    {obj: np.eye(4).astype(np.float32)}
                )


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

        if self.conf.load_object_data:
            data.update({
                "object_info": [masks0, masks1 ,object_indices0, object_indices1, object_transform]
            })
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
        """Load depth"""
        depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth, _ = resize(depth, self.conf.resize, interp="nearest")
        if self.truncate_depth:
            # the accurate range of kinect depth
            valid_depth = (depth > 0.5) & (depth < 5.0)
            depth = np.where(valid_depth, depth, 0.0)
        return depth[None, :]  # channel first convention
    
    def __load_mask_tensor(self, path):
        """Load mask"""
        mask = cv2.imread(path).astype(np.int8)
        mask, _ = resize(mask, self.conf.resize, interp="nearest")
        return mask[None, :]

    def __read_file_list(self, seq_name, folder):
        """Read a list of image_paths"""
        file_list = natsorted(glob.glob(osp.join(self.root, seq_name, folder, "*")))
        return file_list

    def __read_pose_list(self, posefile):
        """Read a list of pose from a text file"""
        file = open(posefile)
        data = file.read()
        lines = data.split("\n")
        list = [
            [v.strip() for v in line.split(" ") if v.strip() != ""]
            for line in lines
            if len(line) > 0 and line[0] != "#"
        ]
        list = [((int(l[0]), [float(x) for x in l[1:]])) for l in list if len(l) > 1]
        return dict(list)


if __name__ == "__main__":
    test_conf = {
        "name": "CoFusion",
        "num_workers": 8,
        "train_batch_size": 1,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "dataset_dir": "/home/jingkun/Dataset/CoFusion/",
        "select_traj": "room4-full",
        "category": "test",
        "keyframes": [1],
        "truncate_depth": True,
        "noisy_depth": True,
        "grayscale": False,
        "resize": 0.25,
    }

    conf = OmegaConf.create(test_conf)

    loader = [CoFusion(conf).get_dataset()[i] for i in [799, 832]]
    import torchvision.utils as torch_utils
    from utils.visualize import create_mosaic

    torch_loader = data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=4)

    for batch in torch_loader:
        item = batch
        color0, color1, depth0, depth1, transform, calib = item["data"]
        B, C, H, W = color0.shape

        bcolor0_img = torch_utils.make_grid(color0, nrow=4)
        depth0_img = depth0.squeeze().cpu().numpy()

        import matplotlib.pyplot as plt

        plt.figure()
        # plt.imshow(bcolor0_img.numpy().transpose(1, 2, 0))
        plt.axis('off')
        plt.imshow(depth0_img, cmap="gray")
        plt.show()

        # cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
        # cv2.imshow("Depth", depth0_img)
        # cv2.waitKey(10)
