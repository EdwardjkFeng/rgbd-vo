"""
Data loader fot TUM RBGD benchmark
"""


import sys, os
import os.path as osp
import pickle
from pathlib import Path

import numpy as np
import logging
import torch.utils.data as data

import cv2
from tqdm import tqdm

from .base_dataset import BaseDataset
from .setting import DATA_PATH
from .utils import *
from dataset import logger

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

def tum_trainval_dict():
    """ the sequence dictionary of TUM dataset
        https://vision.in.tum.de/data/datasets/rgbd-dataset/download

        The calibration parameters refers to: 
        https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats 
    """
    return  {
        'fr1': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['rgbd_dataset_freiburg1_desk2',
                    'rgbd_dataset_freiburg1_floor',
                    'rgbd_dataset_freiburg1_room',
                    'rgbd_dataset_freiburg1_xyz',
                    'rgbd_dataset_freiburg1_rpy',
                    'rgbd_dataset_freiburg1_plant',
                    'rgbd_dataset_freiburg1_teddy',
                    ]
        },

        'fr2': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['rgbd_dataset_freiburg2_360_hemisphere',
                    'rgbd_dataset_freiburg2_large_no_loop',
                    'rgbd_dataset_freiburg2_large_with_loop',
                    'rgbd_dataset_freiburg2_pioneer_slam',
                    'rgbd_dataset_freiburg2_pioneer_slam2',
                    'rgbd_dataset_freiburg2_pioneer_slam3',
                    'rgbd_dataset_freiburg2_xyz',
                    'rgbd_dataset_freiburg2_rpy',
                    'rgbd_dataset_freiburg2_coke',
                    'rgbd_dataset_freiburg2_dishes',
                    'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
                    'rgbd_dataset_freiburg2_metallic_sphere2',
                    'rgbd_dataset_freiburg2_flowerbouquet',
                    'rgbd_dataset_freiburg2_360_kidnap',
                    'rgbd_dataset_freiburg2_desk_with_person',
                    ]
        },

        'fr3': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': [
                'rgbd_dataset_freiburg3_cabinet',
                'rgbd_dataset_freiburg3_nostructure_notexture_far',
                'rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                'rgbd_dataset_freiburg3_nostructure_texture_far',
                'rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
                # 'rgbd_dataset_freiburg3_long_office_household',
                'rgbd_dataset_freiburg3_structure_notexture_near',
                'rgbd_dataset_freiburg3_structure_texture_far',
                'rgbd_dataset_freiburg3_structure_texture_near',
                'rgbd_dataset_freiburg3_teddy',
                'rgbd_dataset_freiburg3_walking_halfsphere',
                'rgbd_dataset_freiburg3_walking_rpy',
                'rgbd_dataset_freiburg3_sitting_rpy',
                'rgbd_dataset_freiburg3_sitting_static',
                'rgbd_dataset_freiburg3_sitting_xyz',
            ]
        }
    }

def tum_test_dict():
    """ the trajectorys held out for testing TUM dataset
    """
    return  {
        'fr1': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['rgbd_dataset_freiburg1_360',
                    'rgbd_dataset_freiburg1_desk']
        },

        'fr2': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['rgbd_dataset_freiburg2_desk',
                    'rgbd_dataset_freiburg2_pioneer_360']
        },

        # 'fr3': {
        #     'calib': [525.0, 525.0, 319.5, 239.5],
            # 'seq': ['rgbd_dataset_freiburg3_walking_static', # dynamic scene
            #         'rgbd_dataset_freiburg3_walking_xyz',        # dynamic scene
            #         'rgbd_dataset_freiburg3_long_office_household']
        # },

        'default': {
            'calib': [525.0, 525.0, 319.5, 239.5],
            'seq': ['None',  # anything not list here
                    ]
        }
    }

class TUM(BaseDataset, data.Dataset):
    default_conf = {
        'dataset_dir': 'TUM/',
        'sequence': 'rgbd_dataset_freiburg1_desk',
        'truncate_depth': True,
        'grayscale': False,
        'resize': None,
    }

    def _init(self, conf):
        pass

    def get_dataset(self):
        return _Dataset(self.conf)



class _Dataset(data.Dataset):
    def __init__(self, conf):
        super().__init__()
        self.root = Path(DATA_PATH, conf.dataset_dir, conf.sequence)
        self.conf = conf

        self.rgb_files = sorted([f for f in os.listdir(os.path.join(self.root, 'rgb')) if f.endswith('.png')])
        self.depth_files = sorted([f for f in os.listdir(os.path.join(self.root, 'depth')) if f.endswith('.png')])
              
    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        rgb_image = read_image(str(Path(self.root, 'rgb', rgb_path)), self.conf.grayscale)
        size = rgb_image.shape[:2]
        data = {
            'name': str(rgb_path),
            'image': numpy_image_to_torch(rgb_image),
            'original_image_size': np.array(size)
        }
        return data

    def __len__(self):
        return len(self.rgb_files)