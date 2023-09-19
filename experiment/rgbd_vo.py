""" 
Scripts to run keyframe visual odometry on a sequence of images
"""

# import standard library
import os
import sys
import argparse
import os.path as osp

# import third party
import cv2
import numpy as np
import open3d as o3d
import torch
from imageio import imread

from dataset.dataloader import load_data





def main(options):

    # load data


    