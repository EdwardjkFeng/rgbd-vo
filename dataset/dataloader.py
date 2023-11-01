""" The dataloaders for running VO """

import torchvision.transforms as transforms
import numpy as np
import os
import socket
import yaml

try:
    # use fast C loader if available
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def get_datasets_path(which_dataset):
    """Smarter getter of the dataset path of the specified dataset from a yaml file. Since datasets are stored in different paths on different end, this function detects the hostname of the machine and pick the predefined path under this name."""

    curr_path = os.path.realpath(__file__)
    env_file_path = os.path.realpath(
        os.path.join(curr_path, "../../config/datasets.yaml")
    )
    hostname = str(socket.gethostname())
    env_config = yaml.load(open(env_file_path), Loader=Loader)
    return env_config[which_dataset][hostname]["dataset_root"]


TUM_DATASET_DIR = get_datasets_path("TUM_RGBD")
BONN_DATASET_DIR = get_datasets_path("Bonn_RGBD")
# COFUSION_DIR = get_datasets_path("CoFusion")
# ICL_DIR = get_datasets_path("ICL_NUIM")
# ETH3D_DATASET_DIR = get_datasets_path("ETH3D")
# OXFORD_DIR = get_datasets_path("Oxford_multimotion")
# TARTAN_DATASET_DIR = get_datasets_path("TartanAir")


def load_data(dataset_name, conf):
    if dataset_name == "TUM_RGBD":
        from dataset.tum_rgbd import TUM

        conf["dataset_dir"] = TUM_DATASET_DIR
        loader = TUM(conf).get_dataset()
    
    elif dataset_name == "Bonn_RGBD":
        from dataset.bonn_rgbd import Bonn

        conf["dataset_dir"] = BONN_DATASET_DIR
        loader = Bonn(conf).get_dataset()

    else:
        raise NotImplementedError()

    return loader
