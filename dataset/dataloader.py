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
COFUSION_DIR = get_datasets_path("CoFusion")
# ICL_DIR = get_datasets_path("ICL_NUIM")
# ETH3D_DATASET_DIR = get_datasets_path("ETH3D")
# OXFORD_DIR = get_datasets_path("Oxford_multimotion")
# TARTAN_DATASET_DIR = get_datasets_path("TartanAir")


def load_data(
    dataset_name,
    keyframes=None,
    load_type="train",
    setup=None,
    select_trajectory="",
    load_numpy=False,
    image_resize=0.25,
    truncate_depth=True,
    options=None,
    pair="incremental",
):
    image_resize = options.image_resize

    if not load_numpy:
        if load_type == "train":
            data_transform = image_transforms(["color_augment", "numpy2torch"])
        else:
            data_transform = image_transforms(["numpy2torch"])
    else:
        data_transform = image_transforms([])

    if dataset_name == "TUM_RGBD":
        from dataset.tum_rgbd import TUM

        loader = TUM(
            basedir=TUM_DATASET_DIR,
            category=load_type,
            setup=setup,
            select_traj=select_trajectory,
            keyframes=keyframes,
            data_transform=data_transform,
            image_resize=image_resize,
            truncate_depth=truncate_depth,
            add_vl_dataset=False,
        )

    elif dataset_name == "Bonn_RGBD":
        from dataset.bonn_rgbd import Bonn
        loader = Bonn(
            basedir=BONN_DATASET_DIR,
            category=load_type,
            setup=setup,
            select_traj=select_trajectory,
            keyframes=keyframes,
            data_transform=data_transform,
            image_resize=image_resize,
            truncate_depth=truncate_depth,
        )

    elif dataset_name == "CoFusion":
        from dataset.cofusion import CoFusion

        loader = CoFusion(
            basedir=COFUSION_DIR,
            category=load_type,
            select_traj=select_trajectory,
            keyframes=keyframes,
            data_transform=data_transform,
            image_resize=image_resize,
            truncate_depth=truncate_depth,
            noisy_depth=True,
            load_object_data=True,
        )

    else:
        raise NotImplementedError()

    return loader


def image_transforms(options):
    transform_list = []

    if "color_augment" in options:
        augment_parameters = [0.9, 1.1, 0.9, 1.1, 0.9, 1.1]
        transform_list.append(AugmentImages(augment_parameters))

    if "numpy2torch" in options:
        transform_list.append(ToTensor())

    # if 'color_normalize' in options: # we do it on the fly
    #     transform_list.append(ColorNormalize())

    return transforms.Compose(transform_list)


class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        return [self.transform(x) for x in sample]


class AugmentImages(object):
    def __init__(self, augment_parameters):
        self.gamma_low = augment_parameters[0]  # 0.9
        self.gamma_high = augment_parameters[1]  # 1.1
        self.brightness_low = augment_parameters[2]  # 0.9
        self.brightness_high = augment_parameters[3]  # 1,1
        self.color_low = augment_parameters[4]  # 0.9
        self.color_high = augment_parameters[5]  # 1.1

        self.thresh = 0.5

    def __call__(self, sample):
        p = np.random.uniform(0, 1, 1)
        if p > self.thresh:
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            random_brightness = np.random.uniform(
                self.brightness_low, self.brightness_high
            )
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            for x in sample:
                x = x**random_gamma  # randomly shift gamma
                x = x * random_brightness  # randomly shift brightness
                for i in range(3):  # randomly shift color
                    x[:, :, i] *= random_colors[i]
                    x[:, :, i] *= random_colors[i]
                x = np.clip(x, a_min=0, a_max=1)  # saturate
            return sample
        else:
            return sample
