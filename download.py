""" Scripts to download datasets and checkpoints. """

import os
import requests
from tqdm import tqdm
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
import zipfile
import tarfile
from typing import Optional, List

# URL of TUM RGB-D dataset
URLs = dict(
    TUM_RGBD = "https://vision.in.tum.de/rgbd/dataset/",
    Bonn_RGBD = "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/"
)

TUM_RGBD_Sequences = {
    "freiburg1": [
        "rgbd_dataset_freiburg1_360",
        "rgbd_dataset_freiburg1_desk",
        "rgbd_dataset_freiburg1_desk2",
        "rgbd_dataset_freiburg1_floor",
        "rgbd_dataset_freiburg1_plant",
        "rgbd_dataset_freiburg1_room",
        "rgbd_dataset_freiburg1_rpy",
        "rgbd_dataset_freiburg1_xyz",
    ],
    "freiburg2": [
        "rgbd_dataset_freiburg2_360_hemisphere",
        "rgbd_dataset_freiburg2_360_kidnap",
        "rgbd_dataset_freiburg2_desk",
        "rgbd_dataset_freiburg2_desk_with_person",
        "rgbd_dataset_freiburg2_dishes",
        "rgbd_dataset_freiburg2_large_no_loop",
        "rgbd_dataset_freiburg2_large_with_loop",
        "rgbd_dataset_freiburg2_pioneer_360",
        "rgbd_dataset_freiburg2_pioneer_slam",
        "rgbd_dataset_freiburg2_pioneer_slam2",
        "rgbd_dataset_freiburg2_pioneer_slam3",
        "rgbd_dataset_freiburg2_rpy",
        "rgbd_dataset_freiburg2_xyz",
    ],
    "freiburg3": [
        "rgbd_dataset_freiburg3_long_office_household",
        # "rgbd_dataset_freiburg3_nostructure_notexture_far",
        # "rgbd_dataset_freiburg3_nostructure_notexture_near",
        # "rgbd_dataset_freiburg3_nostructure_texture_far",
        # "rgbd_dataset_freiburg3_nostructure_texture_near",
        "rgbd_dataset_freiburg3_sitting_halfsphere",
        "rgbd_dataset_freiburg3_sitting_rpy",
        "rgbd_dataset_freiburg3_sitting_static",
        "rgbd_dataset_freiburg3_sitting_xyz",
        # "rgbd_dataset_freiburg3_structure_notexture_far",
        # "rgbd_dataset_freiburg3_structure_notexture_near",
        # "rgbd_dataset_freiburg3_structure_texture_far",
        # "rgbd_dataset_freiburg3_structure_texture_near",
        "rgbd_dataset_freiburg3_teddy",
        "rgbd_dataset_freiburg3_walking_halfsphere",
        "rgbd_dataset_freiburg3_walking_rpy",
        "rgbd_dataset_freiburg3_walking_static",
        "rgbd_dataset_freiburg3_walking_xyz",
    ],
}

Bonn_RGBD_Sequences = {
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
        "rgbd_bonn_person_tracking2",
        "rgbd_bonn_removing_nonobstructing_box",
        "rgbd_bonn_placing_nonobstructing_box2",
        "rgbd_bonn_synchronous",
        "rgbd_bonn_crowd3",
        "rgbd_bonn_removing_obstructing_box",
        "rgbd_bonn_placing_nonobstructing_box",
        "rgbd_bonn_balloon2",
        "rgbd_bonn_crowd2",
        "rgbd_bonn_synchronous2",
        "rgbd_bonn_removing_nonobstructing_box2",
        "rgbd_bonn_static_close_far",
        "rgbd_bonn_moving_obstructing_box2",
        "rgbd_bonn_kidnapping_box",
        "rgbd_bonn_person_tracking",
        "rgbd_bonn_balloon_tracking2",
    ],
}

def download_from_url(url: str, save_path: Path, overwrite: bool = False,
                      exclude_files: Optional[List[str]] = None,
                      exclude_dirs: Optional[List[str]] = None):
    subpath = Path(urlsplit(url).path)
    num_parents = len(subpath.parents)
    exclude = ['index.html*']
    if exclude_files is not None:
        exclude += exclude_files
    cmd = [
        'wget', '-r', '-np', '-nH', '-q', '--show-progress',
        '-R', f'"{",".join(exclude)}"',
        '--cut-dirs', str(num_parents), url,
        '-P', str(save_path)
    ]
    if exclude_dirs is not None:
        path = Path(urlsplit(url).path)
        cmd += ['-X', ' '.join(str(path / d) for d in exclude_dirs)]
    if not overwrite:
        cmd += ['-nc']
    print('Downloading %s.', url)
    subprocess.run(' '.join(cmd), check=True, shell=True)


def extract_tar(tarpath: Path, extract_path: Optional[Path] = None,
                remove: bool = True):
    if extract_path is None:
        extract_path = tarpath.parent
    print('Extracting %s.', tarpath)
    with tarfile.open(tarpath, 'r') as f:
        f.extractall(extract_path)
    if remove:
        tarpath.unlink()



def download_TUM_RGBD(datapath, sequences_to_download: Optional[list] = None):
    if sequences_to_download:
        sequences = sequences_to_download
    else:
        sequences = TUM_RGBD_Sequences
        configs = [k for k in sequences.keys()]
        sequences = [f'{c}/{seq}' for c in configs for seq in sequences[c]]
    
    url = URLs['TUM_RGBD']
    out_path = Path(datapath) / 'TUM_RGBD_Dataset'

    # Create a folder to hold the dataset if it doesn't exist
    out_path.mkdir(exist_ok=True, parents=True)
    print('Downloading the TUM RGB-D Dataset...')
    for sequence in sequences:
        # Check if sequence is already downloaded
        if os.path.exists(os.path.join(out_path, sequence[10:])):
            print(f"{sequence} already exists, skip...")
            continue
        download_from_url(url + f'{sequence}.tgz', out_path)
        extract_tar(out_path / f'{sequence[10:]}.tgz', out_path)


def download_Bonn_RGBD(datapath, sequences_to_download: Optional[list] = None):
    if sequences_to_download:
        sequences = sequences_to_download
    else:
        sequences = Bonn_RGBD_Sequences
        sequences = [f'{seq}' for seq in sequences["seq"]]
    
    url = URLs["Bonn_RGBD"]
    out_path = Path(datapath) / "Bonn_RGBD_Dataset"

    out_path.mkdir(exist_ok=True, parents=True)
    print('Downloading the Bonn RGB-D Dataset...')
    for sequence in sequence:
        # Check if sequence is already downloaded
        if os.path.exists(os.path.join(out_path, sequence)):
            print(f"{sequence} already exists, skip...")
            continue
        download_from_url(url + f'{sequence}.zip', out_path)
        extract_tar(out_path / f'{sequence}.zip', out_path)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="TUM_RGBD", type=str, help="dataset to download")
    parser.add_argument("-s", "--sequences", default=[], nargs='+', help="sequences to download")
    parser.add_argument("-p", "--datapath", default='/home/jingkun/Dataset', type=str, help="directory to store the downloaded sequences")

    args = parser.parse_args()

    download_func = 'download_' + args.dataset
    download_func = locals()[download_func]

    if len(args.sequences) != 0:
        download_func(args.datapath, args.sequences)
    else:
        download_func(args.datapath)

    print("All sequences downloaded.")


