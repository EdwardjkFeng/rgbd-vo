""" Utilities to provide dataset """

import os
import os.path as osp
import pickle

import numpy as np
from transforms3d import quaternions


def tq2mat(tq):
    """Transform translation-quaternion (tq) to (4x4) matrix."""
    tq = np.array(tq)
    T = np.eye(4)
    T[:3, :3] = quaternions.quat2mat(np.roll(tq[3:], 1))
    T[:3, 3] = tq[:3]
    return T


def write_sync_trajectory(local_dir, subject_name, dataset=None, max_diff=0.02):
    """Generate synchronized trajectories.

    Args:
        local_dir: the root of the directory
        subject_name: the name of the sequence
        dataset: the name of the dataset. If provide, save in child directory
    """

    rgb_file = osp.join(local_dir, subject_name, "rgb.txt")
    depth_file = osp.join(local_dir, subject_name, "depth.txt")
    pose_file = osp.join(local_dir, subject_name, "groundtruth.txt")

    rgb_list = filter_exist_files(
        osp.join(local_dir, subject_name), read_file_list(rgb_file)
    )
    depth_list = filter_exist_files(
        osp.join(local_dir, subject_name), read_file_list(depth_file)
    )
    pose_list = read_file_list(pose_file)

    matches = associate_three(
        rgb_list, depth_list, pose_list, offset=0.0, max_difference=max_diff
    )

    trajectory_info = []
    for a, b, c in matches:
        pose = [float(x) for x in pose_list[c]]
        rgb_file = osp.join(local_dir, subject_name, rgb_list[a][0])
        depth_file = osp.join(local_dir, subject_name, depth_list[b][0])
        trajectory_info.append([pose, rgb_file, depth_file])

    if dataset:
        datacache_root = osp.join(osp.dirname(__file__), 'cache', dataset)
    else:
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



""" =============================================================== """
"""      Some utilities to work with TUM RGB-D data                 """
""" =============================================================== """
"""
The following utility files are provided by TUM RGBD dataset benchmark

Refer: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
"""


def read_file_list(filename):
    """Read a list of poses/image_paths from a text file.

    groundtruth file format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3 ..." is the 3D position and 3D orientation associated to this timestamp.

    depth/rgb file format:
    The file format is "stamp image_path", where stamp denotes the time stamp (to be matched) 

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


def filter_exist_files(dir: str, filedict: dict):
    new_list = [(k, v) for k, v in filedict.items() 
                if osp.exists(osp.join(dir, v[0]))]
    return dict(new_list)


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
            first_keys.remove(a)
            second_keys.remove(b)
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
    # matches_ab = associate(first_list, second_list, offset, max_difference)
    potential_matches_ab = [(abs(a - (b + offset)), a, b)
                            for a in first_keys
                            for b in second_keys
                            if abs(a - (b + offset)) < max_difference]
    potential_matches_ab.sort()
    matches_ab = []
    for diff, a, b in potential_matches_ab:
        if a in first_keys and b in second_keys:
            matches_ab.append((a, b))

    matches_ab.sort()

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