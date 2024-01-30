""" Tools """
import os
import inspect
import numpy as np
from transforms3d import quaternions
from omegaconf import OmegaConf, DictConfig


def get_class(mod_name, base_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """

    mod_path = "{}.{}".format(base_path, mod_name)
    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def print_conf(conf: DictConfig, prefix: str = ""):
    """Print configurations."""
    print(f"{' Configurations ':=^80}")
    for k, v in conf.items():
        if isinstance(v, DictConfig):
            # Recursively iterate through nested configs
            print_conf(v, prefix=f"{prefix}{k}.")
        else:
            print(f"{prefix}{k}: {v}")


def save_trajectory(seq, time_pose_pairs, name="est_traj.txt"):
    poses_qt = []
    timestamp = [time_pose_pairs[i][0] for i in range(len(time_pose_pairs))]
    poses_mat = [time_pose_pairs[i][1] for i in range(len(time_pose_pairs))]
    for i, pose_mat in enumerate(poses_mat):
        t = pose_mat[:3, 3].reshape(-1)
        rot_mat = np.identity(4)
        rot_mat[:3, :3] = pose_mat[:3, :3]
        qt = quaternions.mat2quat(rot_mat[:3, :3])
        qt = np.roll(qt, 3)
        poses_qt.append(np.append(t.A1, qt).astype(np.float16))

    out_dir = f"./eval_results/{seq}"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_path = f"{out_dir}/{name}"
    with open(out_path, "w") as f:
        for i, pose_qt in enumerate(poses_qt):
            outstr = f"{timestamp[i]}"
            outstr += " "
            outstr += " ".join(
                ["{:.6f}".format(pose_qt[i]) for i in range(len(pose_qt))]
            )
            f.write(outstr + "\n")
        print(f'Saved trajectory in {out_path}')
