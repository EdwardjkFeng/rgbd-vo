""" 
Scripts to run keyframe visual odometry on a sequence of images
"""

# import standard library
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import argparse
import os.path as osp

# import third party
import cv2
from cv2 import imread
import numpy as np
import open3d as o3d
from copy import deepcopy
import torch

import config
from geometry.geometry import batch_create_transform, batch_mat2Rt
from utils.select_method import select_method
from run_utils import check_cuda
from dataset.dataloader import load_data
from utils.logger import check_directory


R_y_180 = np.eye(4, dtype=float)
R_y_180[0, 0] = R_y_180[2, 2] = -1.0

RED = np.array([1.0, 0, 0]).reshape(1, 3)
GREEN = np.array([0, 1.0, 0]).reshape(1, 3)
BLUE = np.array([0, 0, 1.0]).reshape(1, 3)


class MyScene:
    def __init__(self):
        self.dataloader = None
        self.network = None
        self.index = 0
        self.video_id = None
        self.last_pose = None
        self.camera_transfrom = None
        self.is_gt_tracking = False
        self.init = False
        self.is_play = True
        self.vo_type = None
        self.two_view = False
        self.options = None

        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self._init_visualizer()

        self.geometries = {}  # list to store PointClouds, Lines, etc.

    def add_geometry(self, geometry, transform=None, geom_name=None):
        geom = deepcopy(geometry)
        if transform is not None:
            geom = geom.transform(transform)
        self.visualizer.add_geometry(geom)
        if geom_name in self.geometries.keys():
            if not isinstance(self.geometries[geom_name], list):
                self.geometries[geom_name] = [self.geometries[geom_name], geom]
            else:
                self.geometries[geom_name].append(geom)
        else:
            self.geometries.update({geom_name: geom})

    def update_geometry(self, geometry, transform=None, geom_name=None):
        # Assuming geom_name would be used to find and remove the geometry from self.geometries
        self.visualizer.remove_geometry(self.geometries[geom_name])
        geom = deepcopy(geometry)
        if transform is not None:
            geom = geom.transform(transform)
        self.visualizer.add_geometry(geom)
        self.geometries[geom_name] = geom

    def reset(self):
        self.geometries = {}
        self.init = True

    def _init_visualizer(self):
        self.visualizer.create_window()
        self.vis_ctrl = self.visualizer.get_view_control()
        self.vis_cam = self.vis_ctrl.convert_to_pinhole_camera_parameters()

    def update_renderer(self):
        self.vis_cam = self.vis_ctrl.convert_from_pinhole_camera_parameters(
            self.vis_cam
        )
        self.visualizer.poll_events()
        self.visualizer.update_renderer()


# def init_scene(scene):
#     scene.geometry = {}
#     scene.graph.clear()
#     scene.init = True

#     # clear poses
#     scene.gt_poses = []
#     scene.est_poses = []
#     scene.timestamps = []

#     return scene


def camera_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ R_y_180


def pointcloud_from_depth(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_type: str = "z",
) -> np.ndarray:
    """Generate pointclouds from depth"""
    assert depth_type in ["z", "euclidean"], "Unexpected depth_type"
    assert depth.dtype.kind == "f", "depth must be float and have meter values"

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = ~np.isnan(depth)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)
    pc = np.dstack((x, y, z))

    if depth_type == "euclidean":
        norm = np.linalg.norm(pc, axis=2)
        pc = pc * (z / norm)[:, :, None]
    return pc


def callback(scene: "MyScene"):
    if not scene.is_play:
        return

    dataset = scene.dataloader
    options = scene.options
    if scene.index >= len(dataset):
        return

    scene.vis_cam = scene.vis_ctrl.convert_to_pinhole_camera_parameters()

    if scene.vo_type == "incremental":
        batch = dataset[scene.index - 1]
    else:
        batch = dataset.get_keypair(scene.index)
    color0, color1, depth0, depth1, GT_Rt, intrins = check_cuda(batch["data"])
    name = batch["name"]

    scene_id = name["seq"]

    # Reset scene for new scene.
    if scene_id != scene.video_id:
        scene.reset()
        # scene.init_idx = scene.index
        scene.video_id = scene_id
    else:
        scene.init = False

    GT_WC = dataset.cam_pose_seq[0][scene.index]
    depth_file = dataset.depth_seq[0][scene.index]
    if not options.save_img:
        # half resolution
        rgb = color1.permute((1, 2, 0)).cpu().numpy()
        depth = imread(depth_file, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5e3
        depth = cv2.resize(
            depth,
            None,
            fx=dataset.fx_s,
            fy=dataset.fy_s,
            interpolation=cv2.INTER_NEAREST,
        )
        valid_depth = (depth > 0.5) & (depth < 5.0)
        depth = np.where(valid_depth, depth, 0.0)
        # print(depth.shape)
        # depth = depth1.squeeze().cpu().numpy()
        K = {
            "fx": intrins[0].item(),
            "fy": intrins[1].item(),
            "ux": intrins[2].item(),
            "uy": intrins[3].item(),
        }
    else:
        # original resolution for demo
        rgb = imread(dataset.image_seq[0][scene.index])
        depth = imread(depth_file).astype(np.float32) / 5e3
        calib = np.asarray(dataset.calib[0], dtype=np.float32)
        K = {"fx": calib[0], "fy": calib[1], "ux": calib[2], "uy": calib[3]}

        # save input rgb and depth images
        img_index_png = str(scene.index).zfill(5) + ".png"
        if options.dataset == "VaryLighting":
            output_folder = osp.join(dataset.seq_names[0], "kf_vo", options.vo)
        else:
            output_folder = os.path.join("/home/binbin/Pictures", "kf_vo", options.vo)

        rgb_img = osp.join(output_folder, "rgb", img_index_png)
        depth_img = osp.join(output_folder, "depth", img_index_png)
        check_directory(rgb_img)
        check_directory(depth_img)
        cv2.imwrite(rgb_img, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(depth_img, depth)

    if scene.init:
        if GT_WC is not None:
            T_WC = GT_WC
        else:
            T_WC = np.array(
                [
                    [0.0, -1.0, 0.0, -0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            # T_WC = np.eye(4)
        scene.T_WC = T_WC
        if scene.vo_type == "keyframe":
            scene.T_WK = T_WC
    else:
        if GT_WC is not None:
            pose_gt = batch_mat2Rt(GT_Rt)
        with torch.no_grad():
            color0 = color0.unsqueeze(dim=0)
            color1 = color1.unsqueeze(dim=0)
            depth0 = depth0.unsqueeze(dim=0)
            depth1 = depth1.unsqueeze(dim=0)
            intrins = intrins.unsqueeze(dim=0)
            
            if options.save_img:
                output = scene.network.forward(
                    color0, color1, depth0, depth1, intrins, index=scene.index
                )
            else:
                output = scene.network.forward(color0, color1, depth0, depth1, intrins, pose=None)
        R, t = output
        if scene.is_gt_tracking:
            T_WC = GT_WC
            scene.T_WC = T_WC
        else:
            if scene.vo_type == "incremental":
                T_CR = batch_create_transform(t, R)
                # T_CR = GT_Rt
                T_CR = T_CR.squeeze(dim=0).cpu().numpy()
                T_WC = np.dot(scene.T_WC, np.linalg.inv(T_CR)).astype(np.float32)
            elif scene.vo_type == "keyframe":
                T_CK = batch_create_transform(t, R)
                # T_CK = GT_Rt
                T_CK = T_CK.squeeze(dim=0).cpu().numpy()
                T_WC = np.dot(scene.T_WK, np.linalg.inv(T_CK)).astype(np.float32)

                # print large drift in keyframe tracking,
                # just for noticing a possible tracking failure, not usd later
                T_CC = np.dot(np.linalg.inv(T_WC), scene.T_WC).astype(np.float32)
                trs_drift = np.copy(T_CC[0:3, 3:4]).transpose()
                if np.linalg.norm(trs_drift) > 0.02:
                    print(depth_file)
            else:
                raise NotImplementedError()
            scene.T_WC = T_WC

    pcd = pointcloud_from_depth(depth, fx=K["fx"], fy=K["fy"], cx=K["ux"], cy=K["uy"])
    nonnan = (~np.isnan(depth)) & (depth > 0)
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(pcd[nonnan])
    geom.colors = o3d.utility.Vector3dVector(rgb[nonnan])

    # XYZ->RGB, Z is blue
    if options.dataset == "VaryLighting":
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)
    elif options.dataset in ["TUM_RGBD", "ScanNet", "Bonn_RGBD"]:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    else:
        raise NotImplementedError()

    # two view: keyframe - live frames visualization
    if scene.vo_type == "keyframe" and scene.two_view:
        if scene.init:
            scene.add_geometry(geom, transform=scene.T_WK, geom_name="key")
            scene.add_geometry(geom, transform=T_WC, geom_name="live")
            scene.add_geometry(axis, transform=T_WC, geom_name="camera_view")
        else:
            # after the first view, delete the old live view and add new live view
            scene.update_geometry(geom, transform=T_WC, geom_name="live")
            scene.update_geometry(axis, transform=T_WC, geom_name="camera_view")

    else:
        # scene.add_geometry(geom, transform=T_WC)
        scene.add_geometry(geom, transform=T_WC, geom_name="live")

        # draw camera trajectory
        trs = np.copy(T_WC[0:3, 3:4]).transpose()
        cam = o3d.geometry.PointCloud()
        cam.points = o3d.utility.Vector3dVector(trs)
        cam.colors = o3d.utility.Vector3dVector(RED)
        scene.add_geometry(cam, geom_name="camera")

        if scene.init:
            scene.add_geometry(axis, transform=T_WC, geom_name="camera_view")
        else:
            scene.update_geometry(axis, transform=T_WC, geom_name="camera_view")

        if scene.last_pose is not None:
            poses_seg = np.concatenate((scene.last_pose, trs), axis=0)
            lines = np.array([[0, 1]])
            cam_seg = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(poses_seg),
                lines=o3d.utility.Vector2iVector(lines),
            )
            cam_seg.colors = o3d.utility.Vector3dVector(BLUE)
            # cam_seg = trimesh.load_path(poses_seg)
            scene.add_geometry(cam_seg, geom_name="trajectory")
        scene.last_pose = trs

    # A kind of current camera view, but a bit far away to see whole scene.
    # scene.camera.resolution = (rgb.shape[1], rgb.shape[0])
    # scene.camera.focal = (K['fx'], K['fy'])

    # intrinsics = o3d.camera.PinholeCameraIntrinsic(
    #     rgb.shape[1], rgb.shape[0], K['fx'], K['fy'], K['ux'], K['uy'])
    # camera_params = o3d.camera.PinholeCameraParameters()
    # camera_params.intrinsic = intrinsics
    # scene.visualizer.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

    if dataset.realscene:
        if options.save_img:
            if scene.vo_type == "keyframe":
                T_see = np.array(
                    [
                        [1.000, 0.000, 0.000, 0.2],
                        [0.000, 0.866, 0.500, -0.7],
                        [0.000, -0.500, 0.866, -0.8],
                        [0.000, 0.000, 0.000, 1.0],
                    ]
                )

                scene.camera_transform = camera_transform(np.matmul(scene.T_WK, T_see))

        else:
            # adjust which transformation use to set the see pose
            if scene.vo_type == "keyframe":
                T_see = np.array(
                    [
                        [1.000, 0.000, 0.000, 0.2],
                        [0.000, 0.866, 0.500, -0.7],
                        [0.000, -0.500, 0.866, -0.8],
                        [0.000, 0.000, 0.000, 1.0],
                    ]
                )

                scene.camera_transform = camera_transform(np.matmul(scene.T_WK, T_see))
    else:
        # scene.camera.transform = T_WC @ tf.translation_matrix([0, 0, 2.5])
        scene.camera_transform = T_WC @ np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    # if scene.index == scene.init_idx + 1:
    #     input()
    # print(scene.index)
    sys.stdout.write("Tracking %s: %d frames  \r" % (scene.video_id, scene.index))
    sys.stdout.flush()
    scene.index += 1  # scene.track_config['frame_step']
    # print("<=================================")
    if options.save_img:
        return

    scene.update_renderer()


def main(options):
    conf = {
        "category": "test",
        "keyframes": [1],
        "truncate_depth": True,
        "grayscale": False,
        "resize": 0.25,
        "add_val_dataset": False,
    }

    # Load data
    if options.dataset == "TUM_RGBD":
        sequence = 'rgbd_dataset_freiburg1_desk'
        # sequence = "rgbd_dataset_freiburg3_walking_xyz"
        conf["select_traj"] = sequence
        np_loader = load_data("TUM_RGBD", conf=conf)
    elif options.dataset == "Bonn_RGBD":
        sequence = "rgbd_bonn_balloon_tracking"
        conf["select_traj"] = sequence
        np_loader = load_data(options.dataset, conf=conf)

    scene = MyScene()
    # scene.visualizer = vis
    scene.dataloader = np_loader
    scene.dataloader.realscene = True

    # keyframes = [int(x) for x in options.keyframes.split(',')]
    # if options.dataset in ['BundleFusion', 'TUM_RGBD']:
    #     obj_has_mask = False
    # else:
    #     obj_has_mask = True

    # eval_loaders = create_eval_loaders(options, options.eval_set,
    #                                    [1,], total_batch_size, options.trajectory)

    tracker = select_method(options.vo, options)
    scene.network = tracker

    scene.index = 0  # config['start_frame']  # starting frame e.g. 60
    scene.video_id = None
    scene.last_pose = None
    scene.is_gt_tracking = options.gt_tracker
    scene.init = False  # True only for the first frame
    scene.is_play = True  # immediately start playing when called
    scene.vo_type = options.vo_type
    scene.two_view = options.two_view
    scene.options = options

    # TODO this looks urgly
    def callback_function(vis):
        if not scene.is_play:
            return False

        callback(scene)
        return False

    def pause_resume(vis, action, mods):
        """Key callback function to pause and resume the tracking process."""
        if action == 1:
            scene.is_play = not scene.is_play
            if scene.is_play:
                sys.stdout.write(
                    "Tracking %s: %d frames  \r" % (scene.video_id, scene.index)
                )
            else:
                sys.stdout.write(
                    "Paused   %s: %d frames  \r" % (scene.video_id, scene.index)
                )
            sys.stdout.flush()

    scene.visualizer.register_animation_callback(callback_function)
    scene.visualizer.register_key_action_callback(ord("P"), pause_resume)

    # if not options.save_img:
    #     # scene.show()
    #     pyglet.app.run()
    # else:
    #     # import pyrender
    #     # scene_pyrender = pyrender.Scene.from_trimesh(scene)
    #     # renderer = pyrender.OffscreenRenderer(viewport_height=480, viewport_width=640, point_size=1)
    #     # rgb, depth = renderer.render(scene_pyrender)

    #     if options.dataset == 'VaryLighting':
    #         output_dir = osp.join(np_loader.seq_names[0], 'kf_vo', options.vo)
    #     else:
    #         output_dir = os.path.join(
    #             '/home/binbin/Pictures', 'kf_vo', options.vo)
    #     check_directory(output_dir + '/*.png')
    #     for frame_id in range(len(scene.dataloader)):
    #         # scene.save_image()
    #         callback(scene)
    #         file_name = os.path.join(output_dir, 'render', str(
    #             scene.index-1).zfill(5) + '.png')
    #         check_directory(file_name)
    #         with open(file_name, "wb") as f:
    #             f.write(scene.save_image())
    #             f.close()
    scene.visualizer.run()
    scene.visualizer.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the network")
    config.add_basics_config(parser)
    config.add_test_basics_config(parser)
    config.add_tracking_config(parser)
    config.add_vo_config(parser)

    options = parser.parse_args()
    # to save visualization: --save_img and --vis_feat
    print("---------------------------------------")
    main(options)
