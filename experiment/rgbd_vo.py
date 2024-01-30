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
from geometry import geometry
from utils.select_method import select_method
from utils.run_utils import check_cuda
from dataset.dataloader import load_data
from utils.logger import check_directory

from utils.o3d_rgbd_vo import RGBDOdometry
from utils.o3d_icp_vo import ICP_Odometry
from utils import visualize
from utils.tools import save_trajectory

from kornia.morphology import dilation
from dataset.cofusion import CoFusion

from models.segmentation import MaskRCNN, SamPrompter
from torchvision.transforms.functional import resize


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
        self.last_gt_pose = None
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

        self.est_poses = []  # list of timestamp, estimated poses pair

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
        self.visualizer.get_render_option().point_size = 0.5

    def update_renderer(self):
        self.vis_cam = self.vis_ctrl.convert_from_pinhole_camera_parameters(
            self.vis_cam
        )
        self.visualizer.poll_events()
        self.visualizer.update_renderer()


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
    valid = ~np.isnan(depth) & (depth > 0)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, np.nan)
    y = np.where(valid, z * (r - cy) / fy, np.nan)
    pc = np.dstack((x, y, z))

    if depth_type == "euclidean":
        norm = np.linalg.norm(pc, axis=2)
        pc = pc * (z / norm)[:, :, None]
    return pc


def compute_residual(rgb0, dpt0, rgb1, dpt1, K, pose):
    pose = torch.from_numpy(pose).cuda()
    R = pose[:3, :3]
    t = pose[:3, 3:4]
    pose = [R[None], t[None]]

    B, C, H, W = rgb0.shape
    px, py = geometry.generate_xy_grid(B, H, W, K)
    u_warped, v_warped, z_warped = geometry.batch_warp_coord(px, py, dpt0, pose, K)

    # Occlusion
    occ = geometry.check_occ(
        z_warped, dpt0, u_warped.type_as(dpt0), v_warped.type_as(dpt0)
    )

    # Photometric residual
    rgb1_1to0 = geometry.warp_features(
        rgb1, u_warped.type_as(dpt0), v_warped.type_as(dpt0)
    )
    r_I = torch.mean((rgb1_1to0 - rgb0), dim=1).squeeze()
    r_I[occ.squeeze()] = 0

    # Geometric residaul
    dpt1_1to0 = geometry.warp_features(
        dpt1, u_warped.type_as(dpt0), v_warped.type_as(dpt0)
    )
    r_Z = (dpt1_1to0 - z_warped).squeeze()
    r_Z[occ.squeeze()] = 0

    # 3DEPE
    v0 = geometry.compute_vertex(dpt0, px, py)
    v1 = geometry.compute_vertex(dpt1, px, py)
    x1_warped = torch.mm(R, v1.view(3, H * W)) + t.view(3, 1).expand(3, H * W)
    r_3depe = torch.norm(x1_warped.view(3, H, W) - v0, dim=1)

    # Point-to-Plane Error
    v1 = geometry.compute_vertex(dpt1_1to0, px, py)
    n0 = geometry.compute_normal(v0)
    # n1 = geometry.compute_normal(v1)

    r_p2p = torch.einsum("ijkl, ijkl -> ikl", v0 - v1, n0).squeeze()
    r_p2p[occ.squeeze()] = 0

    residual = visualize.create_mosaic(
        [
            rgb0[0],
            torch.abs(r_I),
            torch.abs(r_Z),
            torch.abs(r_p2p),
            occ.logical_not_().squeeze().int(),
            torch.abs(r_p2p),
        ],
        # cmap=["NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL"],
        cmap=cv2.COLORMAP_JET,
        order="CHW",
        normalize=True,
    )
    cv2.namedWindow("feature-metric residuals", cv2.WINDOW_NORMAL)
    cv2.imshow("feature-metric residuals", residual)
    cv2.waitKey(10)


def callback(scene: "MyScene"):
    if not scene.is_play:
        return

    dataset = scene.dataloader
    options = scene.options
    if scene.index >= len(dataset):
        return

    scene.vis_cam = scene.vis_ctrl.convert_to_pinhole_camera_parameters()

    if scene.vo_type == "incremental":
        batch = dataset[scene.index]
    else:
        batch = dataset.get_keypair(scene.index)
    color0, color1, depth0, depth1, GT_Rt, intrins = check_cuda(batch["data"])
    name = batch["name"]

    # masks0, masks1 ,object_indices0, object_indices1, object_transform = check_cuda(batch["object_info"])


    if scene.filter_moving_objects:
        # color0 = CoFusion.filter_moving_objects(color0, masks0)
        depth0 = CoFusion.filter_moving_objects(depth0, masks0)
        # color1 = CoFusion.filter_moving_objects(color1, masks1)
        depth1 = CoFusion.filter_moving_objects(depth1, masks1)

    # if scene.index > 1:
    #     return

    scene_id = name["seq"]

    # Reset scene for new scene.
    if scene_id != scene.video_id:
        scene.reset()
        # scene.init_idx = scene.index
        scene.video_id = scene_id
    else:
        scene.init = False

    GT_WC = dataset.cam_pose_seq[0][scene.index]  # ground truth camera pose
    depth_file = dataset.depth_seq[0][scene.index]
    if not options.save_img:
        # half resolution
        rgb = color1.permute((1, 2, 0)).cpu().numpy()

        # depth = (
        #     imread(depth_file, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        #     / scene.scale_factor
        # )
        # depth = cv2.resize(
        #     depth,
        #     None,
        #     fx=dataset.fx_s,
        #     fy=dataset.fy_s,
        #     interpolation=cv2.INTER_NEAREST,
        # )
        depth = depth1.squeeze().cpu().numpy()
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

    ########################################################################
    ###                           Segmentation                           ###
    ########################################################################
    if scene.generate_mask:
        mask, boxes = scene.maskrcnn(color0[None])  # maskrcnn

        # coords = torch.cat(
        #     geometry.coord_grid(mask.shape[-2], mask.shape[-1], 1),
        #     dim=1
        # ).cuda()
        # coords = coords[..., mask]
        # keypoints_tensor = coords.view(1, 2, -1).permute(0, 2, 1)
        # keypoints_tensor = keypoints_tensor[:, :: 10]
        # labels = torch.ones(keypoints_tensor.shape[:-1]).cuda()

        # keypoints_tensor = torch.tensor([[[50, 50]]], dtype=torch.float32).cuda()
        # labels = torch.tensor([[1]], dtype=torch.float32).cuda()

        # mask = scene.samprompter(color0[None], keypoints_tensor, labels)
        if boxes is not None:
            mask = scene.samprompter(color0[None], boxes)
        # mask = scene.samprompter(color0[None], mask[:, None].float())

        # depth0[mask[None]] = 0.
        # depth1[mask[None]] = 0.

        # kernel = torch.ones(5, 5).cuda()
        # mask = dilation(mask[None, None], kernel).to(bool).squeeze()
        # depth[mask.squeeze().cpu()] = 0.
        # masked_rgb = torch.where(~mask[None], color0, 0.0)
        # vis_dpt = visualize.create_mosaic(
        #     [masked_rgb, mask[None].to(int), depth[None]],
        #     cmap=["NORMAL", "NORMAL", "NORMAL"],
        #     order="CHW",
        #     normalize=True,
        # )
        # cv2.namedWindow("Masked depth", cv2.WINDOW_NORMAL)
        # cv2.imshow("Masked depth", vis_dpt)
        # cv2.waitKey(10)

    ########################################################################
    ###                              Tracking                            ###
    ########################################################################
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
        scene.est_poses.append((dataset.timestamp[0][scene.index], scene.T_WC))
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

            if scene.last_GT_WC is not None:
                gt_rel_pose = np.dot(np.linalg.inv(GT_WC), scene.last_GT_WC).astype(
                    np.float32
                )
                if scene.compute_res:
                    compute_residual(color0, depth0, color1, depth1, intrins, gt_rel_pose)
            else:
                if scene.compute_res:
                    compute_residual(color0, depth0, color1, depth1, intrins, GT_WC)
        else:
            if scene.vo_type == "incremental":
                T_CR = batch_create_transform(t, R)
                # T_CR = GT_Rt
                T_CR = T_CR.squeeze(dim=0).cpu().numpy()
                if scene.compute_res:
                    compute_residual(color0, depth0, color1, depth1, intrins, np.eye(4, dtype=np.float32))
                T_WC = np.dot(scene.T_WC, np.linalg.inv(T_CR)).astype(np.float32)
                # T_WC = np.dot(scene.T_WC, T_CR)
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
            scene.est_poses.append((dataset.timestamp[0][scene.index], scene.T_WC))

    pcd = pointcloud_from_depth(depth, fx=K["fx"], fy=K["fy"], cx=K["ux"], cy=K["uy"])
    nonnan = (~np.isnan(depth)) & (depth > 0)
    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(pcd[nonnan])
    geom.colors = o3d.utility.Vector3dVector(rgb[nonnan])

    # XYZ->RGB, Z is blue
    if options.dataset == "VaryLighting":
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)
    else:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)

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
        scene.last_GT_WC = np.copy(GT_WC)
        gt_trs = np.copy(GT_WC[0:3, 3:4]).transpose()
        trs = np.copy(T_WC[0:3, 3:4]).transpose()
        # cam = o3d.geometry.PointCloud()
        # cam.points = o3d.utility.Vector3dVector(trs)
        # cam.colors = o3d.utility.Vector3dVector(RED)
        # scene.add_geometry(cam, geom_name="camera")
        # cam_intr = np.array([[K["fx"], 0, K["ux"]],
        #                      [0, K["fy"], K["uy"]],
        #                      [0, 0, 1]])
        # cam = o3d.geometry.LineSet.create_camera_visualization(160, 120, cam_intr, T_WC, 0.07)
        # cam.colors = o3d.utility.Vector3dVector(RED)
        # scene.add_geometry(cam, geom_name="camera")

        if scene.init:
            scene.add_geometry(axis, transform=T_WC, geom_name="camera_view")
        else:
            scene.update_geometry(axis, transform=T_WC, geom_name="camera_view")

        if scene.last_pose is not None:
            lines = np.array([[0, 1]])
            # Visualize ground truth trajectory
            gt_poses_seg = np.concatenate((scene.last_gt_pose, gt_trs), axis=0)
            gt_seg = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(gt_poses_seg),
                lines=o3d.utility.Vector2iVector(lines),
            )
            gt_seg.colors = o3d.utility.Vector3dVector(GREEN)
            scene.add_geometry(gt_seg, geom_name="gt_trajectory")

            if not scene.is_gt_tracking:
                # Visualize predicted trajectory
                poses_seg = np.concatenate((scene.last_pose, trs), axis=0)
                cam_seg = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(poses_seg),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                cam_seg.colors = o3d.utility.Vector3dVector(BLUE)
                # cam_seg = trimesh.load_path(poses_seg)
                scene.add_geometry(cam_seg, geom_name="trajectory")

                # Visualize gt pred pose pairs
                pair = np.concatenate((gt_trs, trs), axis=0)
                pair_seg = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(pair),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                pair_seg.colors = o3d.utility.Vector3dVector(RED)
                scene.add_geometry(pair_seg, geom_name="gt_pred")
        scene.last_pose = trs
        scene.last_gt_pose = gt_trs

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
    # conf = {
    #     "category": "full",
    #     "keyframes": [1],
    #     "truncate_depth": True,
    #     "grayscale": False,
    #     "resize": 0.5,
    # }

    # Load data
    if options.dataset == "TUM_RGBD":
        # sequence = "rgbd_dataset_freiburg1_desk"
        # sequence = "rgbd_dataset_freiburg1_xyz"
        sequence = "rgbd_dataset_freiburg3_walking_rpy"
    elif options.dataset == "Bonn_RGBD":
        # sequence = "rgbd_bonn_balloon_tracking"
        # sequence = "rgbd_bonn_person_tracking2"
        sequence = "rgbd_bonn_synchronous"
        # sequence = "rgbd_bonn_placing_nonobstructing_box"
        # sequence = "rgbd_bonn_static"
    elif options.dataset == "CoFusion":
        sequence = "car4-full"

    print(sequence)

    np_loader = load_data(options.dataset, keyframes=[1], load_type='full', select_trajectory=sequence, truncate_depth=True, options=options)

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

    # tracker = select_method(options.vo, options)
    # tracker = RGBDOdometry("RGBD")
    tracker = ICP_Odometry("Point2Point")
    scene.network = tracker

    scene.index = 0  # config['start_frame']  # starting frame e.g. 60
    scene.video_id = None
    scene.last_pose = None
    scene.last_gt_pose = None
    scene.last_GT_WC = None
    scene.maskrcnn = MaskRCNN(target_labels=[1])
    scene.samprompter = SamPrompter()
    scene.generate_mask = False
    scene.filter_moving_objects = options.filter_moving_objects
    scene.compute_res = False
    scene.is_gt_tracking = options.gt_tracker
    scene.init = False  # True only for the first frame
    scene.is_play = True  # immediately start playing when called
    scene.vo_type = options.vo_type
    scene.two_view = options.two_view
    scene.options = options

    scene.scale_factor = (
        5e3 if scene.options.dataset in ["TUM_RGBD", "Bonn_RGBD"] else 1
    )

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

    def save_viewpoint(vis, action, mods):
        """Key callback funciton to save the viewpoint parameters."""
        if action == 1:
            print(
                "{:<80}".format("Save viewpoint parameters in viewpoint.json"),
                flush=True,
            )
            param = scene.vis_ctrl.convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters("viewpoint.json", param)

    def capture_sceenshot(vis, action, mods):
        """Load saved viewpoint parameters and capture screenshot"""
        if action == 1:
            print(
                "{:<80}".format("Read viewpoint parameters in viewpoint.json"),
                flush=True,
            )
            param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
            scene.vis_ctrl.convert_from_pinhole_camera_parameters(param)
            print("{:<80}".format("Save screenshot in screenshot.png"), flush=True)
            scene.visualizer.capture_screen_image("/home/jingkun/Insync/Jingkun_GoogleDrive/MasterThesis/figures/reconstruction/screenshot.png", do_render=True)

    scene.visualizer.register_animation_callback(callback_function)
    scene.visualizer.register_key_action_callback(ord("P"), pause_resume)
    scene.visualizer.register_key_action_callback(ord("V"), save_viewpoint)
    scene.visualizer.register_key_action_callback(ord("S"), capture_sceenshot)

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
    # scene.network.timers.print()
    save_trajectory(options.dataset + "/" + sequence, scene.est_poses, "ours_nolearning.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the network")
    config.add_basics_config(parser)
    config.add_test_basics_config(parser)
    config.add_tracking_config(parser)
    config.add_vo_config(parser)

    options = parser.parse_args()
    options.filter_moving_objects = False
    # to save visualization: --save_img and --vis_feat
    print("---------------------------------------")
    main(options)
