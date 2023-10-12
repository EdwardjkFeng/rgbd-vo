"""
Visualization tool for debugging and demo
"""
import cv2

import math
import sys
import numpy as np
import torch


def manage_visualization():
    """
    Manage the visualization display according to the user's key inputs.

    Parameters:
    - idx: The current index or frame number being processed.

    Returns:
    - A boolean indicating whether to break out of the calling loop.
    """

    def close_windows():
        cv2.destroyAllWindows()
        return True

    k = cv2.waitKey(10) & 0xFF

    if k == ord("q"):
        return close_windows()  # Indicate to break out of the calling loop

    if k == ord("p"):
        # Pause the visualization
        global_pause_signal = True
        while True:
            # Wait indefinitely for the user to press 'c' to continue
            k = cv2.waitKey(0) & 0xFF
            if k == ord("s"):
                break
            elif k == ord("q"):
                return close_windows()

    return False  # Continue the calling loop


def convert_flow_for_display(flow):
    """Converts a 2D image (e.g. flow) to rgb


    Args:
        flow: optical flow of size [2, H, W]
    """

    ang = np.arctan2(flow[1, :, :], flow[0, :, :])
    ang[ang < 0] += 2 * np.pi
    ang /= 2 * np.pi
    mag = np.sqrt(flow[0, :, :] ** 2.0 + flow[1, :, :] ** 2.0)
    mag = np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0.0, 1.0)
    hfill_hsv = np.stack([ang * 180, mag * 255, np.ones_like(ang) * 255], 2).astype(
        np.uint8
    )
    flow_rgb = cv2.cvtColor(hfill_hsv, cv2.COLOR_HSV2RGB) / 255
    return np.transpose(flow_rgb, [2, 0, 1])


def image_to_display(
    image,
    cmap=cv2.COLORMAP_JET,
    order="CHW",
    normalize=False,
):
    """
    accepts a [1xHxW] or [CxHxW] float image with values in range [0,1]
    => change it range of (0, 255) for visualization

    Args:
        image: input image of size [C, H, W]
        cmap: color map [`cv2.COLORMAP_BONE`, `cv2.COLORMAP_JET`, `NORMAL` (no-processing)]. Defaults to cv2.COLORMAP_JET.
        order: 'CHW' or 'HWC'. Defaults to 'CHW'.
        normalize: if true, noramlize to [0, 1], otherwise clip to [0, 1]. Defaults to False.

    Returns:
        A visiable BGR image in range (0, 255), by default a colored heat map (in JET color map).
    """
    if order == "HWC" and len(image.shape) > 2:
        image = np.rollaxis(image, axis=2)
    image = np.squeeze(image)  # 2d or 3d

    if len(image.shape) == 3 and image.shape[0] == 2:
        image = convert_flow_for_display(image)

    if normalize:
        # handle nan pixels
        min_intensity = np.nanmin(image)
        max_intensity = np.nanmax(image)
        image = (image - min_intensity) / (max_intensity - min_intensity)
        image = np.uint8(image * 255)
    else:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

    if image.ndim == 3:
        if image.shape[0] == 3:
            image = np.transpose(image, [1, 2, 0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 2:
        if cmap == "NORMAL":
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.applyColorMap(image, cmap)
    return image


def create_mosaic(
    image_array,
    cmap=None,
    points=None,
    order="CHW",
    normalize=False,
):
    """Stich array of images into a big concatenated images

    Args:
        image_array: subimages to display as a 2D list (array), if in stretch 1D list, will be stretched back each element is an image of [D, H, W]
        cmap: list of color map => common: cv2.COLORMAP_BONE or cv2.COLORMAP_JET or 'NORMAL'. Defaults to None.
        points: corresponding points to display on all subimages. Defaults to None.
        order: order of the dimensions. Defaults to 'CHW'.
        normalize: if true, noramlize to [0, 1], otherwise clip to [0, 1]. Defaults to False.
    """
    batch_version = len(image_array[0].shape) == 4

    if not isinstance(image_array[0], list):  # if image_array is a stretch 1D list
        image_size = math.ceil(
            math.sqrt(len(image_array))
        )  # stretch back to 2D list [N by N]
        image_array = [
            image_array[i : min(i + image_size, len(image_array))]
            for i in range(0, len(image_array), image_size)
        ]

    max_cols = max(
        [len(row) for row in image_array]
    )  # because not every row (1st array) has the same size
    rows = []

    if cmap is None:
        cmap = [cv2.COLORMAP_JET]
    elif not isinstance(cmap, list):
        cmap = [cmap]

    if not isinstance(normalize, list):
        normalize = [normalize]

    if points is not None:
        if not isinstance(points, list):
            points = [points]

    i = 0
    for image_row in image_array:
        if len(image_row) == 0:
            continue
        image_row_processed = []
        for image in image_row:
            if torch.is_tensor(image):
                if batch_version:
                    image = image[0:1, :, :, :]
                if len(image.shape) == 4:  # [B. C, H, W]
                    image = image.squeeze(0)
                    if order == "CHW":
                        image = image.permute(1, 2, 0)  # [H, W, C]
                    if image.shape[2] not in (0, 3):  # sum all channel features
                        image = image.sum(dim=2)
                image = image.cpu().numpy()
            image_colorized = image_to_display(
                image, cmap[i % len(cmap)], order, normalize[i % len(normalize)]
            )
            if points is not None:
                image_colorized = visualize_matches_on_image(
                    image_colorized, points[i % len(points)]
                )
            image_row_processed.append(image_colorized)
            i += 1
        nimages = len(image_row_processed)
        if nimages < max_cols:  # padding zero(black) images in the empty areas
            image_row_processed += [np.zeros_like(image_row_processed[-1])] * (
                max_cols - nimages
            )
        rows.append(np.concatenate(image_row_processed, axis=1))  # horizontally
    return np.concatenate(rows, axis=0)  # vertically


def visualize_matches_on_image(image, matches):
    """Visualize correspondences on images

    Args:
        image: images
        matches: corresponding points of size [2, N]
    """
    num_matches = matches.shape[1]
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if torch.is_tensor(matches):
        matches = matches.detach().cpu().numpy()
    # just for visualization, round it:
    matches = matches.astype(int)
    output = image.copy()
    red = (0, 0, 255)
    alpha = 0.6
    radius = int(image.shape[1] / 64)  # should be 10 when the image width is 640
    for i in range(num_matches):
        image = cv2.circle(image, (matches[0, i], matches[1, i]), radius, red, -1)
    # blend
    output = cv2.addWeighted(image, alpha, output, 1 - alpha, 0)
    return output


def visualize_feature_channels(
    feat_map,
    rgb=None,
    points=None,
    order="CHW",
    add_ftr_avg=True,
    cmap=None,
):
    """Visualize every channels of the feature map

    Args:
        feat_map: feature map of size [B, H, W, C] or [B, C, H, W]
        rgb: original RGB image of size [B, H, W, 3] or [B, 3, H, W]. Defaults to None.
        points: correspondences of size [2, N]. Defaults to None.
        order: dimension order of the feature map and rgb image, either 'CHW' or 'HWC. Defaults to "CHW".
        add_ftr_avg: add averaged feature in the display grid. Defaults to True.
        cmap: color map. Defaults to None.

    Returns:
        _description_
    """
    assert len(feat_map.shape) == 4, "feature-map should be a 4-dim tensor"
    assert order in ["HWC", "CHW"]

    batch_version = feat_map.shape[0] != 1
    feat_map = feat_map.detach()
    if points is not None:
        points = points.detach()
    if not batch_version:
        feat_map = feat_map.squeeze(dim=0)
        if points is not None:
            points = points.squeeze()
    else:
        # if in batch, only visualize the 1st feature map
        feat_map = feat_map[0, :, :, :]
        if points is not None:
            points = points[0, :, :]

    if order == "CHW":
        feat_map = feat_map.permute(1, 2, 0)  # convert to [H, W, C]
    D = feat_map.shape[2]
    feat_map_sum = feat_map.sum(dim=2)

    if rgb is not None:
        if torch.is_tensor(rgb) and len(rgb.shape) == 4:
            rgb = rgb.detach()
            if not batch_version:
                rgb = rgb.squeeze()
            else:
                # if in batch, only visualize the 1st feature map
                rgb = rgb[0, :, :, :]
            rgb = rgb.permute(1, 2, 0)  # convert to [H, W, C]
        if add_ftr_avg:
            feat_map_channel_list = [rgb, feat_map_sum]
        else:
            feat_map_channel_list = [rgb]
    else:
        if add_ftr_avg:
            feat_map_channel_list = [feat_map_sum]
        else:
            feat_map_channel_list = []

    for d in range(D):
        feat_map_channel = feat_map[:, :, d]
        feat_map_channel_list.append(feat_map_channel)

    if cmap is not None:
        cmap = [cmap] * (D + 1)
    else:
        cmap = [cv2.COLORMAP_JET] * (D + 1)

    if rgb is not None:
        cmap = ["NORMAL"] + cmap
    feature_channels = create_mosaic(
        feat_map_channel_list,
        cmap=cmap,
        points=points,
        order="HWC",
        normalize=True,
    )
    return feature_channels
