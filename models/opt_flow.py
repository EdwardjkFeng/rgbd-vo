import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T


def compute_optical_flow(prev_img, next_img):
    """
    Compute dense optical flow using Farneback method.

    Parameters:
    - prev_img: Grayscale image at time t-1.
    - next_img: Grayscale image at time t.

    Returns:
    - flow: Optical flow.
    """
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow


def convert_flow_for_display(flow):
    """
    Converts a 2D image (e.g. flow) to bgr

    :param flow:
    :type flow: optical flow of size [2, H, W]
    :return:
    :rtype:
    """

    if torch.is_tensor(flow):
        flow = flow.detach().cpu().numpy()

    ang = np.arctan2(flow[1, :, :], flow[0, :, :])
    ang[ang < 0] += 2 * np.pi
    ang /= 2 * np.pi
    mag = np.sqrt(flow[0, :, :] ** 2. + flow[1, :, :] ** 2.)
    mag = np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0., 1.)
    hfill_hsv = np.stack([ang * 180, mag * 255, np.ones_like(ang) * 255], 2).astype(np.uint8)
    flow_rgb = cv2.cvtColor(hfill_hsv, cv2.COLOR_HSV2RGB)
    # return np.transpose(flow_rgb, [2, 0, 1])
    return flow_rgb


plt.rcParams["savefig.bbox"] = "tight"
# sphinx_gallery_thumbnail_number = 2


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            # T.Resize(size=(480, 640)),
        ]
    )
    batch = transforms(batch)
    return batch

if __name__ == '__main__':
    from dataset.tum_rgbd import TUM
    import argparse
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    import torchvision.utils as torch_utils
    from torchvision.utils import flow_to_image
    from torchvision.models.optical_flow import raft_large

    class Args():
        pass

    args = Args()
    args.conf = './config/f2f_dia.yaml'

    if args.conf:
        conf =OmegaConf.load(args.conf)

    loader = TUM(conf.data).get_dataset()

    torch_loader = DataLoader(
        loader,
        shuffle=False,
        num_workers=4,
    )

    # If you can, run this example on a GPU, it will be a lot faster.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # Deep optical flow model
    device = 'cuda'
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()

    with torch.no_grad():
        for batch in torch_loader:
            image1, image2, _, _, _, _ = batch['data']
            B, C, H, W = image1.shape

            img1_batch = torch.cat((image1, image2), dim=0)
            img2_batch = torch.cat((image2, image1), dim=0)
            plot(img1_batch)
            plot(img2_batch)
            plt.show()

            img1_batch = preprocess(img1_batch).to(device)
            img2_batch = preprocess(img2_batch).to(device)

            list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
            predicted_flows = list_of_flows[-1]
            print(f"dtype = {predicted_flows.dtype}")
            print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
            print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

            flow_imgs = flow_to_image(predicted_flows)

            # # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
            img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

            grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
            plot(grid)
            plt.show()

            # image1 = image1.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            # image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            # image2 = image2.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            # image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

            # flow = compute_optical_flow(image1, image2)
            # flow_img1 = flow_to_image(torch.from_numpy(np.transpose(flow, [2, 0, 1])))
            # flow = compute_optical_flow(image2, image1)
            # flow_img2 = flow_to_image(torch.from_numpy(np.transpose(flow, [2, 0, 1])))
            # flow_imgs = [flow_img1, flow_img2]
            # grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
            # plot(grid)
            # plt.show()
