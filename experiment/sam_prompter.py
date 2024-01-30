import matplotlib.pyplot as plt
import torch
from torch.nn.functional import pad
# from kornia.contrib.image_prompter import ImagePrompter
from kornia.contrib.visual_prompter import VisualPrompter
from kornia.contrib.models.sam import SamConfig, Sam
from kornia.contrib.models import SegmentationResults
from kornia.geometry.boxes import Boxes
from kornia.geometry.keypoints import Keypoints
from kornia.geometry import resize
from kornia.enhance import normalize
from kornia.utils import get_cuda_device_if_available, tensor_to_image

from dataset.dataloader import load_data

models = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

device = get_cuda_device_if_available()
print(device)



# Utilities functions
def colorize_masks(binary_masks: torch.Tensor, merge: bool = True, alpha: None | float = None) -> list[torch.Tensor]:
    """Convert binary masks (B, C, H, W), boolean tensors, into masks with colors (B, (3, 4) , H, W) - RGB or RGBA. Where C refers to the number of masks.
    Args:
        binary_masks: a batched boolean tensor (B, C, H, W)
        merge: If true, will join the batch dimension into a unique mask.
        alpha: alpha channel value. If None, will generate RGB images

    Returns:
        A list of `C` colored masks.
    """
    B, C, H, W = binary_masks.shape
    OUT_C = 4 if alpha else 3

    output_masks = []

    for idx in range(C):
        _out = torch.zeros(B, OUT_C, H, W, device=binary_masks.device, dtype=torch.float32)
        for b in range(B):
            color = torch.rand(1, 3, 1, 1, device=binary_masks.device, dtype=torch.float32)
            if alpha:
                color = torch.cat([color, torch.tensor([[[[alpha]]]], device=binary_masks.device, dtype=torch.float32)], dim=1)

            to_colorize = binary_masks[b, idx, ...].view(1, 1, H, W).repeat(1, OUT_C, 1, 1)
            _out[b, ...] = torch.where(to_colorize, color, _out[b, ...])
        output_masks.append(_out)

    if merge:
        output_masks = [c.max(dim=0)[0] for c in output_masks]

    return output_masks


def show_binary_masks(binary_masks: torch.Tensor, axes) -> None:
    """plot binary masks, with shape (B, C, H, W), where C refers to the number of masks.

    will merge the `B` channel into a unique mask.
    Args:
        binary_masks: a batched boolean tensor (B, C, H, W)
        ax: a list of matplotlib axes with lenght of C
    """
    colored_masks = colorize_masks(binary_masks, True, 0.6)

    for ax, mask in zip(axes, colored_masks):
        ax.imshow(tensor_to_image(mask))


def show_boxes(boxes: Boxes, ax) -> None:
    boxes_tensor = boxes.to_tensor(mode="xywh").detach().cpu().numpy()
    for batched_boxes in boxes_tensor:
        for box in batched_boxes:
            x0, y0, w, h = box
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="orange", facecolor=(0, 0, 0, 0), lw=2))


def show_points(points: tuple[Keypoints, torch.Tensor], ax, marker_size=200):
    coords, labels = points
    pos_points = coords[labels == 1].to_tensor().detach().cpu().numpy()
    neg_points = coords[labels == 0].to_tensor().detach().cpu().numpy()

    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="+", s=marker_size, linewidth=2)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="x", s=marker_size, linewidth=2)


def show_image(image: torch.Tensor):
    plt.imshow(tensor_to_image(image))
    plt.axis("off")
    plt.show()


def show_predictions(
    image: torch.Tensor,
    predictions: SegmentationResults,
    points: tuple[Keypoints, torch.Tensor] | None = None,
    boxes: Boxes | None = None,
) -> None:
    n_masks = predictions.logits.shape[1]

    fig, axes = plt.subplots(1, n_masks, figsize=(21, 16))
    axes = [axes] if n_masks == 1 else axes

    for idx, ax in enumerate(axes):
        score = predictions.scores[:, idx, ...].mean()
        ax.imshow(tensor_to_image(image))
        ax.set_title(f"Mask {idx+1}, Score: {score:.3f}", fontsize=18)

        if points:
            show_points(points, ax)

        if boxes:
            show_boxes(boxes, ax)

        ax.axis("off")

    show_binary_masks(predictions.binary_masks, axes)
    plt.show()


def load_image(seq="rgbd_bonn_balloon_tracking", index=300):
    conf = {
        "category": "full",
        "keyframes": [1],
        "select_traj": seq,
        "truncate_depth": False,
        "grayscale": False,
        "resize": 0.25,
    }
    dataset = load_data(dataset_name="Bonn_RGBD", conf=conf)
    batch = dataset[index]
    rgb0, *_ = batch["data"]
    return rgb0



def main():
    model_type = "mobile_sam"
    samconfig = SamConfig(model_type, pretrained=True)

    # prompter = ImagePrompter(samconfig, device=device)
    prompter = VisualPrompter(samconfig, device=device)

    rgb = load_image() # 3 x H x W, numpy
    rgb_t = torch.from_numpy(rgb).to(device)
    show_image(rgb_t)

    prompter.set_image(rgb_t)
    print(prompter.is_image_set)

    # Keypoints 
    Keypoints_tensor = torch.tensor([[[104, 103], [125, 80], [105, 47]]], device=device, dtype=torch.float32)
    keypoints = Keypoints(Keypoints_tensor)
    labels = torch.tensor([[1, 1, 1]], device=device, dtype=torch.float32)

    pred_by_kp = prompter.predict(keypoints, labels, multimask_output=False)

    for i in range(pred_by_kp.binary_masks.shape[1]):
        show_image(pred_by_kp.binary_masks[:, i])


    k = 2

    for idx in range(max(keypoints.data.size(1), k)):
        print("-" * 79, f"\nQuery {idx}:")
        _kpts = keypoints[:, idx, ...][None, ...]
        _lbl = labels[:, idx, ...][None, ...]

        predictions = prompter.predict(keypoints=_kpts, keypoints_labels=_lbl)

        show_predictions(rgb_t, predictions, points=(_kpts, _lbl))


if __name__ == "__main__":
    main()