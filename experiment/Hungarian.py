import torch

import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    bbox1, bbox2: Arrays or lists in the format [x1, y1, x2, y2],
    where (x1, y1) is the top-left corner, and (x2, y2) is the bottom-right corner.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = bbox1_area + bbox2_area - inter_area

    iou = inter_area / union_area
    return iou

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_bounding_boxes_with_ids(image, bboxes, ids):
    """
    Draws bounding boxes with assigned IDs on the image.

    Parameters:
    image: The image on which to draw (as a NumPy array).
    bboxes: A list of bounding boxes in the format [x1, y1, x2, y2].
    ids: A list of IDs corresponding to each bounding box.
    """
    assert len(bboxes) == len(ids), "The number of bounding boxes must match the number of IDs."

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox, id in zip(bboxes, ids):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f"ID: {id}", color='white', fontsize=12, verticalalignment='top', bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

# Example usage:
# Assuming 'image' is a NumPy array representing your image,
# 'detections' are your bounding boxes, and 'row_ind', 'col_ind' are from the Hungarian algorithm output:


# Mock data: bounding boxes for existing tracks and new detections
# Format: [x1, y1, x2, y2]
tracks = [[10, 10, 50, 50], [60, 60, 100, 100]]  # Example existing tracks
detections = [[15, 15, 55, 55], [65, 65, 105, 105]]  # Example new detections

# Compute distance (1 - IoU) matrix
distance_matrix = np.zeros((len(tracks), len(detections)))
for i, track in enumerate(tracks):
    for j, detection in enumerate(detections):
        distance_matrix[i, j] = 1 - iou(track, detection)

# Apply the Hungarian algorithm
row_ind, col_ind = linear_sum_assignment(distance_matrix)

assigned_ids = [f"T{r}-D{c}" for r, c in zip(row_ind, col_ind)]
image = np.ones((400, 500))
show_bounding_boxes_with_ids(image, detections, assigned_ids)

# Print the assignments
for r, c in zip(row_ind, col_ind):
    print(f"Track {r} assigned to Detection {c}")
