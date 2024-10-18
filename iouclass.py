import os
import numpy as np
from PIL import Image


def read_masks(path):
    """Read mask from the given path and convert it to a numpy array."""
    mask = Image.open(path)
    mask = np.array(mask)
    return mask


# Replace submit_dir with your result path here
submit_dir = './predictions'

# Replace truth_dir with the ground-truth path here
truth_dir = './dataset/val/val_mask'

# Replace output_dir with the desired output path, and you will find 'scores.txt' containing the calculated mIoU
output_dir = '.'

if not os.path.isdir(submit_dir):
    print(f"{submit_dir} doesn't exist")

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submit_dir_list = os.listdir(submit_dir)
    if len(submit_dir_list) == 1:
        submit_dir = os.path.join(submit_dir, "%s" % submit_dir_list[0])
        assert os.path.isdir(submit_dir)

    intersection_all = np.zeros(19)  # Intersection for each class
    union_all = np.zeros(19)         # Union for each class

    for idx in range(1000):
        pred_mask = read_masks(os.path.join(submit_dir, f"{idx}.png"))
        gt_mask = read_masks(os.path.join(truth_dir, f"{idx}.png"))

        for cls_idx in range(19):
            intersection = np.sum((pred_mask == cls_idx) & (gt_mask == cls_idx))

            union = np.sum((pred_mask == cls_idx) | (gt_mask == cls_idx))

            intersection_all[cls_idx] += intersection
            union_all[cls_idx] += union

    iou_all = intersection_all / (union_all + 1e-6)

    miou = np.mean(iou_all)

    output_filename = os.path.join(output_dir, 'iou_scores.txt')
    with open(output_filename, 'w') as f:
        for cls_idx in range(19):
            f.write(f"Class {cls_idx}: IoU = {iou_all[cls_idx]:.4f}\n")
        f.write(f"\nMean IoU (mIoU, average across all classes): {miou:.4f}\n")

    print(f"Mean IoU: {miou:.4f}")
