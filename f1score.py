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

    # Initialize arrays to store the counts for Precision, Recall, and F1 calculation
    tp_all = np.zeros(19)  # True positives
    fp_all = np.zeros(19)  # False positives
    fn_all = np.zeros(19)  # False negatives

    for idx in range(1000):
        # Load predicted and ground truth masks
        pred_mask = read_masks(os.path.join(submit_dir, f"{idx}.png"))
        gt_mask = read_masks(os.path.join(truth_dir, f"{idx}.png"))

        for cls_idx in range(19):
            # True Positive (TP): predicted and ground truth are both the class
            tp = np.sum((pred_mask == cls_idx) & (gt_mask == cls_idx))

            # False Positive (FP): predicted is the class, ground truth is not
            fp = np.sum((pred_mask == cls_idx) & (gt_mask != cls_idx))

            # False Negative (FN): ground truth is the class, predicted is not
            fn = np.sum((pred_mask != cls_idx) & (gt_mask == cls_idx))

            # Accumulate TP, FP, FN for each class
            tp_all[cls_idx] += tp
            fp_all[cls_idx] += fp
            fn_all[cls_idx] += fn

    # Now calculate Precision, Recall, and F1 score for each class
    precision_all = tp_all / (tp_all + fp_all + 1e-6)  # Avoid division by zero
    recall_all = tp_all / (tp_all + fn_all + 1e-6)

    f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all + 1e-6)

    # Calculate Macro (average) Precision, Recall, and F1-score
    precision_macro = np.mean(precision_all)
    recall_macro = np.mean(recall_all)
    f1_macro = np.mean(f1_all)

    # Output Precision, Recall, and F1 scores to a text file
    output_filename = os.path.join(output_dir, 'scores_with_f1_precision_recall.txt')
    with open(output_filename, 'w') as f:
        for cls_idx in range(19):
            f.write(f"Class {cls_idx}: Precision = {precision_all[cls_idx]:.4f}, "
                    f"Recall = {recall_all[cls_idx]:.4f}, F1-score = {f1_all[cls_idx]:.4f}\n")
        f.write(f"\nMacro Precision (average across all classes): {precision_macro:.4f}\n")
        f.write(f"Macro Recall (average across all classes): {recall_macro:.4f}\n")
        f.write(f"Macro F1-score (average across all classes): {f1_macro:.4f}\n")

    # Print the Macro scores
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")
