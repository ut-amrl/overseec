import numpy as np
import cv2
from scipy.spatial import KDTree

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def renormalize_and_resize(images, preds_shape, renorm = True):

    images = images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format
    
    # if renorm:
    if renorm:
        images = (images * STD + MEAN) * 255  # Reverse normalization
    # else:
        # images = images * 255
    images = np.clip(images, 0, 255).astype(np.uint8)  # Ensure valid pixel range

    # Resize images to match preds_np shape
    B, H_out, W_out, _ = preds_shape  # Get output shape from preds_np
    resized_images = np.stack([cv2.resize(img, (W_out, H_out)) for img in images], axis=0)

    return resized_images


def unified_directed_hausdorff(method_mask, target_masks):
    """
    Compute a unified directed Hausdorff distance from method_mask to the union of all target masks.
    Args:
        method_mask (np.ndarray): Binary mask of method (path = 1 or 255).
        target_masks (List[np.ndarray]): List of 2D binary masks.

    Returns:
        float: Directed Hausdorff distance from method to all targets.
    """
    method_pts = np.column_stack(np.where(method_mask > 0))
    if len(method_pts) == 0:
        raise ValueError("Method mask has no path pixels.")

    # Combine all target points into one array
    target_pts = np.vstack([
        np.column_stack(np.where(mask > 0))
        for mask in target_masks
        if np.any(mask > 0)
    ])

    if len(target_pts) == 0:
        raise ValueError("All target masks are empty.")

    # Build a single KDTree from the union of all target points
    tree = KDTree(target_pts)

    # Query nearest distance for each method point
    dists, _ = tree.query(method_pts)

    # Return max of those minimum distances (directed Hausdorff)
    return float(np.max(dists))