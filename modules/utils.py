import numpy as np
import cv2

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