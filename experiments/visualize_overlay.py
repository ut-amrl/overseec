#!/usr/bin/env python3
"""
Overlay human (blue) and OverSeeC (red) trajectories on the RGB satellite image.

Usage:
    python experiments/visualize_overlay.py \
        --human experiments/human-baseline/onion-creek/human/quattro/oc_mission1.tiff \
        --overseec experiments/overseec/onion-creek/oc_mission1/plan_on_white.png \
        --rgb experiments/overseec/onion-creek/oc_mission1/plan_on_rgb.png \
        --output overlay_oc_mission1.png

    If --rgb is not given, uses plan_on_rgb.png from the same folder as --overseec.
"""

import os
import argparse
import numpy as np
from PIL import Image
import cv2


def extract_red_mask(img_np):
    """Pure red: R >= 250, G < 50, B < 50."""
    return ((img_np[:, :, 0] >= 250) & (img_np[:, :, 1] < 50) & (img_np[:, :, 2] < 50)).astype(np.uint8)


def extract_green_mask(img_np):
    """Pure green: G >= 250, R < 50, B < 50."""
    return ((img_np[:, :, 1] >= 250) & (img_np[:, :, 0] < 50) & (img_np[:, :, 2] < 50)).astype(np.uint8)


def thicken_mask(mask, thickness=5):
    """Dilate a binary mask to thicken the line."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    return cv2.dilate(mask, kernel, iterations=1)


def main():
    parser = argparse.ArgumentParser(description="Overlay human + OverSeeC paths on RGB")
    parser.add_argument("--human", required=True, help="Path to human trajectory TIFF (blue line)")
    parser.add_argument("--overseec", required=True, help="Path to OverSeeC plan (red on white, e.g. plan_on_white.png)")
    parser.add_argument("--rgb", default=None, help="Path to RGB satellite image (default: plan_on_rgb.png next to --overseec)")
    parser.add_argument("--output", default="overlay.png", help="Output image path")
    parser.add_argument("--thickness", type=int, default=7, help="Line thickness in pixels (default: 7)")
    parser.add_argument("--white-alpha", type=float, default=0.4, help="White overlay opacity 0-1 (default: 0.4)")
    args = parser.parse_args()

    # Resolve RGB path
    rgb_path = args.rgb or os.path.join(os.path.dirname(args.overseec), "plan_on_rgb.png")

    # Load images
    human_img = np.array(Image.open(args.human).convert("RGB"))
    overseec_img = np.array(Image.open(args.overseec).convert("RGB"))
    rgb_img = np.array(Image.open(rgb_path).convert("RGB"))

    # Extract masks
    green_mask = extract_green_mask(human_img)
    red_mask = extract_red_mask(overseec_img)

    # Thicken
    green_thick = thicken_mask(green_mask, args.thickness)
    red_thick = thicken_mask(red_mask, args.thickness)

    # Start with RGB, apply white transparent layer for visibility
    canvas = rgb_img.copy().astype(np.float64)
    white = np.ones_like(canvas) * 255.0
    canvas = canvas * (1 - args.white_alpha) + white * args.white_alpha
    canvas = canvas.astype(np.uint8)

    # Draw OverSeeC (red) first, then human (blue) on top
    canvas[red_thick > 0] = [255, 0, 0]
    canvas[green_thick > 0] = [0, 0, 255]

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    Image.fromarray(canvas).save(args.output)
    print(f"Saved overlay to {args.output}")
    print(f"  RGB base   : {rgb_path}")
    print(f"  Human      : {args.human}  (blue, {np.count_nonzero(green_mask)} px)")
    print(f"  OverSeeC   : {args.overseec}  (red, {np.count_nonzero(red_mask)} px)")
    print(f"  Thickness  : {args.thickness} px")
    print(f"  White alpha: {args.white_alpha}")


if __name__ == "__main__":
    main()
