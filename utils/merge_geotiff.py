#!/usr/bin/env python3
"""
merge_geotiff.py

Takes a GeoTIFF (for its spatial metadata) and another image (for its RGB values),
and outputs a GeoTIFF combining the metadata from the first with the pixels from the second.

Usage:
    python merge_geotiff.py <reference.tif> <rgb_image> <output.tif>

Requirements:
    pip install rasterio Pillow numpy
"""

import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image


def merge_geotiff(reference_tif: str, rgb_image_path: str, output_path: str) -> None:
    # Open the reference GeoTIFF to extract metadata
    with rasterio.open(reference_tif) as ref:
        ref_meta = ref.meta.copy()
        ref_width = ref.width
        ref_height = ref.height

    # Open the RGB image
    img = Image.open(rgb_image_path).convert("RGB")
    img_width, img_height = img.size

    # Validate dimensions match
    if img_width != ref_width or img_height != ref_height:
        raise ValueError(
            f"Dimension mismatch: GeoTIFF is {ref_width}x{ref_height}, "
            f"but the RGB image is {img_width}x{img_height}."
        )

    # Convert image to numpy array: shape (height, width, 3)
    rgb_array = np.array(img)

    # Update metadata for a 3-band uint8 RGB output
    ref_meta.update({
        "count": 3,
        "dtype": "uint8",
        "driver": "GTiff",
    })

    # Write the output GeoTIFF
    with rasterio.open(output_path, "w", **ref_meta) as dst:
        # rasterio expects bands in shape (band, height, width)
        for band_idx in range(3):
            dst.write(rgb_array[:, :, band_idx], band_idx + 1)

        # Copy color interpretation tags if supported
        dst.update_tags(ns="rio_overview", resampling="nearest")

    print(f"Done! Output written to: {output_path}")
    print(f"  CRS:       {ref_meta.get('crs')}")
    print(f"  Transform: {ref_meta.get('transform')}")
    print(f"  Size:      {ref_width} x {ref_height}")
    print(f"  Bands:     3 (RGB)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_geotiff.py <reference.tif> <rgb_image> <output.tif>")
        sys.exit(1)

    reference_tif = sys.argv[1]
    rgb_image_path = sys.argv[2]
    output_path = sys.argv[3]

    merge_geotiff(reference_tif, rgb_image_path, output_path)
