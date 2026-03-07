#!/usr/bin/env python3
"""Convert GeoTIFF files in a demo folder to web-friendly formats.

Usage:
    python convert_tifs.py <tif_folder> [--out <output_folder>]

Expects the folder to contain:
    rgb.tif         - RGB satellite image  -> rgb.jpg
    costmap.tif     - Colored costmap      -> costmap.png
    costmap_gray.tif - Grayscale costmap   -> costmap_gray.png

If --out is not specified, outputs are written to the same folder as the TIFs.
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")


def convert(tif_folder: Path, out_folder: Path) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)

    conversions = [
        ("rgb.tif", "rgb.jpg", "JPEG", {"quality": 90}),
        ("costmap.tif", "costmap.png", "PNG", {}),
        ("costmap_gray.tif", "costmap_gray.png", "PNG", {}),
    ]

    for src_name, dst_name, fmt, save_kwargs in conversions:
        src = tif_folder / src_name
        dst = out_folder / dst_name

        if not src.exists():
            print(f"  SKIP  {src_name} (not found)")
            continue

        img = Image.open(src)

        # JPEG doesn't support alpha or palette modes
        if fmt == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        img.save(dst, fmt, **save_kwargs)

        src_kb = src.stat().st_size / 1024
        dst_kb = dst.stat().st_size / 1024
        print(f"  {src_name} ({src_kb:.0f} KB) -> {dst_name} ({dst_kb:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert demo TIFs to web formats")
    parser.add_argument("tif_folder", type=Path, help="Folder containing TIF files")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: same as tif_folder)")
    args = parser.parse_args()

    out = args.out or args.tif_folder
    print(f"Converting TIFs in {args.tif_folder} -> {out}")
    convert(args.tif_folder, out)
    print("Done.")


if __name__ == "__main__":
    main()
