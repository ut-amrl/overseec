#!/usr/bin/env python3
"""Convert all GeoTIFF files in a demo folder to web-friendly formats.

Usage:
    python convert_tifs.py <tif_folder> [--out <output_folder>] [--max-size 1536]

Rules:
    rgb.tif              -> rgb.jpg          (JPEG, quality 90)
    <name>.tif           -> <name>.png       (PNG, color)
                         -> <name>_gray.png  (PNG, grayscale — used by planner)

If --out is not specified, outputs are written into the same folder as the TIFs.
"""

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install Pillow")


def convert_one(src: Path, out_folder: Path, max_size: int) -> None:
    img = Image.open(src)

    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    stem = src.stem  # filename without extension

    if stem == "rgb":
        # RGB satellite image -> JPEG
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        dst = out_folder / "rgb.jpg"
        img.save(dst, "JPEG", quality=90)
        print(f"  {src.name} ({src.stat().st_size // 1024} KB) -> {dst.name} ({dst.stat().st_size // 1024} KB)")
    else:
        # Costmap TIF -> color PNG + grayscale PNG
        dst_color = out_folder / f"{stem}.png"
        img.save(dst_color, "PNG")
        print(f"  {src.name} ({src.stat().st_size // 1024} KB) -> {dst_color.name} ({dst_color.stat().st_size // 1024} KB)")

        dst_gray = out_folder / f"{stem}_gray.png"
        img.convert("L").save(dst_gray, "PNG")
        print(f"  {src.name} -> {dst_gray.name} ({dst_gray.stat().st_size // 1024} KB)")


def convert(tif_folder: Path, out_folder: Path, max_size: int = 1536) -> None:
    out_folder.mkdir(parents=True, exist_ok=True)

    tifs = sorted(tif_folder.glob("*.tif")) + sorted(tif_folder.glob("*.tiff"))
    if not tifs:
        print("  No TIF files found.")
        return

    for src in tifs:
        convert_one(src, out_folder, max_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert all TIFs in a demo folder to web formats")
    parser.add_argument("tif_folder", type=Path, help="Folder containing TIF files")
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: same as tif_folder)")
    parser.add_argument("--max-size", type=int, default=1536, help="Max pixel dimension for longest side (default: 1536)")
    args = parser.parse_args()

    out = args.out or args.tif_folder
    print(f"Converting TIFs in {args.tif_folder} -> {out} (max size: {args.max_size})")
    convert(args.tif_folder, out, args.max_size)
    print("Done.")


if __name__ == "__main__":
    main()
