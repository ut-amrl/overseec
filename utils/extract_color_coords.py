#!/usr/bin/env python3
"""
extract_color_coords.py

Scans a GeoTIFF for pixels that are a pure color (red, green, or blue) and
outputs a CSV of their pixel coordinates and corresponding UTM coordinates
derived from the GeoTIFF's spatial metadata.

Usage:
    python extract_color_coords.py <input.tif> [--color green] [--output coords.csv]

Arguments:
    input.tif         Path to the input GeoTIFF file.

Options:
    --color COLOR     Color to search for: red, green, or blue (default: green)
    --output FILE     Output CSV file path (default: <color>_pixels.csv)

Pure color definitions:
    red   -> RGB (255,   0,   0)
    green -> RGB (  0, 255,   0)
    blue  -> RGB (  0,   0, 255)

Requirements:
    pip install rasterio numpy pyproj
"""

import sys
import csv
import argparse
import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import CRS, Transformer


# Pure color targets: (R, G, B)
PURE_COLORS = {
    "red":   (255,   0,   0),
    "green": (  0, 255,   0),
    "blue":  (  0,   0, 255),
}


def find_pure_color_pixels(tif_path: str, color: str, output_csv: str) -> None:
    color = color.lower()
    if color not in PURE_COLORS:
        raise ValueError(f"Invalid color '{color}'. Choose from: {', '.join(PURE_COLORS)}")

    target_r, target_g, target_b = PURE_COLORS[color]

    with rasterio.open(tif_path) as src:
        if src.count < 3:
            raise ValueError(f"GeoTIFF has only {src.count} band(s); need at least 3 (RGB).")

        transform = src.transform
        crs = src.crs

        if crs is None:
            raise ValueError("GeoTIFF has no CRS defined. Cannot compute UTM coordinates.")

        # Read R, G, B bands (assumed to be bands 1, 2, 3)
        r = src.read(1).astype(np.uint8)
        g = src.read(2).astype(np.uint8)
        b = src.read(3).astype(np.uint8)

    # Find pixels matching the pure color exactly
    mask = (r == target_r) & (g == target_g) & (b == target_b)
    rows, cols = np.where(mask)

    if len(rows) == 0:
        print(f"No pure {color} pixels found in '{tif_path}'.")
        return

    print(f"Found {len(rows):,} pure {color} pixel(s). Converting coordinates...")

    # Convert pixel coordinates to the GeoTIFF's native CRS (x, y)
    # rasterio.transform.xy returns (x_list, y_list) for given (row, col) arrays
    native_x, native_y = xy(transform, rows, cols)
    native_x = np.array(native_x)
    native_y = np.array(native_y)

    # Determine if we need to reproject to UTM
    proj_crs = CRS.from_user_input(crs)

    if proj_crs.is_projected and "utm" in proj_crs.name.lower():
        # Already in UTM — use coordinates directly
        utm_x = native_x
        utm_y = native_y
        utm_zone = proj_crs.name
        print(f"CRS is already UTM: {utm_zone}")
    else:
        # Reproject from the native CRS to the best matching UTM zone
        # Use the centroid of found points to pick UTM zone
        if proj_crs.is_geographic:
            center_lon = float(np.mean(native_x))
            center_lat = float(np.mean(native_y))
        else:
            # Project centroid to WGS84 first to get lon/lat for zone selection
            to_wgs84 = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
            center_lon, center_lat = to_wgs84.transform(
                float(np.mean(native_x)), float(np.mean(native_y))
            )

        zone_number = int((center_lon + 180) / 6) + 1
        hemisphere = "north" if center_lat >= 0 else "south"
        utm_epsg = 32600 + zone_number if hemisphere == "north" else 32700 + zone_number
        utm_crs = CRS.from_epsg(utm_epsg)

        transformer = Transformer.from_crs(proj_crs, utm_crs, always_xy=True)
        utm_x, utm_y = transformer.transform(native_x, native_y)
        utm_zone = f"UTM Zone {zone_number}{hemisphere[0].upper()} (EPSG:{utm_epsg})"
        print(f"Reprojected to: {utm_zone}")

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pixel_col", "pixel_row", "utm_easting", "utm_northing"])
        for col, row, ex, ny in zip(cols, rows, utm_x, utm_y):
            writer.writerow([int(col), int(row), f"{ex:.3f}", f"{ny:.3f}"])

    print(f"CSV written to: {output_csv}")
    print(f"  Color searched: pure {color} {PURE_COLORS[color]}")
    print(f"  Pixels found:   {len(rows):,}")
    print(f"  UTM system:     {utm_zone}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract pixel + UTM coordinates of pure-color pixels from a GeoTIFF."
    )
    parser.add_argument("input", help="Path to the input GeoTIFF file")
    parser.add_argument(
        "--color",
        choices=["red", "green", "blue"],
        default="green",
        help="Pure color to search for (default: green)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file path (default: <color>_pixels.csv)",
    )

    args = parser.parse_args()

    output_csv = args.output if args.output else f"{args.color}_pixels.csv"

    find_pure_color_pixels(args.input, args.color, output_csv)


if __name__ == "__main__":
    main()
