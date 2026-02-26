#!/usr/bin/env python3
"""
validate_color_coords.py

Validates a CSV produced by extract_color_coords.py against the source GeoTIFF by:
  1. Checking that every (pixel_col, pixel_row) in the CSV is actually the expected
     pure color in the GeoTIFF.
  2. Checking that the UTM coordinates in the CSV match those derived from the
     GeoTIFF's spatial metadata (within a configurable tolerance).
  3. Optionally displaying the GeoTIFF with the CSV pixels drawn on top in a
     chosen color.

Usage:
    python validate_color_coords.py <input.tif> <coords.csv> [OPTIONS]

Options:
    --csv-color COLOR     Pure color that was searched for when building the CSV:
                          red, green, or blue (default: green)
    --display             Display the GeoTIFF with CSV pixels overlaid
    --draw-color COLOR    Color used to draw the overlay markers:
                          red, green, or blue (default: red)
                          [ignored if --display is not set]
    --marker-size INT     Radius in pixels of each overlay marker (default: 3)
    --tolerance FLOAT     Acceptable UTM coordinate mismatch in metres (default: 1.0)

Requirements:
    pip install rasterio numpy pyproj matplotlib
"""

import sys
import csv
import argparse
import numpy as np
import rasterio
from rasterio.transform import xy
from pyproj import CRS, Transformer
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── color definitions ──────────────────────────────────────────────────────────
PURE_COLORS = {
    "red":   (255,   0,   0),
    "green": (  0, 255,   0),
    "blue":  (  0,   0, 255),
}

# Matplotlib-compatible float colors for drawing
MPL_COLORS = {
    "red":   (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue":  (0.0, 0.0, 1.0),
}


# ── helpers ────────────────────────────────────────────────────────────────────

def build_utm_transformer(src_crs, sample_x, sample_y):
    """Return a Transformer to UTM and the UTM zone label."""
    proj_crs = CRS.from_user_input(src_crs)

    if proj_crs.is_projected and "utm" in proj_crs.name.lower():
        return None, proj_crs.name  # already UTM – no transform needed

    # Determine the appropriate UTM zone from the centroid of sample points
    if proj_crs.is_geographic:
        center_lon, center_lat = float(np.mean(sample_x)), float(np.mean(sample_y))
    else:
        to_wgs84 = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
        center_lon, center_lat = to_wgs84.transform(
            float(np.mean(sample_x)), float(np.mean(sample_y))
        )

    zone_number = int((center_lon + 180) / 6) + 1
    hemisphere  = "north" if center_lat >= 0 else "south"
    utm_epsg    = 32600 + zone_number if hemisphere == "north" else 32700 + zone_number
    utm_crs     = CRS.from_epsg(utm_epsg)
    transformer = Transformer.from_crs(proj_crs, utm_crs, always_xy=True)
    utm_label   = f"UTM Zone {zone_number}{hemisphere[0].upper()} (EPSG:{utm_epsg})"
    return transformer, utm_label


def to_display_rgb(tif_array: np.ndarray) -> np.ndarray:
    """
    Convert a (bands, H, W) uint8 array (at least 3 bands) into an (H, W, 3)
    uint8 RGB array suitable for imshow.
    """
    rgb = tif_array[:3].transpose(1, 2, 0).astype(np.uint8)
    return rgb


# ── main validation logic ──────────────────────────────────────────────────────

def validate(tif_path: str, csv_path: str, csv_color: str,
             tolerance: float, display: bool, draw_color: str,
             marker_size: int) -> None:

    csv_color  = csv_color.lower()
    draw_color = draw_color.lower()

    if csv_color not in PURE_COLORS:
        raise ValueError(f"--csv-color must be one of: {', '.join(PURE_COLORS)}")
    if draw_color not in PURE_COLORS:
        raise ValueError(f"--draw-color must be one of: {', '.join(PURE_COLORS)}")

    target_r, target_g, target_b = PURE_COLORS[csv_color]

    # ── 1. Load GeoTIFF ────────────────────────────────────────────────────────
    print(f"Opening GeoTIFF: {tif_path}")
    with rasterio.open(tif_path) as src:
        if src.count < 3:
            raise ValueError(f"GeoTIFF has only {src.count} band(s); need at least 3 (RGB).")
        transform = src.transform
        crs       = src.crs
        r_band    = src.read(1).astype(np.uint8)
        g_band    = src.read(2).astype(np.uint8)
        b_band    = src.read(3).astype(np.uint8)
        tif_data  = np.stack([r_band, g_band, b_band], axis=0)  # (3, H, W)
        height, width = r_band.shape

    if crs is None:
        raise ValueError("GeoTIFF has no CRS defined.")

    # ── 2. Load CSV ────────────────────────────────────────────────────────────
    print(f"Reading CSV:    {csv_path}")
    csv_rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        expected_fields = {"pixel_col", "pixel_row", "utm_easting", "utm_northing"}
        if not expected_fields.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV is missing required columns. Expected: {expected_fields}\n"
                f"Found: {set(reader.fieldnames or [])}"
            )
        for row in reader:
            csv_rows.append({
                "col":      int(row["pixel_col"]),
                "row":      int(row["pixel_row"]),
                "utm_e":    float(row["utm_easting"]),
                "utm_n":    float(row["utm_northing"]),
            })

    total = len(csv_rows)
    print(f"CSV entries:   {total:,}")

    if total == 0:
        print("CSV is empty – nothing to validate.")
        return

    # ── 3. Build UTM transformer using the CSV pixel centroids ─────────────────
    cols_arr = np.array([r["col"] for r in csv_rows])
    rows_arr = np.array([r["row"] for r in csv_rows])
    native_x_arr, native_y_arr = xy(transform, rows_arr, cols_arr)
    native_x_arr = np.array(native_x_arr)
    native_y_arr = np.array(native_y_arr)

    transformer, utm_label = build_utm_transformer(crs, native_x_arr, native_y_arr)
    print(f"UTM reference: {utm_label}")

    if transformer is not None:
        calc_e, calc_n = transformer.transform(native_x_arr, native_y_arr)
    else:
        calc_e, calc_n = native_x_arr, native_y_arr

    # ── 4. Run checks ──────────────────────────────────────────────────────────
    color_pass      = 0
    color_fail      = 0
    color_oob       = 0
    utm_pass        = 0
    utm_fail        = 0
    utm_errors      = []   # (col, row, delta_e, delta_n, distance)
    color_failures  = []   # (col, row, actual_rgb)

    csv_e_arr = np.array([r["utm_e"] for r in csv_rows])
    csv_n_arr = np.array([r["utm_n"] for r in csv_rows])

    # UTM check (vectorised)
    delta_e  = np.abs(calc_e - csv_e_arr)
    delta_n  = np.abs(calc_n - csv_n_arr)
    distance = np.sqrt(delta_e**2 + delta_n**2)
    utm_fail_mask = distance > tolerance
    utm_pass = int(np.sum(~utm_fail_mask))
    utm_fail = int(np.sum(utm_fail_mask))

    if utm_fail > 0:
        fail_indices = np.where(utm_fail_mask)[0]
        for i in fail_indices[:20]:   # report first 20
            utm_errors.append((
                int(cols_arr[i]), int(rows_arr[i]),
                float(delta_e[i]), float(delta_n[i]), float(distance[i])
            ))

    # Pixel color check (vectorised where possible)
    in_bounds_mask = (
        (rows_arr >= 0) & (rows_arr < height) &
        (cols_arr >= 0) & (cols_arr < width)
    )
    color_oob = int(np.sum(~in_bounds_mask))

    ib_rows = rows_arr[in_bounds_mask]
    ib_cols = cols_arr[in_bounds_mask]
    actual_r = r_band[ib_rows, ib_cols]
    actual_g = g_band[ib_rows, ib_cols]
    actual_b = b_band[ib_rows, ib_cols]

    match_mask = (
        (actual_r == target_r) &
        (actual_g == target_g) &
        (actual_b == target_b)
    )
    color_pass = int(np.sum(match_mask))
    color_fail_count = int(np.sum(~match_mask))

    if color_fail_count > 0:
        fail_indices = np.where(~match_mask)[0]
        for i in fail_indices[:20]:
            color_failures.append((
                int(ib_cols[i]), int(ib_rows[i]),
                (int(actual_r[i]), int(actual_g[i]), int(actual_b[i]))
            ))

    color_fail = color_fail_count + color_oob

    # ── 5. Report ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"Total CSV entries : {total:,}")
    print()
    print(f"[PIXEL COLOR CHECK]  target = pure {csv_color} {PURE_COLORS[csv_color]}")
    print(f"  Passed : {color_pass:,}")
    print(f"  Failed : {color_fail:,}  (out-of-bounds: {color_oob})")
    if color_failures:
        print(f"  First up to 20 mismatches (col, row, actual_rgb):")
        for c, r, rgb in color_failures:
            print(f"    col={c:5d}  row={r:5d}  actual={rgb}")

    print()
    print(f"[UTM COORDINATE CHECK]  tolerance = {tolerance} m")
    print(f"  Passed : {utm_pass:,}")
    print(f"  Failed : {utm_fail:,}")
    if utm_errors:
        print(f"  First up to 20 mismatches (col, row, Δeast, Δnorth, dist):")
        for c, r, de, dn, dist in utm_errors:
            print(f"    col={c:5d}  row={r:5d}  Δe={de:.4f}m  Δn={dn:.4f}m  dist={dist:.4f}m")

    print()
    overall = (color_fail == 0 and utm_fail == 0)
    status = "✅ ALL CHECKS PASSED" if overall else "❌ SOME CHECKS FAILED"
    print(status)
    print("=" * 60)

    # ── 6. Optional display ────────────────────────────────────────────────────
    if display:
        print()
        print(f"Rendering overlay (draw color: {draw_color})...")
        display_rgb = to_display_rgb(tif_data)  # (H, W, 3) uint8

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(display_rgb)

        mpl_color = MPL_COLORS[draw_color]

        # Draw each CSV point as a filled circle
        for col, row in zip(cols_arr, rows_arr):
            circle = plt.Circle(
                (col, row), radius=marker_size,
                color=mpl_color, fill=True, linewidth=0
            )
            ax.add_patch(circle)

        patch = mpatches.Patch(
            color=mpl_color,
            label=f"CSV pixels ({total:,} pts) – drawn as pure {draw_color}"
        )
        ax.legend(handles=[patch], loc="upper right", fontsize=9)
        ax.set_title(
            f"GeoTIFF with CSV overlay\n"
            f"Searched color: pure {csv_color}  |  "
            f"Color checks: {color_pass}/{total} passed  |  "
            f"UTM checks: {utm_pass}/{total} passed",
            fontsize=10
        )
        ax.set_xlabel("Pixel column")
        ax.set_ylabel("Pixel row")
        plt.tight_layout()
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a color-coordinate CSV against its source GeoTIFF, "
            "and optionally display the GeoTIFF with CSV pixels overlaid."
        )
    )
    parser.add_argument("input",  help="Path to the source GeoTIFF file")
    parser.add_argument("csv",    help="Path to the CSV produced by extract_color_coords.py")

    parser.add_argument(
        "--csv-color",
        choices=["red", "green", "blue"],
        default="green",
        help="Pure color that was extracted into the CSV (default: green)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Maximum acceptable UTM coordinate error in metres (default: 1.0)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the GeoTIFF with CSV pixels drawn on top",
    )
    parser.add_argument(
        "--draw-color",
        choices=["red", "green", "blue"],
        default="red",
        help="Color used for the overlay markers (default: red; ignored without --display)",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=3,
        help="Radius of each overlay marker in pixels (default: 3)",
    )

    args = parser.parse_args()

    validate(
        tif_path    = args.input,
        csv_path    = args.csv,
        csv_color   = args.csv_color,
        tolerance   = args.tolerance,
        display     = args.display,
        draw_color  = args.draw_color,
        marker_size = args.marker_size,
    )


if __name__ == "__main__":
    main()
