#!/usr/bin/env python3
"""
Compute modified Hausdorff distance between OverSeeC planned paths and human baselines.

Usage:
    python experiments/compute_hd.py
    python experiments/compute_hd.py --human-names quattro alice bob --output results.txt
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# Add project root to path so we can import modules.utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modules.utils import unified_directed_hausdorff


# ── Area / mission configuration ──────────────────────────────────────────────
AREAS = [
    {"area_dir": "onion-creek",  "prefix": "oc", "overseec_missions": range(1, 6), "human_missions": range(1, 5)},
    {"area_dir": "pickle-north", "prefix": "pn", "overseec_missions": range(1, 5), "human_missions": range(1, 5)},
    {"area_dir": "pickle-south", "prefix": "ps", "overseec_missions": range(1, 5), "human_missions": range(1, 5)},
]


# ── Path extraction helpers ───────────────────────────────────────────────────

def extract_red_mask(image_path):
    """Extract pure-red path from an OverSeeC plan image (plan_on_white.png).
    Pure red: R >= 250, G < 50, B < 50.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    mask = (img[:, :, 0] >= 250) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50)
    return mask.astype(np.uint8)


def extract_green_mask(image_path):
    """Extract pure-green path from a human trajectory TIFF.
    Pure green: G >= 250, R < 50, B < 50.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    mask = (img[:, :, 1] >= 250) & (img[:, :, 0] < 50) & (img[:, :, 2] < 50)
    return mask.astype(np.uint8)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(human_baseline_dir, overseec_dir, human_names, output_path):
    results = []

    for area in AREAS:
        area_dir = area["area_dir"]
        prefix = area["prefix"]

        for mission_num in area["overseec_missions"]:
            mission_name = f"{prefix}_mission{mission_num}"

            # OverSeeC plan image (plan_on_white is cleanest for extraction)
            overseec_plan = os.path.join(overseec_dir, area_dir, mission_name, "plan_on_white.png")
            if not os.path.exists(overseec_plan):
                print(f"  [SKIP] OverSeeC plan not found: {overseec_plan}")
                continue

            overseec_mask = extract_red_mask(overseec_plan)
            if not np.any(overseec_mask):
                print(f"  [WARN] No red pixels found in {overseec_plan}")
                continue

            # Iterate over each human annotator
            for human_name in human_names:
                human_tiff = os.path.join(
                    human_baseline_dir, area_dir, "human", human_name,
                    f"{prefix}_mission{mission_num}.tiff"
                )
                if not os.path.exists(human_tiff):
                    # Mission may not exist for this human (e.g., oc has 5 overseec missions but only 4 human)
                    results.append({
                        "area": area_dir,
                        "mission": mission_name,
                        "human": human_name,
                        "hd": "N/A",
                        "note": "human file missing",
                    })
                    continue

                human_mask = extract_green_mask(human_tiff)
                if not np.any(human_mask):
                    results.append({
                        "area": area_dir,
                        "mission": mission_name,
                        "human": human_name,
                        "hd": "N/A",
                        "note": "no green pixels in human tiff",
                    })
                    continue

                # Compute HD: OverSeeC -> Human
                hd = unified_directed_hausdorff(overseec_mask, [human_mask])

                results.append({
                    "area": area_dir,
                    "mission": mission_name,
                    "human": human_name,
                    "hd": f"{hd:.2f}",
                    "note": "",
                })

    # ── Format and print table ────────────────────────────────────────────────
    header = f"{'Area':<16} {'Mission':<16} {'Human':<12} {'HD':<12} {'Note'}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    current_area = None
    for r in results:
        area_label = r["area"] if r["area"] != current_area else ""
        current_area = r["area"]
        lines.append(
            f"{area_label:<16} {r['mission']:<16} {r['human']:<12} "
            f"{r['hd']:<12} {r['note']}"
        )
        # Add a blank separator between areas
        if r == results[-1] or results[results.index(r) + 1]["area"] != current_area:
            lines.append(sep)

    table_str = "\n".join(lines)
    print(table_str)

    # ── Generate LaTeX table ──────────────────────────────────────────────────
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{|l|l|r|}",
        r"\hline",
        r"Area & Mission & HD \\",
        r"\hline",
    ]
    current_area = None
    for r in results:
        area_label = r["area"] if r["area"] != current_area else ""
        current_area = r["area"]
        hd_val = r["hd"] if r["hd"] != "N/A" else "--"
        latex_lines.append(f"{area_label} & {r['mission']} & {hd_val} \\\\")
        latex_lines.append(r"\hline")
    latex_lines.extend([
        r"\end{tabular}",
        r"\caption{Modified Hausdorff Distance: OverSeeC vs Human Baselines}",
        r"\label{tab:hd_results}",
        r"\end{table}",
    ])
    latex_str = "\n".join(latex_lines)
    print("\n\n" + latex_str)

    # Save to file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table_str + "\n\n\n" + latex_str + "\n")
    print(f"\nResults saved to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute Hausdorff distance: OverSeeC vs Human baselines")
    parser.add_argument("--human-baseline-dir", default=None,
                        help="Path to human-baseline root (default: experiments/human-baseline)")
    parser.add_argument("--overseec-dir", default=None,
                        help="Path to overseec root (default: experiments/overseec)")
    parser.add_argument("--human-names", nargs="+", default=["quattro"],
                        help="List of human annotator names (default: quattro)")
    parser.add_argument("--output", default=None,
                        help="Output txt file (default: experiments/hd_results.txt)")
    args = parser.parse_args()

    # Resolve defaults relative to the experiments dir
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    human_baseline_dir = args.human_baseline_dir or os.path.join(exp_dir, "human-baseline")
    overseec_dir = args.overseec_dir or os.path.join(exp_dir, "overseec")
    output_path = args.output or os.path.join(exp_dir, "hd_results.txt")

    print(f"Human baseline dir : {human_baseline_dir}")
    print(f"OverSeeC dir       : {overseec_dir}")
    print(f"Human annotators   : {args.human_names}")
    print(f"Output file        : {output_path}")
    print()

    run_pipeline(human_baseline_dir, overseec_dir, args.human_names, output_path)


if __name__ == "__main__":
    main()
