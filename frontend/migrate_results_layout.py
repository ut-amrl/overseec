#!/usr/bin/env python3
"""
Migrate results/ from the legacy layout:

  results/<tiff>/
    original.tif
    temp_latest/
    <YYYY-MM-DD>/<run>/(refined|semantic|costmap/)

to the newer layout:

  results/<tiff>/
    original.tif
    mask/
      temp_latest/
      <YYYY-MM-DD>/<run>/(refined|semantic)
    costmap/
      temp_latest/(costmap.png, costmap_bw.png)
      <YYYY-MM-DD>/<run>/(costmap.png, costmap_bw.png)

Default mode is dry-run. Use --apply to perform changes.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Action:
    kind: str
    src: Path | None = None
    dst: Path | None = None
    note: str | None = None


def is_tiff_result_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "original.tif").exists()


def safe_mkdir(path: Path, apply: bool, actions: list[Action]) -> None:
    if path.exists():
        return
    actions.append(Action("mkdir", dst=path))
    if apply:
        path.mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dst: Path, apply: bool, actions: list[Action]) -> None:
    actions.append(Action("move", src=src, dst=dst))
    if not apply:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def safe_rmdir_if_empty(path: Path, apply: bool, actions: list[Action]) -> None:
    if not path.exists() or not path.is_dir():
        return
    try:
        next(path.iterdir())
    except StopIteration:
        actions.append(Action("rmdir", dst=path))
        if apply:
            path.rmdir()


def flatten_costmap_dir(run_dir: Path, apply: bool, actions: list[Action]) -> None:
    """
    Legacy costmaps may be stored as <run_dir>/costmap/costmap.png.
    This flattens them to <run_dir>/costmap.png, <run_dir>/costmap_bw.png.
    """
    legacy = run_dir / "costmap"
    if not legacy.exists() or not legacy.is_dir():
        return

    for name in ("costmap.png", "costmap_bw.png"):
        src = legacy / name
        if src.exists():
            safe_move(src, run_dir / name, apply, actions)

    safe_rmdir_if_empty(legacy, apply, actions)

def extract_legacy_costmap_subdir(src_run_dir: Path, cost_dst: Path, apply: bool, actions: list[Action]) -> None:
    """
    If src_run_dir contains a legacy 'costmap/' subdir, move costmap.png (+ costmap_bw.png)
    into cost_dst and remove the legacy subdir if empty.
    """
    legacy = src_run_dir / "costmap"
    if not legacy.exists() or not legacy.is_dir():
        return

    safe_mkdir(cost_dst, apply, actions)

    for name in ("costmap.png", "costmap_bw.png"):
        p = legacy / name
        if p.exists():
            safe_move(p, cost_dst / name, apply, actions)

    safe_rmdir_if_empty(legacy, apply, actions)


def migrate_one_tiff_dir(tiff_dir: Path, apply: bool, actions: list[Action]) -> None:
    mask_root = tiff_dir / "mask"
    costmap_root = tiff_dir / "costmap"

    safe_mkdir(mask_root, apply, actions)
    safe_mkdir(costmap_root, apply, actions)

    # Move temp_latest -> mask/temp_latest
    legacy_temp_latest = tiff_dir / "temp_latest"
    new_temp_latest = mask_root / "temp_latest"
    if legacy_temp_latest.exists() and legacy_temp_latest.is_dir():
        if new_temp_latest.exists():
            actions.append(
                Action(
                    "skip",
                    src=legacy_temp_latest,
                    note="mask/temp_latest already exists; leaving legacy temp_latest in place",
                )
            )
        else:
            safe_move(legacy_temp_latest, new_temp_latest, apply, actions)

    # Move legacy date folders
    for date_dir in sorted(tiff_dir.iterdir()):
        if not date_dir.is_dir() or not DATE_RE.match(date_dir.name):
            continue

        for run_dir in sorted(date_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            has_masks = (run_dir / "refined").is_dir() or (run_dir / "semantic").is_dir()
            has_costmap = (run_dir / "costmap").is_dir()

            if not has_masks and not has_costmap:
                continue

            if has_masks:
                dst = mask_root / date_dir.name / run_dir.name
                safe_move(run_dir, dst, apply, actions)

                # If there was an embedded legacy costmap folder, move it into costmap/<date>/<run>/.
                cost_dst = costmap_root / date_dir.name / run_dir.name
                extract_legacy_costmap_subdir(dst, cost_dst, apply, actions)
            else:
                # costmap-only run
                dst = costmap_root / date_dir.name / run_dir.name
                safe_move(run_dir, dst, apply, actions)
                flatten_costmap_dir(dst, apply, actions)

        safe_rmdir_if_empty(date_dir, apply, actions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Path to the frontend/results directory",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform moves (default: dry-run)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    apply = bool(args.apply)

    if not results_dir.exists() or not results_dir.is_dir():
        raise SystemExit(f"results dir not found: {results_dir}")

    actions: list[Action] = []
    for entry in sorted(results_dir.iterdir()):
        if entry.name in {"final_costmaps", "temp_downloads"}:
            continue
        if is_tiff_result_dir(entry):
            migrate_one_tiff_dir(entry, apply, actions)

    for a in actions:
        if a.kind == "mkdir":
            print(f"mkdir {a.dst}")
        elif a.kind == "move":
            print(f"move  {a.src} -> {a.dst}")
        elif a.kind == "rmdir":
            print(f"rmdir {a.dst}")
        elif a.kind == "skip":
            msg = f"skip  {a.src}"
            if a.note:
                msg += f"  ({a.note})"
            print(msg)
        elif a.kind == "note":
            msg = "note"
            if a.dst:
                msg += f"  {a.dst}"
            if a.note:
                msg += f": {a.note}"
            print(msg)

    if not actions:
        print("No changes needed.")
    elif not apply:
        print("\nDry-run only. Re-run with --apply to perform these changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
