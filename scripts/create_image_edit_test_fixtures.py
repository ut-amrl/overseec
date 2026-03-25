#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
RESULTS_DIR = FRONTEND_DIR / "results"
SCENE_NAME = "edit_test_scene"
SCENE_DIR = RESULTS_DIR / SCENE_NAME


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def make_rgb_preview(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    r = np.clip(60 + x * 2, 0, 255)
    g = np.clip(80 + y * 2, 0, 255)
    b = np.clip(120 + ((x + y) // 3), 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    rgb[30:90, 40:110] = np.array([40, 140, 60], dtype=np.uint8)
    rgb[95:150, 120:205] = np.array([90, 90, 90], dtype=np.uint8)
    rgb[45:155, 150:168] = np.array([230, 210, 120], dtype=np.uint8)
    return rgb


def make_masks(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices((height, width))

    road_semantic = np.where(np.abs(x - 159) < 18, 180, 25).astype(np.uint8)
    road_refined = np.where(np.abs(x - 159) < 10, 255, 0).astype(np.uint8)

    water_semantic = np.where(((x - 90) ** 2 + (y - 70) ** 2) < 35 ** 2, 170, 20).astype(np.uint8)
    water_refined = np.where(((x - 90) ** 2 + (y - 70) ** 2) < 28 ** 2, 255, 0).astype(np.uint8)

    return road_semantic, road_refined, water_semantic, water_refined


def make_costmap_bw(road_mask: np.ndarray, water_mask: np.ndarray, width: int, height: int) -> np.ndarray:
    base = np.full((height, width), 170, dtype=np.uint8)
    base[road_mask > 0] = 35
    base[water_mask > 0] = 245
    return base


def make_costmap_color(costmap_bw: np.ndarray) -> np.ndarray:
    normalized = costmap_bw.astype(np.float32) / 255.0
    red = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    green = np.clip((1.0 - np.abs(normalized - 0.45) * 2.2) * 255, 0, 255).astype(np.uint8)
    blue = np.clip((1.0 - normalized) * 255, 0, 255).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def main() -> None:
    width, height = 256, 192

    ensure_dir(RESULTS_DIR / "final_costmaps")
    ensure_dir(SCENE_DIR / "mask" / "temp_latest" / "semantic")
    ensure_dir(SCENE_DIR / "mask" / "temp_latest" / "refined")
    ensure_dir(SCENE_DIR / "costmap" / "temp_latest")

    rgb = make_rgb_preview(width, height)
    road_sem, road_ref, water_sem, water_ref = make_masks(width, height)
    costmap_bw = make_costmap_bw(road_ref, water_ref, width, height)
    costmap_color = make_costmap_color(costmap_bw)

    save_image(SCENE_DIR / "original.tif", rgb)
    save_image(SCENE_DIR / "preview.png", rgb)

    save_image(SCENE_DIR / "mask" / "temp_latest" / "semantic" / "road.png", road_sem)
    save_image(SCENE_DIR / "mask" / "temp_latest" / "refined" / "road.png", road_ref)
    np.save(SCENE_DIR / "mask" / "temp_latest" / "semantic" / "road.npy", road_sem.astype(np.float32) / 255.0)
    np.save(SCENE_DIR / "mask" / "temp_latest" / "refined" / "road.npy", road_ref.astype(np.float32) / 255.0)

    save_image(SCENE_DIR / "mask" / "temp_latest" / "semantic" / "water.png", water_sem)
    save_image(SCENE_DIR / "mask" / "temp_latest" / "refined" / "water.png", water_ref)
    np.save(SCENE_DIR / "mask" / "temp_latest" / "semantic" / "water.npy", water_sem.astype(np.float32) / 255.0)
    np.save(SCENE_DIR / "mask" / "temp_latest" / "refined" / "water.npy", water_ref.astype(np.float32) / 255.0)

    save_image(SCENE_DIR / "costmap" / "temp_latest" / "costmap_bw.png", costmap_bw)
    save_image(SCENE_DIR / "costmap" / "temp_latest" / "costmap.png", costmap_color)

    save_image(RESULTS_DIR / "final_costmaps" / "edit_test_costmap_bw.png", costmap_bw)
    save_image(RESULTS_DIR / "final_costmaps" / "edit_test_costmap.png", costmap_color)

    with open(SCENE_DIR / "goals.json", "w", encoding="utf-8") as handle:
        json.dump([], handle)

    print(f"Created image editing fixtures under {SCENE_DIR}")


if __name__ == "__main__":
    main()
