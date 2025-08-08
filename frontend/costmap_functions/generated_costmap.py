import numpy as np
import cv2

def mask_and(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    return np.logical_and(mask1, mask2).astype(np.uint8)

def mask_or(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    return np.logical_or(mask1, mask2).astype(np.uint8)

def mask_not(mask: np.ndarray) -> np.ndarray:
    return np.logical_not(mask).astype(np.uint8)

def generate_costmap(mask_dict):
    shape = next(iter(mask_dict.values())).shape

    classes = ['road', 'trail or footway', 'water', 'grass', 'building', 'river']
    assert all(cls in mask_dict for cls in classes), "mask_dict must contain all default classes"

    road_mask = mask_dict.get('road', np.zeros(shape, dtype=np.float32))
    trees_mask = mask_dict.get('tree', np.zeros(shape, dtype=np.float32))
    buildings_mask = mask_dict.get('building', np.zeros(shape, dtype=np.float32))
    grass_mask = mask_dict.get('grass', np.zeros(shape, dtype=np.float32))
    trail_mask = mask_dict.get('trail or footway', np.zeros(shape, dtype=np.float32))
    water_mask = mask_dict.get('water', np.zeros(shape, dtype=np.float32))
    river_mask = mask_dict.get('river', np.zeros(shape, dtype=np.float32))

    costmap = np.ones(shape, dtype=np.float32) * 1000.0
    costmap[road_mask.astype(bool)] = 0.0
    costmap[trail_mask.astype(bool)] = 0.0
    costmap[grass_mask.astype(bool)] = 100.0
    costmap[buildings_mask.astype(bool)] = 1000.0
    costmap[trees_mask.astype(bool)] = 1000.0
    costmap[water_mask.astype(bool)] = 1000.0
    costmap[river_mask.astype(bool)] = 1000.0

    return costmap
