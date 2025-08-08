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

    # costmap = np.ones(shape, dtype=np.float32) * 1000.0
    # costmap[road_mask.astype(bool)] = 0.0
    # costmap[trail_mask.astype(bool)] = 0.0
    # costmap[grass_mask.astype(bool)] = 700.0
    # costmap[buildings_mask.astype(bool)] = 1000.0
    # costmap[trees_mask.astype(bool)] = 1000.0
    # costmap[water_mask.astype(bool)] = 1000.0
    # costmap[river_mask.astype(bool)] = 1000.0
    mask_addition = (road_mask + trail_mask + grass_mask + buildings_mask + trees_mask + water_mask + river_mask) / 7.0
    data_region = (mask_addition > 0.3)
    data_region_float = data_region.astype(np.float32)

    costmap = np.zeros(shape, dtype=np.float32)
    costmap += 0 * road_mask * data_region_float
    costmap += 0 * trail_mask * data_region_float
    costmap += 0 * grass_mask * data_region_float
    costmap += 1000 * buildings_mask * data_region_float
    costmap += 1000 * trees_mask * data_region_float
    costmap += 1000 * water_mask * data_region_float
    costmap += 1000 * river_mask * data_region_float

    costmap += 1000 * (1 - data_region_float)  # Assign high cost to non-data regions

    return costmap
