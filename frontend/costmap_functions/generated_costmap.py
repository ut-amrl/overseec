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

    road_logit = mask_dict.get('road', np.zeros(shape, dtype=np.float32))
    trees_logit = mask_dict.get('tree', np.zeros(shape, dtype=np.float32))
    buildings_logit = mask_dict.get('building', np.zeros(shape, dtype=np.float32))
    grass_logit = mask_dict.get('grass', np.zeros(shape, dtype=np.float32))
    trail_logit = mask_dict.get('trail or footway', np.zeros(shape, dtype=np.float32))
    water_logit = mask_dict.get('water', np.zeros(shape, dtype=np.float32))
    river_logit = mask_dict.get('river', np.zeros(shape, dtype=np.float32))


    road_mask = road_logit > 0.5
    trail_mask = trail_logit > 0.5
    grass_mask = grass_logit > 0.5  
    buildings_mask = buildings_logit > 0.5
    trees_mask = trees_logit > 0.5
    water_mask = water_logit > 0.5
    river_mask = river_logit > 0.5

    # smoothen the logits
    # trees_logit = cv2.GaussianBlur(trees_logit, (151, 151), 0)
    # buildings_logit = cv2.GaussianBlur(buildings_logit, (151, 151), 0)
    # water_logit = cv2.GaussianBlur(water_logit, (151, 151), 0)
    # river_logit = cv2.GaussianBlur(river_logit, (151, 151), 0)
    # grass_logit = cv2.GaussianBlur(grass_logit, (151, 301), 0)

    mask_count = road_mask.astype(np.float32) + trail_mask.astype(np.float32) + grass_mask.astype(np.float32) + \
                 buildings_mask.astype(np.float32) + trees_mask.astype(np.float32) + water_mask.astype(np.float32) + river_mask.astype(np.float32)

    # costmap = np.ones(shape, dtype=np.float32) * 1000.0
    # costmap[road_mask.astype(bool)] = 0.0
    # costmap[trail_mask.astype(bool)] = 0.0
    # costmap[grass_mask.astype(bool)] = 700.0
    # costmap[buildings_mask.astype(bool)] = 1000.0
    # costmap[trees_mask.astype(bool)] = 1000.0
    # costmap[water_mask.astype(bool)] = 1000.0
    # costmap[river_mask.astype(bool)] = 1000.0
    data_region = (mask_count > 0)
    data_region_float = data_region.astype(np.float32)

    costmap = np.zeros(shape, dtype=np.float32)
    costmap[road_mask] += 0 * road_logit[road_mask]
    costmap[trail_mask] += 0 * trail_logit[trail_mask]
    costmap[grass_mask] += 900 * grass_logit[grass_mask]
    costmap[buildings_mask] += 2000 * buildings_logit[buildings_mask]
    costmap[trees_mask] += 2000 * trees_logit[trees_mask]
    costmap[water_mask] += 2000 * water_logit[water_mask]
    costmap[river_mask] += 2000 * river_logit[river_mask]

    costmap[data_region] = costmap[data_region] / mask_count[data_region]

    costmap += costmap.max() * (1 - data_region_float)  # Assign high cost to non-data regions


    return costmap
