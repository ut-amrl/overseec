import numpy as np
import cv2

def mask_and(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    return np.logical_and(mask1, mask2).astype(np.uint8)

def mask_or(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    return np.logical_or(mask1, mask2).astype(np.uint8)

def mask_not(mask: np.ndarray) -> np.ndarray:
    return np.logical_not(mask).astype(np.uint8)

def remove_mask(mask1 : np.ndarray, mask2: np.ndarray) -> np.ndarray:
    return np.logical_and(mask1, mask_not(mask2)).astype(np.uint8)

def distance_from_center_blob_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    mask_dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = mask_dist.max() or 1.0
    normalized_dist = mask_dist / max_dist
    return normalized_dist

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


    costmap = np.ones(shape, dtype=np.float32) * 1000.0
    costmap[road_mask.astype(bool)] = 0.0


    costmap[trail_mask.astype(bool)] = 0.0
    costmap[grass_mask.astype(bool)] = 100.0

    lethal_mask = mask_or(mask_or(mask_or(trees_mask, buildings_mask), water_mask))
    non_lethal_mask = mask_not(lethal_mask)
    lethal_mask = lethal_mask.astype(np.float32)
    non_lethal_mask = non_lethal_mask.astype(np.float32)


    costmap[lethal_mask.astype(bool)] = 1000.0 

    return costmap