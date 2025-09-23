import numpy as np
import cv2
import torch

def mask_and(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
            return torch.logical_and(mask1.bool(), mask2.bool()).to(torch.uint8)

def mask_or(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
            return torch.logical_or(mask1.bool(), mask2.bool()).to(torch.uint8)

def mask_not(mask: torch.Tensor) -> torch.Tensor:
            return torch.logical_not(mask.bool()).to(torch.uint8)

def mask_remove(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
            return (mask1.bool() & ~mask2.bool()).to(torch.uint8)

def convert_masks2torch(mask_dict, device):
    for key in mask_dict:
        if isinstance(mask_dict[key], np.ndarray):
            mask_dict[key] = torch.from_numpy(mask_dict[key]).to(device)
    return mask_dict  

def generate_costmap(mask_dict, t_dict={"t_l":0.4, "t_a":0.6}):
    shape = next(iter(mask_dict.values())).shape
    mask_dict = convert_masks2torch(mask_dict, device)

    device = next(iter(mask_dict.values())).device
    road_logit = mask_dict.get('road', torch.zeros(shape, dtype=torch.float32, device=device))
    trees_logit = mask_dict.get('tree', torch.zeros(shape, dtype=torch.float32, device=device))
    buildings_logit = mask_dict.get('building', torch.zeros(shape, dtype=torch.float32, device=device))
    grass_logit = mask_dict.get('grass', torch.zeros(shape, dtype=torch.float32, device=device))
    trail_logit = mask_dict.get('trail or footway', torch.zeros(shape, dtype=torch.float32, device=device))
    water_logit = mask_dict.get('water', torch.zeros(shape, dtype=torch.float32, device=device))

    t_l = t_dict.get("t_l", 0.4)
    t_a = t_dict.get("t_a", 0.6)

    road_logit = torch.from_numpy(road_logit).to(device)
    trees_logit = torch.from_numpy(trees_logit).to(device)
    buildings_logit = torch.from_numpy(buildings_logit).to(device)
    grass_logit = torch.from_numpy(grass_logit).to(device)
    trail_logit = torch.from_numpy(trail_logit).to(device)
    water_logit = torch.from_numpy(water_logit).to(device)

    road_mask = road_logit > t_l
    trail_mask = trail_logit > t_l
    grass_mask = grass_logit > t_a  
    buildings_mask = buildings_logit > t_a
    trees_mask = trees_logit > t_a
    water_mask = water_logit > t_a
    



    # Hierarchy
    # Nothing for now
    # Geometry
    # Nothing for now

    # Unknown Mask
    mask_count = (road_mask.to(torch.float32) + trail_mask.to(torch.float32) + grass_mask.to(torch.float32)                       
                + buildings_mask.to(torch.float32) + trees_mask.to(torch.float32) + water_mask.to(torch.float32))

    data_region = (mask_count > 0)
    data_region_float = data_region.to(torch.float32)


    costmap = torch.zeros(shape, dtype=torch.float32, device=device)
    costmap[road_mask] += 0 * road_logit[road_mask]
    costmap[trail_mask] += 0 * trail_logit[trail_mask]
    costmap[grass_mask] += 300 * grass_logit[grass_mask]
    costmap[buildings_mask] += 2000 * buildings_logit[buildings_mask]
    costmap[trees_mask] += 2000 * trees_logit[trees_mask]
    costmap[water_mask] += 2000 * water_logit[water_mask]

    costmap[data_region] = costmap[data_region] / mask_count[data_region]

    costmap += costmap.max() * (1 - data_region_float)  # Assign high cost to non-data regions
    costmap = costmap.cpu().numpy()
    return costmap
