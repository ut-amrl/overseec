import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from overseec.modules.semseg import *
from overseec.modules.mask_refiner import *
from overseec.overseec_config import AllConfig



model_dict = {
    "dinounet": DinoUNet,
    "segformer": SegFormer,
    "clipseg": CLIPSeg,
}

mask_refiner_dict = {
    "samrefiner": SamRefiner_wrapper,
    "samv1_exemplar": Exemplar_Points_SAMv1_Mask_Refiner,
}

def temperature_sigmoid(x, temperature=1.0):
    return torch.sigmoid(x / temperature)

class OVerSeeC_ImageDataset(Dataset):
    def __init__(self, all_patches_dict, num_patches):
        self.all_patches_dict = all_patches_dict
        self.num_patches = num_patches
    
    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        patch_dict = {}
        for key, value in self.all_patches_dict.items():
            patch_dict[key] = value[idx]
        
        return patch_dict

    
class OVerSeeC(nn.Module):
    def __init__(self,
                 config = AllConfig,
                 ):
        super().__init__()
        self.config : AllConfig = config

        self.set_semseg_model()

    def set_semseg_model(self,):
        # model = model_dict[self.config.model_name](self.config.semseg_config, 
        #                                                 self.config.model_config)
        
        ###################### Setting Semantic Segmentation ##################
        self.semseg = OVerSeeC_Semseg(
                semseg_config = self.config.semseg_config,
                model_config = self.config.model_config,
            )
        

        if self.config.model_name == "clipseg":
            semseg_model_class = CLIPSeg
        elif self.config.model_name == "segformer":
            semseg_model_class = SegFormer
        elif self.config.model_name == "dinounet":
            semseg_model_class = DinoUNet


        self.semseg.set_model(model_class=semseg_model_class)

        if self.config.model_name != "clipseg":
            checkpoint = torch.load(self.config.model_ckpt,
                                    map_location=self.config.semseg_config.device,
                                    weights_only=True)
            self.semseg.load_state_dict(checkpoint['state_dict'])
            self.semseg.eval()        
        

        ################# Setting Mask Refiner #################
        self.mask_refiner = OVerSeeC_Mask_Refiner(self.config.semseg_config,
                                    self.config.mask_refiner_config)
        
        if self.config.mask_refiner_name == "samrefiner":
            mask_refiner_model_class = SamRefiner_wrapper
        elif self.config.mask_refiner_name == "samv1_exemplar":
            mask_refiner_model_class = Exemplar_Points_SAMv1_Mask_Refiner

        # mask_refiner_model_class = Exemplar_Points_SAMv1_Mask_Refiner
        # mask_refiner_model_class = SamRefiner_wrapper
        self.mask_refiner.set_model(model_class=mask_refiner_model_class)
        


    def set_classes(self, classes, 
                    classes_semseg_knobs, # for semseg to sam - internal 
                    classes_sigmoid_semseg_knobs, # final thresholding for semseg outputs 
                    classes_sigmoid_sam_knobs, # final thresholding for sam outputs
                    classes_prefs =None):
        self.config.classes = classes
        self.config.classes_semseg_knobs = classes_semseg_knobs
        self.config.classes_sigmoid_semseg_knobs = classes_sigmoid_semseg_knobs
        self.config.classes_sigmoid_sam_knobs = classes_sigmoid_sam_knobs

        if classes_prefs!= None:
            self.config.classes_prefs = classes_prefs
        
        self.config.reset()

        # self.semseg.semseg_config = self.config.semseg_config
        # self.semseg.model_config = self.config.model_config

        # self.semseg.model.semseg_config = self.config.semseg_config
        # self.semseg.model.model_config = self.config.model_config
        self.semseg.set_classes(classes=classes)
        self.mask_refiner.set_classes(classes=classes)
    
    def batch_crop_logits_and_image(self, logits_tensor, rgb_image):

        tile_size = self.config.sat_2_cmap_config.mask_refiner_tile_size  # (tile_h, tile_w)
        stride = self.config.sat_2_cmap_config.mask_refiner_stride
        num_threads = self.config.sat_2_cmap_config.image_2_tiles_threads

        N, H, W = logits_tensor.shape
        h, w = tile_size

        def crop_patch(y, x):
            y_end = min(y + h, H)
            x_end = min(x + w, W)

            # Crop logits
            logit_patch = logits_tensor[:, y:y_end, x:x_end]

            if logit_patch.shape[1:] != (h, w):
                padded_logit_patch = torch.zeros((N, h, w), dtype=logit_patch.dtype, device=logit_patch.device)
                padded_logit_patch[:, :logit_patch.shape[1], :logit_patch.shape[2]] = logit_patch
                logit_patch = padded_logit_patch

            # Crop RGB
            rgb_patch = rgb_image[y:y_end, x:x_end, :]  # shape: (patch_h, patch_w, 3)

            if rgb_patch.shape[:2] != (h, w):
                padded_rgb_patch = np.zeros((h, w, 3), dtype=np.uint8)
                padded_rgb_patch[:rgb_patch.shape[0], :rgb_patch.shape[1], :] = rgb_patch
                rgb_patch = padded_rgb_patch

            rgb_patch_torch = torch.tensor(rgb_patch).permute(2, 0, 1).float()
            # else:
            #     rgb_patch_tr = rgb_patch.transpose(2, 0, 1)  # (C, H, W)
            #     rgb_patch_torch = self.config.img_transform(rgb_patch_tr)

            return logit_patch.to("cpu"), rgb_patch_torch, (y, x)

        # Prepare patch positions
        y_positions = list(range(0, H - h + 1, stride)) + ([H - h] if (H - h) % stride != 0 else [])
        x_positions = list(range(0, W - w + 1, stride)) + ([W - w] if (W - w) % stride != 0 else [])

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(lambda pos: crop_patch(*pos),
                                            [(y, x) for y in y_positions for x in x_positions]),
                                total=len(y_positions) * len(x_positions),
                                desc="Cropping Logits and Image Patches"))

        logits_patches, rgb_patches, positions = zip(*results)

        return list(logits_patches), list(rgb_patches), list(positions)


    def batch_crop_image(self, image):

        tile_size = self.config.sat_2_cmap_config.semseg_tile_size
        stride = self.config.sat_2_cmap_config.semseg_stride
        num_threads = self.config.sat_2_cmap_config.image_2_tiles_threads

        H, W, C = image.shape
        h, w = tile_size

        def crop_patch(y, x):
            # Ensure we don't go beyond the image boundary
            y_end = min(y + h, H)
            x_end = min(x + w, W)

            # Extract the exact patch without unnecessary padding
            patch = image[y:y_end, x:x_end, :]

            # If the patch is smaller than expected (only for last row/col), pad it correctly
            if patch.shape[:2] != (h, w):
                padded_patch = np.zeros((h, w, C), dtype=np.uint8)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch  # Replace patch with the correctly padded version

            # convert to torch tensor
            if self.config.img_transform == None:
                patch_torch = torch.tensor(patch).permute(2, 0, 1).float()
            else:
                patch_tr = patch
                patch_torch = self.config.img_transform(patch_tr)



            return patch_torch, (y, x)

        # Generate (y, x) positions that **always align** with the image edges
        y_positions = list(range(0, H - h + 1, stride)) + ([H - h] if (H - h) % stride != 0 else [])
        x_positions = list(range(0, W - w + 1, stride)) + ([W - w] if (W - w) % stride != 0 else [])

        # Process patches in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(lambda pos: crop_patch(*pos),
                                            [(y, x) for y in y_positions for x in x_positions]),
                                total=len(y_positions) * len(x_positions),
                                desc="Cropping Image Patches"))

        patches_torch, positions = zip(*results)
        return list(patches_torch), list(positions)

    def combine_logits_threadsafe(self,
                              tile_logits,          # [N,C,h,w]  torch tensor
                              patch_positions,      # list[(y,x)]
                              H, W, *,
                              reduce_type="mean",
                              step="semseg"):

        tile_h, tile_w = (self.config.sat_2_cmap_config.semseg_tile_size
                        if step == "semseg"
                        else self.config.sat_2_cmap_config.mask_refiner_tile_size)

        num_classes = self.config.semseg_config.num_classes
        device      = tile_logits.device
        threads     = self.config.sat_2_cmap_config.raw_logit_joiner_threads

        # target tensors
        full_accum = torch.zeros((num_classes, H, W),
                                dtype=tile_logits.dtype,
                                device=device)
        full_count = (torch.zeros((1, H, W),
                                dtype=torch.int32,
                                device=device)
                    if reduce_type == "mean" else None)

        if reduce_type == "max":
            full_accum.fill_(-float("inf"))

        # --------------------------------------------------------------
        #  lock to make the += / max operation atomic
        # --------------------------------------------------------------
        lock = threading.Lock()

        def process_tile(idx_pos):
            idx, (y, x) = idx_pos
            patch = tile_logits[idx]                     # [C,h,w]

            y_end = min(y + tile_h, H)
            x_end = min(x + tile_w, W)
            ph, pw = y_end - y, x_end - x

            with lock:
                if reduce_type == "mean":
                    full_accum[:, y:y_end, x:x_end] += patch[:, :ph, :pw]
                    full_count[:, y:y_end, x:x_end] += 1
                elif reduce_type == "max":
                    full_accum[:, y:y_end, x:x_end] = torch.maximum(
                        full_accum[:, y:y_end, x:x_end],
                        patch[:, :ph, :pw])
                else:
                    raise ValueError("reduce_type must be 'mean' or 'max'.")

        with ThreadPoolExecutor(max_workers=threads) as pool:
            pool.map(process_tile, enumerate(patch_positions))

        if reduce_type == "mean":
            full_count.clamp_(min=1)
            return full_accum / full_count
        else:
            return full_accum
    
    def combine_logits(self, overall_raw_logits, patch_positions, H, W, reduce_type="mean", step="semseg"):
        """
        Combines tile logits into a full-size logits tensor.
        
        Args:
            overall_raw_logits (torch.Tensor): [B, C, tile_h, tile_w]
            patch_positions (List[Tuple[int, int]]): (y, x) positions
            H (int): Full image height
            W (int): Full image width
            reduce_type (str): 'mean' or 'max' — how to handle overlaps
        Returns:
            torch.Tensor: [C, H, W] stitched logits
        """

        if step=="semseg":
            tile_size = self.config.sat_2_cmap_config.semseg_tile_size
        elif step=="sam":
            tile_size = self.config.sat_2_cmap_config.mask_refiner_tile_size

        num_classes = self.config.semseg_config.num_classes
        num_threads = self.config.sat_2_cmap_config.raw_logit_joiner_threads

        device = overall_raw_logits.device
        tile_h, tile_w = tile_size

        full_logits_sum_or_max = torch.zeros((num_classes, H, W), device=device)
        full_logits_count = torch.zeros((1, H, W), device=device) if reduce_type == "mean" else None

        if reduce_type == "max":
            # Initialize with very low values for max pooling
            full_logits_sum_or_max.fill_(-float('inf'))

        def process_tile(idx_pos):
            idx, (y, x) = idx_pos
            logits = overall_raw_logits[idx]  # [num_classes, tile_h, tile_w]

            y_end = min(y + tile_h, H)
            x_end = min(x + tile_w, W)

            patch_h = y_end - y
            patch_w = x_end - x

            if reduce_type == "mean":
                full_logits_sum_or_max[:, y:y_end, x:x_end] += logits[:, :patch_h, :patch_w]
                full_logits_count[:, y:y_end, x:x_end] += 1
            elif reduce_type == "max":
                full_logits_sum_or_max[:, y:y_end, x:x_end] = torch.maximum(
                    full_logits_sum_or_max[:, y:y_end, x:x_end],
                    logits[:, :patch_h, :patch_w]
                )
            else:
                raise ValueError(f"Unknown reduce_type '{reduce_type}'. Use 'mean' or 'max'.")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(executor.map(process_tile, enumerate(patch_positions)))

        if reduce_type == "mean":
            full_logits_count = torch.clamp(full_logits_count, min=1)
            full_logits_mean = full_logits_sum_or_max / full_logits_count
            return full_logits_mean
        else:  # max
            return full_logits_sum_or_max

    def forward(self, tiff_path_img, tiff_img=None):
        
        if tiff_img is None:
            tiff_img = cv2.imread(tiff_path_img)  # read a small patch of the image
        tiff_img = cv2.cvtColor(tiff_img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

        # cv2.imwrite("rgb_image.png", cv2.cvtColor(tiff_img, cv2.COLOR_RGB2BGR))
        sigmoid_semseg_logits, semseg_logits_map, high_res_image = self.run_semseg(tiff_img)

        # logits to argmax
        semseg_map = torch.argmax(semseg_logits_map, dim=0)
        one_hot_semseg_map = F.one_hot(semseg_map, num_classes=self.config.semseg_config.num_classes).permute(2, 0, 1)


        semseg_colours = np.array(list(self.config.classes.values()), dtype=np.uint8)
        class_indices = torch.argmax(one_hot_semseg_map, dim=0).cpu().numpy()
        color_image = semseg_colours[class_indices]
        # cv2.imwrite("semantic_color_map_semseg.png", cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        sigmoid_mask_refiner_logits, mask_refiner_logits_map, mask_refiner_logits_tiled = self.run_mask_refiner(sigmoid_semseg_logits, high_res_image)
        
        overall_sigmoid_mask_refiner_logits = temperature_sigmoid(mask_refiner_logits_map)
        semseg_map = torch.argmax(overall_sigmoid_mask_refiner_logits, dim=0)
        one_hot_semseg_map = F.one_hot(semseg_map, num_classes=self.config.semseg_config.num_classes).permute(2, 0, 1)


        semseg_colours = np.array(list(self.config.classes.values()), dtype=np.uint8)
        class_indices = torch.argmax(one_hot_semseg_map, dim=0).cpu().numpy()
        color_image = semseg_colours[class_indices]
        # cv2.imwrite("semantic_color_map.png", cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        for c in range(self.config.semseg_config.num_classes):
            mask = sigmoid_mask_refiner_logits[c]
            mask = (mask * 255).cpu().numpy().astype(np.uint8)
            # cv2.imwrite(f"mask_{c}.png", mask)
        
        for c in range(self.config.semseg_config.num_classes):
            mask = mask_refiner_logits_map[c]
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)  # normalize to [0, 1]
            mask = (mask * 255).cpu().numpy().astype(np.uint8)
            # cv2.imwrite(f"mask_raw_{c}.png", mask)
        
        return sigmoid_mask_refiner_logits, mask_refiner_logits_map, mask_refiner_logits_tiled, sigmoid_semseg_logits, semseg_logits_map

    def run_semseg(self, tiff_path_img):
        # check if tiff_apth_img is numpy array
        if isinstance(tiff_path_img, np.ndarray):
            high_res_image = tiff_path_img
        else:
            if self.config.img_format == "RGB":
                high_res_image = cv2.imread(tiff_path_img, cv2.IMREAD_COLOR_RGB)
            else:
                high_res_image = cv2.imread(tiff_path_img, cv2.IMREAD_COLOR_BGR)

        H_img, W_img, _ = high_res_image.shape

        print("\n\n")
        print("Semantic Segmentation ..... ")
        torch_image_patches_list, patch_positions = self.batch_crop_image(high_res_image)

        dataset = OVerSeeC_ImageDataset(all_patches_dict={
            "image": torch_image_patches_list
        }, num_patches=len(torch_image_patches_list))

        batch_size = self.config.semseg_config.batch_size
        dataloader = DataLoader(dataset, 
                                batch_size = batch_size,
                                num_workers = self.config.semseg_config.num_workers,
                                pin_memory = self.config.semseg_config.pin_memory,
                                )
        
        semseg_logits_tiled = torch.zeros(len(dataset), 
                                     self.config.semseg_config.num_classes, 
                                     *self.config.sat_2_cmap_config.semseg_tile_size).to(self.config.semseg_config.device)
        
        for i, batch_input_dict in enumerate(tqdm(dataloader, desc="semseg")):
            image_batch = batch_input_dict["image"].to(device=self.config.semseg_config.device)
            # image_batch = image_batch.to(device=self.config.semseg_config.device)
            semseg_logits_tile = self.semseg.get_semseg_logits(image_batch)
            semseg_logits_tiled[i * batch_size : (i+1) * batch_size] = semseg_logits_tile
        

        semseg_logits_map  = self.combine_logits_threadsafe(semseg_logits_tiled, patch_positions, H_img, W_img,
                                                    reduce_type="max")
        
        sigmoid_semseg_logits = torch.sigmoid(semseg_logits_map)

        return sigmoid_semseg_logits, semseg_logits_map, high_res_image

    def run_mask_refiner(self, sigmoid_semseg_logits, high_res_image):
        H_img, W_img, _ = high_res_image.shape
        print("\n\n")
        print("Mask Refiner ..... ")
        logits_patches_list, rgb_pathes_list, patch_positions = self.batch_crop_logits_and_image(sigmoid_semseg_logits, high_res_image)


        dataset = OVerSeeC_ImageDataset(all_patches_dict={
            "image": rgb_pathes_list,
            "logits": logits_patches_list
        }, num_patches=len(rgb_pathes_list))

        batch_size = self.config.mask_refiner_config.batch_size
        dataloader = DataLoader(dataset, 
                                batch_size = batch_size,
                                num_workers = self.config.mask_refiner_config.num_workers,
                                pin_memory = self.config.mask_refiner_config.pin_memory,
                                )

        mask_refiner_logits_tiled = torch.zeros((len(dataset), 
                                     self.config.semseg_config.num_classes, 
                                     *self.config.sat_2_cmap_config.mask_refiner_tile_size))
        

        def process_batch(i, batch_input_dict):
            image_batch = batch_input_dict["image"].to(self.config.mask_refiner_config.device)
            logit_batch = batch_input_dict["logits"].to(self.config.mask_refiner_config.device)

            mask_refiner_logits_tile = self.mask_refiner.get_refined_masks(image_batch, logit_batch, i)
            return i, mask_refiner_logits_tile

        futures = []
        # results = [None] * len(dataloader)

        with ThreadPoolExecutor(max_workers=2) as executor:
            for i, batch_input_dict in enumerate(tqdm(dataloader, desc="Mask refiner")):
                futures.append(executor.submit(process_batch, i, batch_input_dict))

            for future in tqdm(futures, desc="Collecting results"):
                i, mask_refiner_logits_tile = future.result()
                mask_refiner_logits_tiled[i * batch_size : (i + 1) * batch_size] = mask_refiner_logits_tile

        for c in range(self.config.semseg_config.num_classes):
            class_logits = mask_refiner_logits_tiled[:, c]
            try:
                min_val = class_logits[~torch.isnan(class_logits)].min()
                min_val = min_val.item()
            except:
                min_val = -40
            class_logits = torch.nan_to_num(class_logits, nan=min_val)
            mask_refiner_logits_tiled[:, c] = class_logits

        # (Pdb) sig_0 = 1 / (1 + np.exp(-mask_refiner_logits_map[0].cpu().numpy()))
        # (Pdb) sig_0 = 1 / (1 + np.exp(-mask_refiner_logits_map[0].cpu().numpy())) * 255
        # (Pdb) cv2.imwrite("rwik.png", sig_0.astype(np.uint8))

        sigmoid_mask_refiner_tiled = torch.sigmoid(mask_refiner_logits_tiled)

        mask_refiner_logits_map = self.combine_logits_threadsafe(mask_refiner_logits_tiled, patch_positions, H_img, W_img, 
                                             reduce_type="mean", step="sam")    
        
        sigmoid_mask_refiner_logits = self.combine_logits_threadsafe(sigmoid_mask_refiner_tiled, patch_positions, H_img, W_img,
                                             reduce_type="mean", step="sam")
        # sigmoid_mask_refiner_logits = torch.sigmoid(mask_refiner_logits_map)


        return sigmoid_mask_refiner_logits, mask_refiner_logits_map, mask_refiner_logits_tiled
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--model", type=str, default="dinounet", choices=model_dict.keys())
    parser.add_argument("--refiner", type=str, default="samrefiner", choices=mask_refiner_dict.keys())

    args = parser.parse_args()






    classes = {
        "road": (128, 64, 128),
        "trail or footway": (244, 35, 232),
        "tree": (0, 128, 0),
        "grass": (0, 128, 128),
        "building": (0, 0, 128),
        "water": (0, 0, 255),
    }

    clipseg_classes_semseg_knobs = {
        "road": 0.4,
        "trail or footway": 0.4,
        "tree": 0.65,
        "grass": 0.65,
        "building": 0.65,
        # "trail or footway": 0.3,
        "water": 0.85,
    }

    config = AllConfig(
        model_ckpt = args.checkpoint,
        model_name = args.model,
        mask_refiner_name= args.refiner,

        classes=classes,
        classes_semseg_knobs=clipseg_classes_semseg_knobs,
        # classes_sigmoid_semseg_knobs=clipseg_classes_sigmoid_semseg_knobs,
        # classes_sigmoid_sam_knobs=clipseg_classes_sigmoid_sam_knobs,


    )

    config.sat_2_cmap_config.mask_refiner_tile_size = (512, 512)
    config.sat_2_cmap_config.mask_refiner_stride = 128
    config.semseg_config.tile_combine_method = "mean"

    config.sat_2_cmap_config.mask_refiner_tile_size = (512, 512)
    config.sat_2_cmap_config.mask_refiner_stride = 256

    config.mask_refiner_config.use_negative_points = True

    config.reset()
    config.mask_refiner_config.tile_combine_method = "max"

    sat_2_mask = OVerSeeC(
        config=config
    )

    

    # img_path = "/scratch/rwik/SARA/molmo_testing/geotiffs/bluff_springs_zoom_in.tif"
    img_path = "/scratch/rwik/SARA/molmo_testing/geotiffs/tiffs/austin_lady_bird_south.tif"
    # sigmoid_semseg_logits, semseg_logits_map, sigmoid_mask_refiner_logits, mask_refiner_logits_map = sat_2_mask("/scratch/rwik/SARA/molmo_testing/geotiffs/bluff_springs_zoom_in.tif")
    # sigmoid_semseg_logits, semseg_logits_map, high_res_image = sat_2_mask.run_semseg(img_path)
    # sigmoid_mask_refiner_logits, mask_refiner_logits_map = sat_2_mask.run_mask_refiner(sigmoid_semseg_logits, high_res_image)
    img = cv2.imread(img_path)[1000:6000, 1000:6000, :]
    sat_2_mask(img_path, img)