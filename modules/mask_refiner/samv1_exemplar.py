import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from molmo_testing.models.semseg_config import SAM_Exemplar_Config, SemSegConfig
    


class Exemplar_Points_SAMv1_Mask_Refiner(nn.Module):
    def __init__(self, 
                 semseg_config: "SemSegConfig",
                 mask_refiner_config : "SAM_Exemplar_Config"):
        """
        Initializes the SAMProcessor class with batch processing capability.
        Loads the original Segment Anything Model (SAM) with ViT-H backbone.
        """
        super(Exemplar_Points_SAMv1_Mask_Refiner, self).__init__()

        self.semseg_config = semseg_config
        self.mask_refiner_config = mask_refiner_config

        self.device = self.mask_refiner_config.device

        # Load the SAM model with the .pth checkpoint
        sam = sam_model_registry[self.mask_refiner_config.model_type](checkpoint=self.mask_refiner_config.ckpt_path).to(self.device)
        sam.eval()  # Set the model to evaluation mode
        self.predictor = SamPredictor(sam)
        self.images = None  # Store the batch of images
    
    ###########################################################
    ###########################################################
    ################ Exemplar Point Extraction ################
    def extract_points(self, masks, label_type = "pos"):
        """Extracts randomly sampled points from a batch of multi-class masks (logits or binary class maps).
        
        Args:
            masks (Tensor): Shape (B, N, H, W), where N is the number of classes.
        
        Returns:
            List[List[Tuple[int, int]]]: A list of length B*N with sampled (x, y) coordinates.
        """
        B, N, H, W = masks.shape
        sampled_points = []
        sampled_points_classes = []

        sample_frac = self.mask_refiner_config.sample_fraction
        if label_type == "neg":
            sample_frac = self.mask_refiner_config.neg_sample_fraction

        for b in range(B):
            sampled_points_per_batch = []
            sampled_points_classes_per_batch =[]
            for n in range(N):
                y, x = torch.where(masks[b, n] == 1)  # Get all pixels for class n
                num_points = y.shape[0]

                if num_points == 0:
                    points = []
                else:
                # Shuffle and select a subset
                    num_to_sample = max(10, int(num_points * sample_frac))
                    shuffled_indices = torch.randperm(num_points)[:num_to_sample]

                    # Collect (x, y) points
                    points = torch.stack((x[shuffled_indices], y[shuffled_indices]), dim=-1).tolist()
                
                sampled_points_classes_per_batch.append([n] * len(points))
                sampled_points_per_batch.append(points)
            sampled_points.append(sampled_points_per_batch)
            sampled_points_classes.append(sampled_points_classes_per_batch)
        return sampled_points, sampled_points_classes

    def extract_points_from_masks(self, logits_semseg):
        """
        logits_semseg: (B, num_classes, H, W)
        """
        semseg_threshold_knobs = torch.tensor(list(self.semseg_config.classes_semseg_knobs.values()))
        positive_thresholded_masks = torch.zeros_like(logits_semseg)
        negative_thresholded_masks = torch.zeros_like(logits_semseg)
        for i in range(len(self.semseg_config.classes.keys())):
            positive_thresholded_masks[:, i] = (logits_semseg[:, i] > semseg_threshold_knobs[i]).float()
            negative_thresholded_masks[:, i] = 1 - positive_thresholded_masks[:, i]
        
        pos_points, pos_points_classes = self.extract_points(positive_thresholded_masks)
        neg_points, neg_points_classes = self.extract_points(negative_thresholded_masks, label_type="neg")

        return {
            "pos_points": pos_points,
            "pos_points_classes": pos_points_classes,
            "neg_points": neg_points,
            "neg_points_classes": neg_points_classes,
        }

    ################ Exemplar Point Extraction ################
    ###########################################################
    ###########################################################


    ########################################
    ########################################
    ################ SAM v1 ################

    def generate_masks_from_points(self, 
                                   foreground_points, 
                                   foreground_labels,
                                   background_points, 
                                   background_labels,
                                   torch_image,
                                   tile_semseg_logits,
                                   mask_threshold):
        
        point_labels    = foreground_labels[:self.mask_refiner_config.exemplar_num_points] #+ background_labels[:self.mask_refiner_config.sam_num_points]
        point_positions = foreground_points[:self.mask_refiner_config.exemplar_num_points] #+ class_neg_points[:self.mask_refiner_config.sam_num_points]

        if self.mask_refiner_config.use_negative_points:
            point_labels += background_labels[:self.mask_refiner_config.exemplar_num_points]
            point_positions += background_points[:self.mask_refiner_config.exemplar_num_points]
        
        _, H, W = torch_image.shape
        

        # reduce the size to 256,256
        tile_semseg_logits = torch.nn.functional.interpolate(
            tile_semseg_logits[None, None].float(), 
            size=(256, 256), 
            mode='bilinear', 
            align_corners=False
        )[0, 0]
        mask = (tile_semseg_logits.cpu().numpy() > mask_threshold).astype(np.uint8)[None]
        best_logits = np.zeros((H, W), dtype=np.float32) + self.mask_refiner_config.no_prompt_logit
        if len(point_labels) > 0:
            with torch.no_grad():
                sam_logits, scores, _ = self.predictor.predict(
                    point_coords=np.array(point_positions),
                    point_labels=np.array(point_labels),
                    mask_input=mask,
                    multimask_output=True,
                    return_logits=True,
                )
            best_logits = sam_logits[np.argmax(scores)]

        return best_logits

    ################ SAM v1 ################
    ########################################
    ########################################


    def forward(self, batch_images_torch, logits_semseg, idx):
        exemplar_point_dict = self.extract_points_from_masks(logits_semseg)
        B = batch_images_torch.shape[0]

        batch_pos_points = exemplar_point_dict["pos_points"]
        batch_pos_labels = exemplar_point_dict["pos_points_classes"]
        batch_neg_points = exemplar_point_dict["neg_points"]
        batch_neg_labels = exemplar_point_dict["neg_points_classes"]

        logits = []
        semseg_threshold_knobs = list(self.semseg_config.classes_semseg_knobs.values())
        # print(semseg_threshold_knobs)

        for b in range(B):
            tile_pos_points = batch_pos_points[b]
            tile_pos_labels = batch_pos_labels[b]
            tile_neg_points = batch_neg_points[b]
            tile_neg_labels = batch_neg_labels[b]

            tile_image = batch_images_torch[b]            
            tile_logits = []
            tile_image_np = tile_image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
            self.predictor.set_image(tile_image_np)

            for class_points in zip(tile_pos_points, tile_pos_labels, tile_neg_points, tile_neg_labels, range(len(tile_pos_points))):
                class_logit = self.generate_masks_from_points(
                    foreground_points=class_points[0],
                    foreground_labels=class_points[1],
                    background_points=class_points[2],
                    background_labels=class_points[3],
                    torch_image=tile_image,
                    tile_semseg_logits=logits_semseg[b][class_points[4]],
                    mask_threshold=semseg_threshold_knobs[class_points[4]]
                )

                tile_logits.append(class_logit)

            
            #################################
            # Take only class 0 points

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            def thresh_logit_img(logit, thresh):
                sigmoid_logit = sigmoid(logit)
                mask = (sigmoid_logit > thresh).astype(np.uint8)
                mask_img = 255 * mask
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
                return mask_img
            
            def mask_2_img(mask):
                # mask = (logit > thresh).astype(np.uint8)
                mask_img = (255 * mask).astype(np.uint8)
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
                return mask_img
            
            comb_imgs = []
            for logit_idx in range(len(tile_logits)):
                img_copy = tile_image_np.copy()
                class_pos_points = tile_pos_points[logit_idx]  # Class 0
                class_neg_points = tile_neg_points[logit_idx]  # Class 0

                # Draw positive points (green circles)
                for point in class_pos_points[0:25]:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(img_copy, (x, y), radius=10, color=(0, 255, 0), thickness=-1)

                # Draw negative points (red circles)
                for point in class_neg_points[:25]:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(img_copy, (x, y), radius=10, color=(0, 0, 255), thickness=-1)

                temp_logit = tile_logits[logit_idx]
                temp_logit = sigmoid(temp_logit)
                temp_logit = (temp_logit - temp_logit.min()) / (temp_logit.max() - temp_logit.min() + 1e-6) * 255
                temp_logit = temp_logit.astype(np.uint8)
                temp_logit = cv2.cvtColor(temp_logit, cv2.COLOR_GRAY2BGR)



                mask_img = thresh_logit_img(tile_logits[logit_idx], 0.85)
                # mask_img = mask_2_img(batch_logits[0])

                # print(logits_semseg[b].shape)
                semseg_logits_tile = logits_semseg[b][logit_idx]
                semseg_masks_tile = (semseg_logits_tile > semseg_threshold_knobs[logit_idx]) * 255
                semseg_masks_tile = semseg_masks_tile.cpu().numpy().astype(np.uint8)
                semseg_masks_tile = cv2.cvtColor(semseg_masks_tile, cv2.COLOR_GRAY2BGR)

                semseg_logits_tile = (semseg_logits_tile.cpu().numpy() * 255.0).astype(np.uint8)
                semseg_logits_tile = cv2.cvtColor(semseg_logits_tile, cv2.COLOR_GRAY2BGR)

                comb_img = np.hstack([tile_image_np[..., ::-1], 
                                      img_copy[..., ::-1], 
                                      semseg_masks_tile, 
                                      semseg_logits_tile, 
                                      mask_img, temp_logit])
                comb_imgs.append(comb_img)
            
            comb_imgs = np.vstack(comb_imgs)


            cv2.imwrite(f"points_{idx}.png", comb_imgs)

            # cv2.imwrite(f"img_{idx}.png", tile_image_np[..., ::-1])
            # import pdb;pdb.set_trace()
            ###########################################
            
            logits.append(tile_logits)
        logits = np.array(logits)

        logits_tensor = torch.from_numpy(logits).to(self.mask_refiner_config.device)
        return logits_tensor