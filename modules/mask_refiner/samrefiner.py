import numpy as np
import torch

# from segment_anything import sam_model_registry, SamPredictor
from overseec.modules.mask_refiner.SAMRefiner.sam_refiner import sam_refiner_image_embedding
from overseec.modules.mask_refiner.SAMRefiner.samrefiner_sam.samrefiner_sam.utils.transforms import ResizeLongestSide
from overseec.modules.mask_refiner.SAMRefiner.samrefiner_sam.samrefiner_sam import sam_model_registry as samrefiner_model_registry
from overseec.modules.mask_refiner.SAMRefiner.utils import prepare_image

import torch.nn as nn
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from overseec.semseg_config import SAM_Exemplar_Config, SemSegConfig
    


class SamRefiner_wrapper(nn.Module):
    def __init__(self, 
                 semseg_config: "SemSegConfig",
                 mask_refiner_config : "SAM_Exemplar_Config"):
        """
        Initializes the SAMProcessor class with batch processing capability.
        Loads the original Segment Anything Model (SAM) with ViT-H backbone.
        """
        super(SamRefiner_wrapper, self).__init__()

        self.semseg_config = semseg_config
        self.mask_refiner_config = mask_refiner_config

        self.device = self.mask_refiner_config.device

        # Load the SAM model with the .pth checkpoint
        self.sam_refiner = samrefiner_model_registry[self.mask_refiner_config.model_type](checkpoint=self.mask_refiner_config.ckpt_path).to(self.device)
        # sam.eval()  # Set the model to evaluation mode
        # self.predictor = SamPredictor(sam)
        self.images = None  # Store the batch of images


    ########################################
    ########################################
    ################ SAM Refiner ################

    def generate_mask_from_coarse_mask_img_embeddings(self, 
                                   image_np,
                                   sam_input_image,
                                   sam_image_embeddings,
                                   tile_semseg_logits,
                                   mask_threshold):
        
        H, W, _ = image_np.shape
        
        mask = (tile_semseg_logits.cpu().numpy() > mask_threshold).astype(np.uint8)[None]

        num_points = np.count_nonzero(mask)
        if num_points < 100:
            return np.full((H, W), -40, dtype=np.float32)

        best_logits = np.zeros((H, W), dtype=np.float32) + self.mask_refiner_config.no_prompt_logit
        with torch.no_grad():
            _, scores, samrefiner_logits = sam_refiner_image_embedding(
                sam_input_image,
                sam_image_embeddings,
                mask,
                self.sam_refiner,
                
                add_neg=self.mask_refiner_config.use_negative_points,
                iters = 5,
                gamma=4.0,
                strength=30
            )
        samrefiner_logits = samrefiner_logits.cpu().numpy()
        best_logits = samrefiner_logits[0][np.argmax(scores[0].cpu().numpy())]

        return best_logits

    ################ SAM Refiner ################
    ########################################
    ########################################


    def forward(self, batch_images_torch, logits_semseg, idx):
        # exemplar_point_dict = self.extract_points_from_masks(logits_semseg)
        B = batch_images_torch.shape[0]



        logits = []
        semseg_threshold_knobs = list(self.semseg_config.classes_semseg_knobs.values())
        # print(semseg_threshold_knobs)

        for b in range(B):
            tile_image = batch_images_torch[b]            
            tile_logits = []
            tile_image_np = tile_image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
            # self.predictor.set_image(tile_image_np)

            resize_transform = ResizeLongestSide(self.sam_refiner.image_encoder.img_size)
            sam_input_image = [prepare_image(tile_image_np, resize_transform, self.sam_refiner.device)]
            sam_input_image = torch.stack([self.sam_refiner.preprocess(x) for x in sam_input_image], dim=0)
            sam_image_embeddings = self.sam_refiner.image_encoder(sam_input_image)

            for class_idx in range(len(semseg_threshold_knobs)):
                class_logit = self.generate_mask_from_coarse_mask_img_embeddings(
                    image_np=tile_image_np,
                    sam_input_image=sam_input_image,
                    sam_image_embeddings=sam_image_embeddings,
                    tile_semseg_logits=logits_semseg[b][class_idx],
                    mask_threshold=semseg_threshold_knobs[class_idx]
                )

                tile_logits.append(class_logit)

            
            # #################################
            # # Take only class 0 points

            # def sigmoid(x):
            #     return 1 / (1 + np.exp(-x))
            # def thresh_logit_img(logit, thresh):
            #     sigmoid_logit = sigmoid(logit)
            #     mask = (sigmoid_logit > thresh).astype(np.uint8)
            #     mask_img = 255 * mask
            #     mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            #     return mask_img
            
            # comb_imgs = []
            # for logit_idx in range(len(tile_logits)):

            #     temp_logit = tile_logits[logit_idx]
            #     temp_logit = sigmoid(temp_logit)
            #     temp_logit = (temp_logit - temp_logit.min()) / (temp_logit.max() - temp_logit.min() + 1e-6) * 255
            #     temp_logit = temp_logit.astype(np.uint8)
            #     temp_logit = cv2.cvtColor(temp_logit, cv2.COLOR_GRAY2BGR)



            #     mask_img = thresh_logit_img(tile_logits[logit_idx], 0.85)
            #     # mask_img = mask_2_img(batch_logits[0])

            #     # print(logits_semseg[b].shape)
            #     semseg_logits_tile = logits_semseg[b][logit_idx]
            #     semseg_masks_tile = (semseg_logits_tile > semseg_threshold_knobs[logit_idx]) * 255
            #     semseg_masks_tile = semseg_masks_tile.cpu().numpy().astype(np.uint8)
            #     semseg_masks_tile = cv2.cvtColor(semseg_masks_tile, cv2.COLOR_GRAY2BGR)

            #     semseg_logits_tile = (semseg_logits_tile.cpu().numpy() * 255.0).astype(np.uint8)
            #     semseg_logits_tile = cv2.cvtColor(semseg_logits_tile, cv2.COLOR_GRAY2BGR)

            #     comb_img = np.hstack([tile_image_np[..., ::-1], 
            #                         #   img_copy[..., ::-1], 
            #                           semseg_masks_tile, 
            #                           semseg_logits_tile, 
            #                           mask_img, temp_logit])
            #     comb_imgs.append(comb_img)
            
            # comb_imgs = np.vstack(comb_imgs)


            # cv2.imwrite(f"points_{idx}.png", comb_imgs)
            # ###########################################
            
            logits.append(tile_logits)
        logits = np.array(logits)

        logits_tensor = torch.from_numpy(logits).to(self.mask_refiner_config.device)
        return logits_tensor