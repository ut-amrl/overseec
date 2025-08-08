import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import TYPE_CHECKING
from molmo_testing.models.semseg.utils import * 
import pytorch_lightning as pl
if TYPE_CHECKING:
    from molmo_testing.models.semseg_config import SemSegConfig, ModelConfig

class OVerSeeC_Semseg(pl.LightningModule):
    def __init__(self, 
                 semseg_config: "SemSegConfig", 
                 model_config: "ModelConfig"):
        super(OVerSeeC_Semseg, self).__init__()

        
        self.semseg_config = semseg_config
        self.model_config = model_config
        # self.device = semseg_config.device
        self.num_classes = semseg_config.num_classes
        self.resize_shape = self.model_config.resize_shape
    
    def set_model(self, model_class):
        self.model = model_class(
                                self.semseg_config,
                                self.model_config
                                )
        

    def load_model(self, ):
        checkpoint = torch.load(self.model_config.model_ckpt,
                        map_location=self.model_config.semseg_config.device,
                        weights_only=True)
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()  

    def set_classes(self, classes):
        self.semseg_config.classes = classes
        self.semseg_config.reset()
        self.num_classes = self.semseg_config.num_classes
    
    ####################################################
    ####################################################
    ############## Custom model training ###############

    def set_train_params(self,):
        self.lr = self.semseg_config.lr
        self.weight_decay = self.semseg_config.weight_decay
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.num_classes)  # ignore unknown class (last one)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]
    
    def shared_step(self, batch, batch_idx, stage="train"):
        images, class_masks, _, _ = batch  # You get class_masks here (B, num_classes+1, H, W)
        targets = torch.argmax(class_masks, dim=1)  # Convert one-hot to class indices (B, H, W)

        # Resize to match model input/output size

        if self.resize_shape is not None:
            images = F.interpolate(images, size=self.resize_shape, mode="bilinear", align_corners=False)
            targets = F.interpolate(targets.unsqueeze(1).float(), size=self.resize_shape, mode="nearest").squeeze(1).long()
        preds = self(images)  # Shape: (B, num_classes, H, W)
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(preds, size=targets.shape[-2:], mode="bilinear", align_corners=False)


        loss = self.criterion(preds, targets)

        pred_labels = torch.argmax(preds, dim=1)
        valid_mask = targets != self.num_classes  # Ignore unknown class
        acc = (pred_labels[valid_mask] == targets[valid_mask]).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_accuracy", acc, prog_bar=True)

        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, batch_idx, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.shared_step(batch, batch_idx, stage="val")

        if batch_idx % 2 == 0:
            images, _, _, rgb_labels = batch

            # (B, H, W)
            pred_labels = torch.argmax(preds, dim=1)

            # Create colored prediction map
            pred_rgb = torch.zeros((*pred_labels.shape, 3), dtype=torch.uint8)

            color_map = self.semseg_config.classes

            class_to_idx = {cls: idx for idx, cls in enumerate(color_map)}

            # Assign RGB values
            for class_name, color in color_map.items():
                idx = class_to_idx[class_name]
                pred_rgb[pred_labels == idx] = torch.tensor(color, dtype=torch.uint8)

            # Convert predictions to NumPy (BGR)
            preds_rgb_np = pred_rgb.cpu().numpy()
            preds_rgb_np = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in preds_rgb_np])

            # Convert GT RGB labels to NumPy (BGR)
            rgb_labels_np = (rgb_labels.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            rgb_labels_np_resized = np.stack([
                cv2.resize(rgb_labels_np[i], self.resize_shape, interpolation=cv2.INTER_NEAREST)
                for i in range(rgb_labels_np.shape[0])
            ])

            # Resize input images
            images_np = renormalize_and_resize(images, preds_rgb_np.shape)

            # Horizontally stack: input RGB | GT | prediction
            stacked_images = [
                np.hstack([images_np[i], rgb_labels_np_resized[i], preds_rgb_np[i]])
                for i in range(images_np.shape[0])
            ]
            stacked_result = np.vstack(stacked_images)

            # Plot with semantic legend (in BGR)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(stacked_result)
            ax.axis("off")
            ax.set_title("RGB | GT | Prediction")

            # Legend: BGR -> RGB normalized for matplotlib
            legend_patches = [
                mpatches.Patch(
                    color=np.array(color[::-1]) / 255.0,  # Convert BGR to RGB
                    label=class_name
                )
                for class_name, color in color_map.items()
            ]
            ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            plt.tight_layout()
            self.logger.experiment.add_figure(f"val/Semantic_Visualization_{batch_idx}", fig, self.global_step)
            plt.close(fig)

        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, batch_idx, stage="test")
        return loss

    ############## Custom model training ###############
    ####################################################
    ####################################################


    def forward(self, x, **kwargs):
        B, C, H, W = x.shape

        try:
            H_semseg = self.resize_shape[0]
            W_semseg = self.resize_shape[1]
    
            if H_semseg != H or W_semseg != W:
                x = F.interpolate(x, size=(H_semseg, W_semseg), mode="bilinear", align_corners=False)
        except:
            pass
        
        return self.model(x, **kwargs)
    
    @torch.no_grad()
    def get_semseg_logits(self, images: torch.Tensor, **kwargs):
        """
        Run inference on a batch of images.

        Args:
            images (torch.Tensor): (B, 3, H, W), normalized input images

        Returns:
            pred_labels (torch.Tensor): (B, H, W), predicted class indices
            pred_rgb (torch.Tensor): (B, 3, H_out, W_out), predicted RGB masks
            input_rgb (torch.Tensor): (B, 3, H_out, W_out), renormalized input
        """
        self.eval()
        images = images.to(self.device)

        _,_,H,W = images.shape

        if self.resize_shape is not None:
            images_resized = F.interpolate(images, size=self.resize_shape, mode="bilinear", align_corners=False)
        else:
            images_resized = images

        logits = self(images_resized, **kwargs)

        logits = F.interpolate(logits, size=(H, W), mode="nearest")

        return logits
