import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import TYPE_CHECKING
from overseec.modules.utils import * 
import pytorch_lightning as pl
if TYPE_CHECKING:
    from overseec.semseg_config import SemSegConfig, Mask_RefinerConfig


class OVerSeeC_Mask_Refiner():
    def __init__(self, 
                 semseg_config: "SemSegConfig",
                 mask_refiner_model_config: "Mask_RefinerConfig"):
        super(OVerSeeC_Mask_Refiner, self).__init__()
        
        # self.semseg_config = semseg_config

        self.semseg_config = semseg_config
        self.mask_refiner_model_config = mask_refiner_model_config
        self.device = self.mask_refiner_model_config.device
    
    def set_model(self, model_class):
        self.mask_refiner_model = model_class(
                                            self.semseg_config,     
                                            self.mask_refiner_model_config
                                            )

    def set_classes(self, classes):
        self.semseg_config.classes = classes
        self.semseg_config.reset()
        self.num_classes = self.semseg_config.num_classes
    
    def get_refined_masks(self, batch_image_torch, logits_semseg, idx):
        logits = self.mask_refiner_model(batch_image_torch, logits_semseg, idx)

        return logits