from typing import TYPE_CHECKING

import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from overseec.modules.semseg.model_base import Model_Base
if TYPE_CHECKING:
    from overseec.modules.semseg.semseg_config import SemSegConfig, CLIPSegConfig


class CLIPSeg(Model_Base):
    semseg_config: "SemSegConfig" = None
    model_config : "CLIPSegConfig" = None
    
    def __init__(self,
                 semseg_config: "SemSegConfig",
                 model_config: "CLIPSegConfig"):
        super().__init__(semseg_config=semseg_config, model_config=model_config)
        
        # Load CLIPSeg model
        self.processor = CLIPSegProcessor.from_pretrained(self.model_config.model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_config.model_name).to(self.device)

    def forward(self, images, **kwargs):
        prompts = kwargs.get("prompts", None)
        if self.model_config.run_type == "prompt":
            return self.inference_from_prompts(images, prompts)
        elif self.model_config.run_type == "semseg":
            return self.inference_semseg(images)
        
    def inference_semseg(self, images):
        B,_,H,W = images.shape
        class_names = list(self.semseg_config.classes.keys())
        images_expanded = images.repeat_interleave(len(class_names), dim=0)  # Repeat for each class
        prompts = class_names * images.shape[0]
        logits = self.inference_from_prompts(images_expanded, prompts, batch_size=B)  # shape: (B * C, 1, H, W)

        return logits  # Shape: (B, C, H, W)

    def inference_from_prompts(self, images, prompts, batch_size):
        _,_, H, W = images.shape
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to GPU

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        raw_logits = outputs.logits
        raw_logits = raw_logits.reshape(batch_size, -1, self.model_config.resize_shape_dummy[0], self.model_config.resize_shape_dummy[0])

        return raw_logits