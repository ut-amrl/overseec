from typing import TYPE_CHECKING

from transformers import SegformerForSemanticSegmentation

from overseec.modules.semseg.model_base import Model_Base
if TYPE_CHECKING:
    from overseec.modules.semseg.semseg_config import SemSegConfig, SegFormerConfig

class SegFormer(Model_Base):
    semseg_config: "SemSegConfig" = None
    model_config : "SegFormerConfig" = None
    
    def __init__(self, 
                 semseg_config : "SemSegConfig",
                 model_config : "SegFormerConfig"):
        super().__init__(semseg_config=semseg_config, model_config=model_config)

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.model_config.model_name,
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )

        if self.model_config.freeze_encoder:
            for name, param in self.model.named_parameters():
                if self.model_config.unfrozen_tag not in name:
                    param.requires_grad = False

    def forward(self, x, **kwargs):
        return self.model(pixel_values=x).logits