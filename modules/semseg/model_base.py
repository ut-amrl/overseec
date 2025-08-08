import torch.nn as nn
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from overseec.semseg_config import SemSegConfig, ModelConfig

class Model_Base(nn.Module):
    def __init__(self, 
                semseg_config: "SemSegConfig", 
                model_config: "ModelConfig"):
        
        super(Model_Base, self).__init__()
        self.semseg_config = semseg_config
        self.model_config = model_config
        self.device = semseg_config.device
        self.num_classes = semseg_config.num_classes

    def set_classes(self, classes):
        self.semseg_config.classes = classes
        self.semseg_config.reset()
        self.num_classes = self.semseg_config.num_classes