import os
import torch
from dataclasses import dataclass, field
from overseec.semseg_config import *
from torchvision.transforms import ToTensor, Normalize, Compose


model_config_dict = {
    "dinounet": DinoUNetConfig(),
    "segformer": SegFormerConfig(),
    "clipseg": CLIPSegConfig(),
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

@dataclass 
class Satellite_2_masks_config:
    device :str  = "cuda:2"
    semseg_tile_size : tuple = (512, 512)
    semseg_stride : int = 256

    mask_refiner_tile_size : tuple = (512, 512)
    mask_refiner_stride : int = 256

    image_2_tiles_threads = 8
    raw_logit_joiner_threads = 8


@dataclass
class AllConfig:
    cmap_device : str = "cuda:2"
    sam_device : str  = "cuda:2"
    semseg_device : str = "cuda:2"

    model_name : str = "clipseg"
    mask_refiner_name : str = "samrefiner"

    model_ckpt : str = None

    # default classes - will be replaced by LLM output
    classes: dict = field(default_factory=lambda: {
        "road": (128, 64, 128),
        "tree": (0, 128, 0),
        "grass": (0, 128, 128),
        "building": (0, 0, 128),
        "trail or footway": (88, 112, 164),
        "water": (221, 191, 166),
    })

    classes_semseg_knobs : dict = field(default_factory=lambda: {
        "road": 0.3,
        "tree": 0.6,
        "grass": 0.6,
        "building": 0.6,
        "trail or footway": 0.3,
        "water": 0.6,
    })   

    img_transform : Compose = None
    img_format : str = "RGB"



    semseg_tile_size : tuple = (512, 512)
    semseg_stride : int = 256
    semseg_tile_combine_method : str = "max"

    mask_refiner_tile_size : tuple = (512, 512)
    mask_refiner_stride : int = 256
    mask_refiner_tile_combine_method : str = "max"
    use_negative_points : bool = True

    sam_model : str = "vit_h"


    def reset(self,):
        self.__post_init__()

    def __post_init__(self, ):
        if self.model_ckpt == "":
            raise ValueError
        
        if self.model_name not in model_config_dict.keys():
            raise KeyError
        

        
        use_default_classes = True
        if self.model_name == "clipseg":
            use_default_classes = False
            self.semseg_config = SemSegConfig(
                device = self.semseg_device,
                classes = self.classes,
                classes_semseg_knobs = self.classes_semseg_knobs,
                use_default_classes =use_default_classes
            )
        else:
            self.semseg_config = SemSegConfig(
                device = self.semseg_device,
                classes = self.classes,
                classes_semseg_knobs = self.classes_semseg_knobs,
                default_classes= self.classes,
                default_classes_semseg_knobs = self.classes_semseg_knobs,
                use_default_classes = use_default_classes
            )


        self.model_config = model_config_dict[self.model_name]

        self.sat_2_cmap_config = Satellite_2_masks_config(
            device = self.cmap_device,
            semseg_tile_size=self.semseg_tile_size,
            semseg_stride=self.semseg_stride,
            mask_refiner_tile_size=self.mask_refiner_tile_size,
            mask_refiner_stride=self.mask_refiner_stride,
        )

        if self.model_name != "clipseg":
            self.img_transform = Compose(
                [
                    ToTensor(),
                    Normalize(mean=MEAN, std=STD)
                ]
            )
        
        self.mask_refiner_config = SAMRefiner_Config(
            device=self.sam_device,
            use_negative_points=self.use_negative_points,
            sam_model=self.sam_model,
        )
