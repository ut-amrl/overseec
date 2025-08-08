import os
from dataclasses import dataclass, field

import numpy as np

import overseec

@dataclass
class SemSegConfig:
    device : str = "cuda:0"
    
    # data
    default_classes: dict = field(default_factory=lambda: {
        "road": (128, 64, 128),
        "tree": (0, 128, 0),
        "grass": (0, 128, 128),
        "building": (0, 0, 128),
        "trail or footway": (88, 112, 164),
        "water": (221, 191, 166),
    })

    default_classes_semseg_knobs : dict = field(default_factory=lambda: {
        "road": 0.3,
        "tree": 0.6,
        "grass": 0.6,
        "building": 0.6,
        "trail or footway": 0.3,
        "water": 0.6,
        })

    classes : dict = None
    classes_prefs : dict = None
    classes_semseg_knobs : dict = None


    use_default_classes : bool = True
    
    traversible_classes : list = field(default_factory=lambda : ["road", "grass"])  # Roads are traversible # NOT USED, just needed for dataloader
    image_dir : str = "../../unified_dataset/images"
    label_dir : str = "../../unified_dataset/labels"

    lr : float = 1e-4
    weight_decay :float = 1e-5

    batch_size = 16
    num_workers = 2
    pin_memory = True

    tile_combine_method : str = "max"

    def reset(self,):
        self.__post_init__()
        
    def __post_init__(self):
        if self.use_default_classes:
            self.classes = self.default_classes
            # self.class_prefs = self.default_classes_prefs
            self.classes_semseg_knobs = self.default_classes_semseg_knobs
            
        self.num_classes = len(self.classes)
        self.num_def_classes = len(self.default_classes)

        


@dataclass
class ModelConfig:
    resize_shape : tuple = None
    input_de_normalize : bool = True   
    mdoel_ckpt : str = ""

@dataclass
class DinoUNetConfig(ModelConfig):
    model_name : str = "dino_vitb8"
    model_source : str = "facebookresearch/dino:main"
    dino_dim : int = 768

    patch_size : int = 8
    resize_shape : tuple = (224, 224)

@dataclass
class SegFormerConfig(ModelConfig):
    model_name : str = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_classes : int = 6
    freeze_encoder : bool = True
    unfrozen_tag : str = "decode_head"

    resize_shape :tuple = (512, 512)


@dataclass
class CLIPSegConfig(ModelConfig):
    model_name : str = "CIDAS/clipseg-rd64-refined"
    run_type : str = "semseg"  # prompt or semseg

    resize_shape_dummy : tuple = (352, 352)
    img_size : int = 352
    input_de_normalize : bool = False



@dataclass
class Mask_RefinerConfig():
    device : str = "cuda:0"

    # mask refiner model
    no_prompt_logit = np.nan

    # Exemplar point extractor
    sample_fraction : float = 0.008
    neg_sample_fraction : float = 0.001
    exemplar_num_points : int = 100
    use_negative_points : bool = True


    # un-tiling 
    tile_combine_method : str = "max"

    



@dataclass
class SAM_Exemplar_Config(Mask_RefinerConfig):
    ckpt_fname : str = "sam_vit_h_4b8939.pth"
    model_type : str = "vit_h"

    tile_combine_method : str = "max"

    ############################
    # this is used during inference in the whole satellite pipeline.
    # wont be used during metric, test of semantic segmentation
    batch_size = 1
    num_workers = 8
    pin_memory = True
    ############################


    def __post_init__(self,):

        module_dir =  os.path.dirname(os.path.abspath(overseec.__file__))
        self.ckpt_path = f"{module_dir}/checkpoints/{self.ckpt_fname}"

@dataclass
class SAMRefiner_Config(Mask_RefinerConfig):

    sam_model : str = "vit_h"

    tile_combine_method : str = "max"

    ############################
    # this is used during inference in the whole satellite pipeline.
    # wont be used during metric, test of semantic segmentation
    batch_size = 1
    num_workers = 8
    pin_memory = True
    ############################


    def __post_init__(self,):

        if self.sam_model == "vit_h":
            self.ckpt_fname : str = "sam_vit_h_4b8939.pth"
            self.model_type : str = "vit_h"
        elif self.sam_model == "vit_b":
            self.ckpt_fname : str = "sam_vit_b_01ec64.pth"
            self.model_type : str = "vit_b"

        module_dir =  os.path.dirname(os.path.abspath(overseec.__file__))
        self.ckpt_path = f"{module_dir}/checkpoints/{self.ckpt_fname}"