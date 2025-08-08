import os
import overseec
from dataclasses import dataclass
import numpy as np

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