"""Adapt pretrained SAM 3 to OVerSeeC's tiled semantic segmentation interface."""

from contextlib import nullcontext
from pathlib import Path

from PIL import Image
import torch
from torch.nn.functional import interpolate

from .model_base import Model_Base


class SAM3(Model_Base):
    def __init__(self, semseg_config, model_config):
        super().__init__(semseg_config, model_config)
        checkpoint = Path(model_config.checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise ValueError("Set SAM3Config.checkpoint_path to a local sam3.pt file.")
        confidence = model_config.confidence_threshold
        if not 0.0 < confidence < 1.0:
            raise ValueError("SAM 3 confidence_threshold must be between 0 and 1.")
        self.tile_batch_size = model_config.tile_batch_size
        if type(self.tile_batch_size) is not int or self.tile_batch_size < 1:
            raise ValueError("SAM 3 tile_batch_size must be a positive integer.")

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Build on CPU, then place the model on the exact configured device.
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint), load_from_HF=False,
            device="cpu", eval_mode=True, compile=False,
        ).to(self.device).eval()
        self.processor = Sam3Processor(
            self.model, device=str(self.device), confidence_threshold=confidence,
        )

    @torch.inference_mode()
    def forward(self, images, **kwargs):
        """Return float32 [batch, classes, height, width] logits in class order."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("SAM 3 expects [batch, 3, height, width] RGB tiles.")
        classes = list(self.semseg_config.classes)
        if not classes or images.shape[0] == 0:
            raise ValueError("SAM 3 requires at least one tile and one class prompt.")
        # OVerSeeC supplies RGB values in [0, 255], typically as float tensors.
        # PIL lets the SAM 3 processor handle its own resize and normalization.
        rgb_tiles = images.to(device="cpu", dtype=torch.uint8).permute(0, 2, 3, 1)
        context = nullcontext()
        device = torch.device(self.device)
        if device.type == "cuda":
            with torch.cuda.device(device):
                if torch.cuda.is_bf16_supported():
                    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        batch_maps = []
        with context:
            for tiles in rgb_tiles.split(self.tile_batch_size):
                if len(tiles) == 1:
                    batch_maps.append(self._predict_tile(tiles[0], classes).unsqueeze(0))
                else:
                    batch_maps.append(self._predict_batch(tiles, classes))
        probabilities = torch.cat(batch_maps, dim=0)
        # run_semseg() expects logits and applies sigmoid after stitching.
        return torch.logit(probabilities.clamp(1e-6, 1.0 - 1e-6))

    def _predict_tile(self, rgb, classes):
        height, width = rgb.shape[:2]
        state = self.processor.set_image(Image.fromarray(rgb.numpy()))
        class_maps = []
        for prompt in classes:
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt=prompt, state=state)

            # In the pinned Meta processor, this field holds probabilities.
            # Its instance scores already include the model's presence factor.
            class_maps.append(self._combine_instances(
                state["masks_logits"][:, 0], state["scores"], (height, width),
            ))
        return torch.stack(class_maps)

    def _predict_batch(self, tiles, classes):
        """Encode the group once and ground one text prompt across it at a time."""
        from sam3.model.data_misc import FindStage

        height, width = tiles.shape[1:3]
        images = [Image.fromarray(tile.numpy()) for tile in tiles]
        image_features = self.processor.set_image_batch(images)["backbone_out"]
        image_ids = torch.arange(len(images), device=self.device)
        find_input = FindStage(
            img_ids=image_ids, text_ids=torch.zeros_like(image_ids),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        class_maps = []
        for prompt in classes:
            text_features = self.model.backbone.forward_text([prompt], device=self.device)
            output = self.model.forward_grounding(
                backbone_out={**image_features, **text_features},
                find_input=find_input,
                find_target=None,
                geometric_prompt=self.model._get_dummy_prompt(num_prompts=len(images)),
            )
            # Mirror the pinned single-image processor before aggregating each tile.
            presence = output["presence_logit_dec"].sigmoid().unsqueeze(1)
            scores = (output["pred_logits"].sigmoid() * presence).squeeze(-1)
            tile_maps = []
            for masks, tile_scores in zip(output["pred_masks"], scores):
                keep = tile_scores > self.processor.confidence_threshold
                masks = masks[keep, None]
                if masks.numel():
                    masks = interpolate(masks, size=(height, width), mode="bilinear",
                                        align_corners=False).sigmoid()
                tile_maps.append(self._combine_instances(masks[:, 0], tile_scores[keep],
                                                         (height, width)))
            class_maps.append(torch.stack(tile_maps))
        return torch.stack(class_maps, dim=1)

    def _combine_instances(self, masks, scores, size):
        class_map = torch.zeros(size, device=self.device, dtype=torch.float32)
        for mask, score in zip(masks, scores):
            class_map = torch.maximum(class_map, mask.float() * score.float())
        return class_map
