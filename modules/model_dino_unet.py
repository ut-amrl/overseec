import torch
import torch.nn as nn
import torch.hub
import torch.nn.functional as F
from molmo_testing.models.semseg.model_base import Model_Base
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from molmo_testing.models.semseg_config import SemSegConfig, DinoUNetConfig


class DinoUNet(Model_Base):

    # TYPE CHECKINGs
    semseg_config: "SemSegConfig" = None
    model_config : "DinoUNetConfig" = None

    def __init__(self, 
                 semseg_config: "SemSegConfig", 
                 model_config: "DinoUNetConfig"):
        super().__init__(semseg_config=semseg_config, model_config=model_config)
        
        self.patch_size = self.model_config.patch_size

        self.encoder = torch.hub.load(self.model_config.model_source, 
                                      self.model_config.model_name)
        
        self.dino_dim = self.model_config.dino_dim  # DINO output dimension

        for param in self.encoder.parameters():
            param.requires_grad = False  # Freeze DINO


        self.proj1 = nn.Conv2d(self.dino_dim, 512, kernel_size=1)
        self.proj2 = nn.Conv2d(self.dino_dim, 256, kernel_size=1)
        self.proj3 = nn.Conv2d(self.dino_dim, 128, kernel_size=1)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)

        self.resize1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # Resize enc2
        self.resize2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)  # Resize enc3

        self.final_upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.final_skip_proj = nn.Conv2d(128, 64, kernel_size=1)  # Reduce `enc3_resized` channels to 64
        self.final_conv = nn.Conv2d(128, self.num_classes, kernel_size=1)  # Final output
    
    def extract_patches(self, x):
        """Splits the input image into non-overlapping patches."""
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image size must be divisible by patch size"

        # Reshape into patches: (B, C, num_patches, patch_size, patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, num_patches_y, num_patches_x, C, patch_size, patch_size)
        x = x.view(-1, C, self.patch_size, self.patch_size)  # Merge batch and patch dims (B * num_patches, C, P, P)

        return x

    def reconstruct_from_patches(self, patch_features, H, W):
        """Reconstructs the spatial feature map from patch-wise DINO embeddings."""
        B = patch_features.shape[0] // ((H // self.patch_size) * (W // self.patch_size))
        num_patches_y = H // self.patch_size
        num_patches_x = W // self.patch_size

        # Reshape back to spatial grid
        patch_features = patch_features.view(B, num_patches_y, num_patches_x, self.dino_dim)
        patch_features = patch_features.permute(0, 3, 1, 2).contiguous()  # (B, 768, num_patches_y, num_patches_x)

        return patch_features

    def forward(self, x, **kwargs):
        """Forward pass through DINO + UNet."""
        B, C, H, W = x.shape

        patches = self.extract_patches(x)

        with torch.no_grad():
            patch_features = self.encoder(patches)

        feature_map = self.reconstruct_from_patches(patch_features, H, W)

        enc1 = self.proj1(feature_map)  # (B, 512, 28, 28)
        enc2 = self.proj2(feature_map)  # (B, 256, 28, 28)
        enc3 = self.proj3(feature_map)  # (B, 128, 28, 28)

        x = self.up1(enc1)  # (B, 256, 56, 56)
        enc2_resized = F.interpolate(enc2, size=(56, 56), mode="bilinear", align_corners=False)
        x = torch.cat([x, enc2_resized], dim=1)  # Skip connection

        x = self.up2(x)  # (B, 128, 112, 112)
        enc3_resized = F.interpolate(enc3, size=(112, 112), mode="bilinear", align_corners=False)
        x = torch.cat([x, enc3_resized], dim=1)  # Skip connection

        x = self.up3(x)  # (B, 64, 224, 224)

        enc3_resized_final = F.interpolate(enc3_resized, size=(224, 224), mode="bilinear", align_corners=False)

        enc3_resized_final = self.final_skip_proj(enc3_resized_final)

        output = self.final_conv(torch.cat([x, enc3_resized_final], dim=1))
        return output
