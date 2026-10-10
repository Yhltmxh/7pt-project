"""Image encoders with explicit spatial-token and pooled-feature outputs.

ConvNeXt uses torchvision, matching ISIC2025/src/model.py. No old diagnostic
or MONET heads are loaded. Pretraining is selected by a concrete enum ID.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EncoderFeatures:
    tokens: Tensor  # [batch, height * width, feature_dim]
    pooled: Tensor  # [batch, feature_dim]
    spatial_shape: tuple[int, int]


class ImageEncoder(nn.Module):
    feature_dim: int

    @staticmethod
    def validate_image(image: Tensor) -> None:
        if not isinstance(image, Tensor) or image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("An image input must be a tensor with shape [B, 3, H, W]")
        if image.shape[0] == 0 or min(image.shape[-2:]) < 1:
            raise ValueError("Image inputs must have a nonempty batch and spatial dimensions")


class ConvNeXtBaseEncoder(ImageEncoder):
    """Independent torchvision ConvNeXt Base, without its ImageNet class head."""

    def __init__(self, pretrained_id: str = "none") -> None:
        super().__init__()
        # A tiny_cnn smoke run does not need torchvision or internet access.
        from torchvision.models import ConvNeXt_Base_Weights, convnext_base

        if pretrained_id == "none":
            weights = None
        elif pretrained_id == "ConvNeXt_Base_Weights.IMAGENET1K_V1":
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        else:
            raise ValueError(
                "convnext_base pretrained_id must be 'none' or "
                "'ConvNeXt_Base_Weights.IMAGENET1K_V1'; DEFAULT is not a frozen ID"
            )
        backbone = convnext_base(weights=weights)
        self.feature_dim = int(backbone.classifier[-1].in_features)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.pool_normalization = nn.Sequential(*list(backbone.classifier.children())[:-1])
        self.pretrained_id = pretrained_id

    def forward(self, image: Tensor) -> EncoderFeatures:
        self.validate_image(image)
        if min(image.shape[-2:]) < 32:
            raise ValueError("convnext_base requires image height and width of at least 32")
        feature_map = self.features(image)
        pooled = self.pool_normalization(self.avgpool(feature_map))
        return EncoderFeatures(
            tokens=feature_map.flatten(2).transpose(1, 2),
            pooled=pooled,
            spatial_shape=tuple(feature_map.shape[-2:]),
        )


class TinyCNNEncoder(ImageEncoder):
    """Small random encoder for synthetic plumbing checks, not a research baseline."""

    def __init__(self, feature_dim: int = 32, pretrained_id: str = "none") -> None:
        super().__init__()
        if pretrained_id != "none":
            raise ValueError("tiny_cnn is a smoke-test encoder and has no pretrained weights")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("tiny_feature_dim must be a positive integer")
        self.feature_dim = feature_dim
        self.pretrained_id = pretrained_id
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, image: Tensor) -> EncoderFeatures:
        self.validate_image(image)
        feature_map = self.features(image)
        return EncoderFeatures(
            tokens=feature_map.flatten(2).transpose(1, 2),
            pooled=feature_map.mean(dim=(-2, -1)),
            spatial_shape=tuple(feature_map.shape[-2:]),
        )


def build_encoder(name: str, pretrained_id: str, *, tiny_feature_dim: int = 32) -> ImageEncoder:
    if name == "convnext_base":
        return ConvNeXtBaseEncoder(pretrained_id=pretrained_id)
    if name == "tiny_cnn":
        return TinyCNNEncoder(feature_dim=tiny_feature_dim, pretrained_id=pretrained_id)
    raise ValueError(f"Unsupported encoder {name!r}; available: convnext_base, tiny_cnn")
