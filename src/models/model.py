"""Independent image-only B01/B02 and pooled B04/B05/B06 architectures."""

from collections.abc import Iterator, Mapping

import torch
from torch import Tensor, nn

from .encoders import build_encoder
from .fusion import build_fusion
from .heads import PredictionHeads


class BaselineModel(nn.Module):
    def __init__(self, model_config: Mapping, num_classes: int = 5):
        super().__init__()
        config = dict(model_config)
        modalities = config.get("modalities", ["clinical", "dermoscopic"])
        if not isinstance(modalities, (list, tuple)) or not modalities:
            raise ValueError("model.modalities must list clinical and/or dermoscopic")
        if len(set(modalities)) != len(modalities) or set(modalities) - {"clinical", "dermoscopic"}:
            raise ValueError("model.modalities must contain distinct clinical/dermoscopic names")
        self.modalities = tuple(modalities)
        if config.get("independent_encoders", True) is not True:
            raise ValueError("These baselines require independent encoders")
        for key in ("metadata_attention", "metadata_late_fusion", "metadata_tokens"):
            if config.get(key, False) not in (False, None):
                raise ValueError(f"{key} is not implemented for image-only baselines")
        if config.get("refinement", "none") not in ("none", None):
            raise ValueError("Refinement belongs to a separately registered research module")
        if config.get("cross_attention_direction") is not None:
            raise ValueError("Pooled baselines do not perform spatial cross-attention")
        mechanism = config.get("mechanism") or {}
        if mechanism.get("name", "none") not in ("none", None) or mechanism.get("parameters"):
            raise ValueError("A research mechanism is not part of the first baseline implementation")
        # Null/omitted weights intentionally disable downloads (also on resume).
        pretrained_id = config.get("pretrained_id")
        if pretrained_id is None:
            pretrained_id = "none"
        tiny_dim = config.get("tiny_feature_dim", 32)
        self.clinical_encoder = (
            build_encoder(config.get("clinical_encoder", "convnext_base"), pretrained_id,
                          tiny_feature_dim=tiny_dim) if "clinical" in modalities else None
        )
        self.dermoscopic_encoder = (
            build_encoder(config.get("dermoscopic_encoder", "convnext_base"), pretrained_id,
                          tiny_feature_dim=tiny_dim) if "dermoscopic" in modalities else None
        )
        self.fusion_name = config.get("fusion", "mean")
        self.fusion = None
        if len(self.modalities) == 2:
            self.fusion = build_fusion(
                self.fusion_name, self.clinical_encoder.feature_dim, self.dermoscopic_encoder.feature_dim,
                output_dim=config.get("fusion_dim"), parameters=config.get("fusion_parameters"),
            )
            self.feature_dim = self.fusion.output_dim
        else:
            if self.fusion_name not in ("single", "mean", "none", None):
                raise ValueError("A single-modal baseline does not use a learned fusion module")
            self.feature_dim = (self.clinical_encoder or self.dermoscopic_encoder).feature_dim
        declared_dim = config.get("feature_dim")
        if declared_dim is not None and declared_dim != self.feature_dim:
            raise ValueError(f"Configured feature_dim={declared_dim} differs from encoder/fusion output {self.feature_dim}")
        attribute_head = config.get("attribute_head", False)
        if not isinstance(attribute_head, bool):
            raise ValueError("attribute_head must be a boolean")
        self.heads = PredictionHeads(self.feature_dim, num_classes=num_classes,
                                     attribute_head=attribute_head, dropout=config.get("head_dropout", 0.0))

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        for encoder in (self.clinical_encoder, self.dermoscopic_encoder):
            if encoder is not None:
                yield from encoder.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        if self.fusion is not None:
            yield from self.fusion.parameters()
        yield from self.heads.parameters()

    def forward(self, *, clinical: Tensor | None = None, dermoscopic: Tensor | None = None,
                modality_mask: Tensor | None = None, metadata: Tensor | None = None,
                metadata_mask: Tensor | None = None, return_analysis: bool = False) -> dict:
        # Named arguments deliberately reject targets and unknown batch fields.
        if metadata is not None or metadata_mask is not None:
            raise ValueError("Image-only baselines require metadata=None and metadata_mask=None")
        images = {"clinical": clinical, "dermoscopic": dermoscopic}
        batch_size = None
        for modality in self.modalities:
            image = images[modality]
            if image is None:
                raise ValueError(f"Required {modality} image is missing")
            if not isinstance(image, Tensor) or image.ndim != 4:
                raise ValueError(f"{modality} must be a tensor with shape [B, 3, H, W]")
            if batch_size is not None and image.shape[0] != batch_size:
                raise ValueError("Required modalities have different batch sizes")
            batch_size = image.shape[0]
        if modality_mask is not None:
            if modality_mask.dtype != torch.bool or tuple(modality_mask.shape) != (batch_size, 2):
                raise ValueError("modality_mask must be bool [B, 2] ordered clinical, dermoscopic")
            columns = [0 if name == "clinical" else 1 for name in self.modalities]
            if not modality_mask[:, columns].all().item():
                raise ValueError("A required modality is unavailable; first baselines use complete required inputs")
        clinical_features = self.clinical_encoder(clinical) if self.clinical_encoder is not None else None
        dermoscopic_features = self.dermoscopic_encoder(dermoscopic) if self.dermoscopic_encoder is not None else None
        fusion_analysis = {}
        if self.fusion is not None:
            fused, fusion_analysis = self.fusion(
                clinical_tokens=clinical_features.tokens, dermoscopic_tokens=dermoscopic_features.tokens,
                clinical_pooled=clinical_features.pooled, dermoscopic_pooled=dermoscopic_features.pooled,
                modality_mask=modality_mask, metadata=None,
            )
        else:
            fused = (clinical_features or dermoscopic_features).pooled
        outputs = self.heads(fused)
        if return_analysis:
            outputs["analysis"] = {
                "modalities": self.modalities,
                "fusion": self.fusion_name if self.fusion is not None else "none",
                "clinical_tokens_shape": tuple(clinical_features.tokens.shape) if clinical_features else None,
                "dermoscopic_tokens_shape": tuple(dermoscopic_features.tokens.shape) if dermoscopic_features else None,
                **fusion_analysis,
            }
        return outputs


def build_model(model_config: dict, num_classes: int = 5) -> BaselineModel:
    """Build from the resolved model subsection; no target or metadata input path."""
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a configuration mapping")
    return BaselineModel(model_config, num_classes=num_classes)
