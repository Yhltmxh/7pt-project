"""Pooled baseline fusion and the registration boundary for later research modules.

Every fusion receives named token and pooled inputs. These three baselines use
pooled features only; mean fusion adds no spatial processing or learned token.
"""

from collections.abc import Callable

import torch
from torch import Tensor, nn


class FusionModule(nn.Module):
    output_dim: int

    def _validate(
        self,
        clinical_tokens: Tensor,
        dermoscopic_tokens: Tensor,
        clinical_pooled: Tensor,
        dermoscopic_pooled: Tensor,
        modality_mask: Tensor | None,
        metadata: Tensor | None,
    ) -> None:
        if metadata is not None:
            raise ValueError("Pooled image baselines do not accept metadata")
        if clinical_pooled.ndim != 2 or dermoscopic_pooled.ndim != 2:
            raise ValueError("Pooled fusion inputs must have shape [B, D]")
        if clinical_pooled.shape[0] != dermoscopic_pooled.shape[0]:
            raise ValueError("Modalities must have the same batch size")
        for tokens, pooled in (
            (clinical_tokens, clinical_pooled), (dermoscopic_tokens, dermoscopic_pooled)
        ):
            if tokens.ndim != 3 or tokens.shape[0] != pooled.shape[0] or tokens.shape[2] != pooled.shape[1]:
                raise ValueError("Tokens must have shape [B, N, D] matching pooled [B, D]")
        if modality_mask is not None:
            if modality_mask.dtype != torch.bool or tuple(modality_mask.shape) != (clinical_pooled.shape[0], 2):
                raise ValueError("modality_mask must be bool [B, 2] ordered clinical, dermoscopic")
            if not modality_mask.all().item():
                raise ValueError("Dual-modal baseline requires both modalities for every sample")


FUSION_REGISTRY: dict[str, Callable[..., FusionModule]] = {}


def register_fusion(name: str, factory: Callable[..., FusionModule] | None = None):
    """Register a factory accepting clinical_dim, dermoscopic_dim and output_dim.

Can be used as a decorator. A research plugin must return a FusionModule with
an output_dim and the same named forward inputs as the built-in modules.
"""
    def register(candidate):
        if not isinstance(name, str) or not name or name in FUSION_REGISTRY:
            raise ValueError(f"Fusion name must be nonempty and unregistered: {name!r}")
        FUSION_REGISTRY[name] = candidate
        return candidate

    return register(factory) if factory is not None else register


@register_fusion("mean")
class MeanFusion(FusionModule):
    def __init__(self, clinical_dim: int, dermoscopic_dim: int, output_dim: int | None = None):
        super().__init__()
        if clinical_dim != dermoscopic_dim:
            raise ValueError("Pooled mean fusion requires equal encoder feature dimensions")
        if output_dim is not None and output_dim != clinical_dim:
            raise ValueError("Pooled mean fusion preserves the encoder feature dimension")
        self.output_dim = clinical_dim

    def forward(self, *, clinical_tokens, dermoscopic_tokens, clinical_pooled, dermoscopic_pooled,
                modality_mask=None, metadata=None):
        self._validate(clinical_tokens, dermoscopic_tokens, clinical_pooled,
                       dermoscopic_pooled, modality_mask, metadata)
        return (clinical_pooled + dermoscopic_pooled) * 0.5, {}


@register_fusion("concat")
class ConcatFusion(FusionModule):
    def __init__(self, clinical_dim: int, dermoscopic_dim: int, output_dim: int | None = None):
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else clinical_dim
        if isinstance(self.output_dim, bool) or not isinstance(self.output_dim, int) or self.output_dim < 1:
            raise ValueError("Fusion output dimension must be a positive integer")
        self.projection = nn.Linear(clinical_dim + dermoscopic_dim, self.output_dim)

    def forward(self, *, clinical_tokens, dermoscopic_tokens, clinical_pooled, dermoscopic_pooled,
                modality_mask=None, metadata=None):
        self._validate(clinical_tokens, dermoscopic_tokens, clinical_pooled,
                       dermoscopic_pooled, modality_mask, metadata)
        return self.projection(torch.cat((clinical_pooled, dermoscopic_pooled), dim=1)), {}


@register_fusion("gated")
class GatedFusion(FusionModule):
    """One learned gate per feature channel, with no metadata input."""

    def __init__(self, clinical_dim: int, dermoscopic_dim: int, output_dim: int | None = None):
        super().__init__()
        if clinical_dim != dermoscopic_dim:
            raise ValueError("Gated fusion requires equal encoder feature dimensions")
        if output_dim is not None and output_dim != clinical_dim:
            raise ValueError("Gated fusion preserves the encoder feature dimension")
        self.output_dim = clinical_dim
        self.gate = nn.Linear(clinical_dim + dermoscopic_dim, clinical_dim)

    def forward(self, *, clinical_tokens, dermoscopic_tokens, clinical_pooled, dermoscopic_pooled,
                modality_mask=None, metadata=None):
        self._validate(clinical_tokens, dermoscopic_tokens, clinical_pooled,
                       dermoscopic_pooled, modality_mask, metadata)
        gate = torch.sigmoid(self.gate(torch.cat((clinical_pooled, dermoscopic_pooled), dim=1)))
        features = gate * clinical_pooled + (1.0 - gate) * dermoscopic_pooled
        return features, {"clinical_gate": gate.detach()}


def build_fusion(name: str, clinical_dim: int, dermoscopic_dim: int,
                 *, output_dim: int | None = None, parameters: dict | None = None) -> FusionModule:
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion {name!r}; available: {', '.join(sorted(FUSION_REGISTRY))}")
    module = FUSION_REGISTRY[name](clinical_dim=clinical_dim, dermoscopic_dim=dermoscopic_dim,
                                   output_dim=output_dim, **(parameters or {}))
    if not isinstance(module, FusionModule) or not isinstance(module.output_dim, int) or module.output_dim < 1:
        raise TypeError("A registered fusion must return a FusionModule with positive output_dim")
    return module
