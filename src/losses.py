"""Diagnostic CE / class-balanced focal loss and masked categorical attributes.

Effective-number weighting and FP32 log softmax follow ISIC2025/src/loss.py,
rewritten for the new outputs/targets contract. No MONET or mechanism loss.
"""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CBFocalLoss(nn.Module):
    def __init__(self, class_counts, beta: float = 0.99, gamma: float = 2.0):
        super().__init__()
        if not math.isfinite(beta) or not 0.0 <= beta < 1.0:
            raise ValueError("cb_beta must lie in [0, 1)")
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError("focal_gamma must be finite and nonnegative")
        counts = torch.tensor(list(class_counts), dtype=torch.float64)
        if counts.ndim != 1 or counts.numel() < 2 or not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("class_counts must be a finite nonnegative vector with at least 2 classes")
        # expm1 avoids cancellation for a beta very close to 1; keep the final
        # FP32 weight buffer to match the FP32 classification arithmetic.
        if beta == 0.0:
            alpha = torch.ones_like(counts)
        else:
            effective_number = -torch.expm1(counts.clamp_min(1) * math.log(beta))
            alpha = (1.0 - beta) / effective_number
        self.register_buffer("alpha", (alpha / alpha.mean()).float())
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.alpha.numel():
            raise ValueError("CB Focal class counts must match the diagnostic logits")
        log_probs = F.log_softmax(logits.float(), dim=1)
        log_pt = log_probs.gather(1, targets[:, None]).squeeze(1)
        return (-self.alpha[targets] * (1.0 - log_pt.exp()) ** self.gamma * log_pt).mean()


class BaselineLoss(nn.Module):
    """Attribute reduction: mean CE per valid task, then mean across valid tasks."""

    def __init__(self, loss_config: Mapping | None = None, class_counts=None):
        super().__init__()
        config = dict(loss_config or {})
        diagnosis = config.get("diagnosis", "cross_entropy")
        if diagnosis == "cross_entropy":
            self.diagnosis_loss = nn.CrossEntropyLoss()
        elif diagnosis in ("cb_focal", "class_balanced_focal"):
            if class_counts is None:
                raise ValueError("CB Focal requires counts from the current training partition")
            if config.get("cb_beta") is None or config.get("focal_gamma") is None:
                raise ValueError("Resolve cb_beta and focal_gamma before enabling CB Focal")
            self.diagnosis_loss = CBFocalLoss(class_counts, beta=config["cb_beta"], gamma=config["focal_gamma"])
        else:
            raise ValueError(f"Unsupported diagnosis loss {diagnosis!r}")
        self.attribute_lambda = float(config.get("attribute_lambda", 0.0))
        if not math.isfinite(self.attribute_lambda) or self.attribute_lambda < 0:
            raise ValueError("attribute_lambda must be finite and nonnegative")
        if config.get("attribute_type", "categorical_cross_entropy") != "categorical_cross_entropy":
            raise ValueError("Seven-point attributes use categorical cross entropy")
        if config.get("attribute_reduction", "mean_valid_tasks") != "mean_valid_tasks":
            raise ValueError("Only mean_valid_tasks attribute reduction is implemented")
        if config.get("mechanism_lambda", 0.0) not in (0, 0.0, None):
            raise ValueError("No mechanism loss is implemented in the baseline")

    def forward(self, outputs: Mapping, targets: Mapping) -> dict[str, Tensor]:
        logits = outputs["diagnosis_logits"]
        labels = targets["diagnosis"]
        if logits.ndim != 2 or labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("Diagnosis requires logits [B, C] and targets [B]")
        if labels.dtype != torch.long:
            raise ValueError("Diagnosis labels must be torch.long")
        diagnosis = self.diagnosis_loss(logits.float(), labels)
        attributes = logits.new_zeros((), dtype=torch.float32)
        if self.attribute_lambda > 0:
            attribute_logits = outputs.get("attribute_logits", {})
            if not attribute_logits:
                raise ValueError("Positive attribute_lambda requires enabled attribute heads")
            attribute_targets = targets.get("attributes", {})
            masks = targets.get("attribute_masks", {})
            valid_task_losses = []
            for name, prediction in attribute_logits.items():
                if name not in attribute_targets or name not in masks:
                    raise ValueError(f"Attribute {name!r} requires both targets and an explicit validity mask")
                target, mask = attribute_targets[name], masks[name]
                if (prediction.ndim != 2 or prediction.shape[0] != logits.shape[0]
                        or target.shape != labels.shape or mask.shape != labels.shape):
                    raise ValueError(f"Invalid batch shapes for attribute {name!r}")
                if target.dtype != torch.long or mask.dtype != torch.bool:
                    raise ValueError(f"Attribute {name!r} requires long targets and bool masks")
                if mask.any().item():
                    valid_task_losses.append(F.cross_entropy(prediction[mask].float(), target[mask]))
            if valid_task_losses:
                attributes = torch.stack(valid_task_losses).mean()
        return {"total": diagnosis + self.attribute_lambda * attributes,
                "diagnosis": diagnosis, "attributes": attributes}


def build_loss(loss_config: dict, class_counts=None) -> BaselineLoss:
    return BaselineLoss(loss_config, class_counts=class_counts)
