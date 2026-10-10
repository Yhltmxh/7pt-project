"""New Derm7pt diagnosis and optional seven categorical attribute heads."""

from torch import Tensor, nn


ATTRIBUTE_CLASSES = {
    "pigment_network": 3,
    "blue_whitish_veil": 2,
    "vascular_structures": 3,
    "pigmentation": 3,
    "streaks": 3,
    "dots_and_globules": 3,
    "regression_structures": 2,
}


class PredictionHeads(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 5,
                 attribute_head: bool = False, dropout: float = 0.0):
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("head_dropout must lie in [0, 1)")
        self.normalization = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.diagnosis = nn.Linear(feature_dim, num_classes)
        self.attributes = nn.ModuleDict(
            {name: nn.Linear(feature_dim, count) for name, count in ATTRIBUTE_CLASSES.items()}
            if attribute_head else {}
        )

    def forward(self, features: Tensor) -> dict:
        features = self.dropout(self.normalization(features))
        return {
            "diagnosis_logits": self.diagnosis(features),
            "attribute_logits": {name: head(features) for name, head in self.attributes.items()},
        }
