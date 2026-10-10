"""Map-style Dataset with separate model inputs, supervision, and identifiers."""

from pathlib import Path

from .labels import ATTRIBUTE_CLASSES
from .manifest import MODALITIES, read_manifest, resolve_image_path, validate_manifest_rows
from .splits import select_config_partition, select_partition
from .transforms import build_transforms


class Derm7ptDataset:
    """Single items have [2] masks, collated batches have bool [B, 2] masks.

    Only selected modality keys reach the model. Missing selected images fail;
    substitution status remains auditable and is never inferred or repaired.
    """

    def __init__(self, manifest, root, partition="train", modalities=MODALITIES, transform=None, cfg=None):
        if isinstance(modalities, str):
            modalities = (modalities,)
        self.modalities = tuple(modalities)
        if not self.modalities or len(set(self.modalities)) != len(self.modalities) or not set(self.modalities) <= set(MODALITIES):
            raise ValueError("modalities must select clinical and/or dermoscopic exactly once")
        self.root = Path(root)
        rows = read_manifest(manifest) if isinstance(manifest, (str, Path)) else validate_manifest_rows(list(manifest))
        self.rows = select_config_partition(rows, cfg, partition) if cfg is not None else select_partition(rows, partition)
        self.partition = partition
        self.transform = transform or build_transforms()
        if not self.rows:
            raise ValueError(f"No samples for partition {partition!r}")
        for row in self.rows:
            if row.get("exclusion_reason"):
                raise ValueError(f"Manifest contains excluded sample {row['sample_id']}; filtering must be an explicit frozen protocol")
            for modality in self.modalities:
                path = resolve_image_path(self.root, row[f"{modality}_path"])
                if not row[f"has_{modality}"] or path is None or not path.is_file():
                    raise ValueError(f"Selected modality missing: {row['sample_id']} {modality}")

    def __len__(self):
        return len(self.rows)

    @property
    def sample_ids(self):
        return [row["sample_id"] for row in self.rows]

    def __getitem__(self, index):
        try:
            import torch
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Derm7ptDataset image loading requires PyTorch and Pillow") from error
        row = self.rows[index]
        images = {}
        for modality in self.modalities:
            with Image.open(resolve_image_path(self.root, row[f"{modality}_path"])) as image:
                images[modality] = image.convert("RGB")
        inputs = self.transform(images)
        if set(inputs) != set(self.modalities):
            raise ValueError("Transform must return exactly the selected image modality keys")
        inputs["modality_mask"] = torch.tensor([name in self.modalities for name in MODALITIES], dtype=torch.bool)
        targets = {
            "diagnosis": torch.tensor(row["diagnosis_index"], dtype=torch.long),
            "attributes": {name: torch.tensor(row[f"attr_{name}"], dtype=torch.long) for name in ATTRIBUTE_CLASSES},
            "attribute_masks": {name: torch.tensor(row[f"attr_mask_{name}"], dtype=torch.bool) for name in ATTRIBUTE_CLASSES},
        }
        ids = {"sample_id": row["sample_id"], "lesion_id": row["lesion_id"], "source_row_index": row["source_row_index"]}
        return inputs, targets, ids
