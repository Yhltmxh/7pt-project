"""Explicit two-image preprocessing; no assumed spatial registration."""

import random

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PairedImageTransform:
    def __init__(self, config=None, training=False):
        config = dict(config or {})
        self.training = training
        if config.get("normalization", "imagenet") != "imagenet":
            raise ValueError("Baseline normalization must be imagenet")
        augmentation = config.get("train_augmentation", "flips")
        if augmentation not in ("flips", "none"):
            raise ValueError("Baseline train_augmentation must be flips or none")
        if config.get("inference_augmentation", "none") != "none":
            raise ValueError("Baseline inference_augmentation must be none")
        size = config.get("image_size", config.get("resize", 224))
        if isinstance(size, int):
            size = (size, size)
        if not isinstance(size, (tuple, list)) or len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size):
            raise ValueError("image_size must be a positive int or [height, width]")
        self.size = tuple(size)
        self.mean = tuple(config.get("mean", IMAGENET_MEAN))
        self.std = tuple(config.get("std", IMAGENET_STD))
        if len(self.mean) != 3 or len(self.std) != 3 or any(value <= 0 for value in self.std):
            raise ValueError("RGB normalization needs three means and positive std values")
        self.geometry_coupling = config.get("geometry_coupling", "shared")
        if self.geometry_coupling not in ("shared", "independent"):
            raise ValueError("geometry_coupling must be shared or independent")
        appearance = config.get("appearance_coupling", "none")
        if appearance != "none":
            raise ValueError("Baseline appearance_coupling must be none")
        # No stochastic appearance augmentation exists in this baseline.
        self.appearance_coupling = appearance
        self.hflip = float(config.get("horizontal_flip_probability", 0.5)) if training and augmentation == "flips" else 0.0
        self.vflip = float(config.get("vertical_flip_probability", 0.5)) if training and augmentation == "flips" else 0.0
        if not 0 <= self.hflip <= 1 or not 0 <= self.vflip <= 1:
            raise ValueError("Flip probabilities must lie in [0, 1]")

    def __call__(self, images):
        try:
            import torch
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Image tensor preprocessing requires PyTorch and Pillow") from error
        unknown = set(images) - {"clinical", "dermoscopic"}
        if unknown:
            raise ValueError(f"Unexpected image input fields: {sorted(unknown)}")
        shared_flips = (random.random() < self.hflip, random.random() < self.vflip)
        result = {}
        for name, image in images.items():
            image = image.convert("RGB").resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
            flips = shared_flips if self.geometry_coupling == "shared" else (random.random() < self.hflip, random.random() < self.vflip)
            if flips[0]:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if flips[1]:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(self.size[0], self.size[1], 3).permute(2, 0, 1).float().div_(255.0)
            mean = tensor.new_tensor(self.mean)[:, None, None]
            std = tensor.new_tensor(self.std)[:, None, None]
            result[name] = (tensor - mean) / std
        return result


def build_transforms(preprocess_config=None, training=False):
    return PairedImageTransform(preprocess_config, training)
