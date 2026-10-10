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
    if (preprocess_config or {}).get("train_augmentation") == "isic2025":
        return AlbumentationsPairedTransform(preprocess_config, training)
    return PairedImageTransform(preprocess_config, training)


class AlbumentationsPairedTransform:
    """Source-inspired policy; per-call seeds derive from checkpointed Python RNG."""

    def __init__(self, config, training=False):
        import albumentations as A
        import cv2
        from albumentations.pytorch import ToTensorV2
        if A.__version__ != "2.0.8":
            raise ValueError("isic2025 augmentation requires albumentations==2.0.8")
        if config["geometry_coupling"] != "shared" or config["appearance_coupling"] != "shared":
            raise ValueError("isic2025 augmentation requires shared geometry and appearance")
        if config["normalization"] != "imagenet" or config["inference_augmentation"] != "none":
            raise ValueError("Expected ImageNet normalization and deterministic inference")
        self.training = training
        self.geometry_coupling = self.appearance_coupling = "shared"
        size, aug = config["image_size"], config["isic2025"]
        # Derm7pt modalities can have different sizes. Normalize the canvas before
        # sharing affine/crop coordinates; this does not imply spatial registration.
        self.resize = A.Resize(size, size, interpolation=cv2.INTER_LINEAR)
        operations = []
        if training:
            operations.extend([
                A.HorizontalFlip(p=aug["horizontal_flip_probability"]),
                A.VerticalFlip(p=aug["vertical_flip_probability"]),
                A.Affine(translate_percent=aug["translate_percent"], scale=tuple(aug["scale"]),
                         rotate=aug["rotate"], border_mode=cv2.BORDER_CONSTANT, fill=0,
                         interpolation=cv2.INTER_LINEAR, p=aug["affine_probability"]),
                A.RandomResizedCrop(size=(size, size), scale=tuple(aug["crop_scale"]),
                                    ratio=tuple(aug["crop_ratio"]), interpolation=cv2.INTER_LINEAR,
                                    p=aug["crop_probability"]),
                A.RandomBrightnessContrast(brightness_limit=aug["brightness_limit"],
                                           contrast_limit=aug["contrast_limit"], p=aug["brightness_contrast_probability"]),
                A.HueSaturationValue(hue_shift_limit=aug["hue_shift_limit"],
                                     sat_shift_limit=aug["saturation_shift_limit"],
                                     val_shift_limit=aug["value_shift_limit"], p=aug["hsv_probability"]),
                A.OneOf([A.GaussianBlur(blur_limit=aug["blur_limit"]),
                         A.GaussNoise(std_range=tuple(aug["noise_std_range"]))], p=aug["blur_noise_probability"]),
            ])
        operations.extend([A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
                           A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
        self.pipeline = A.Compose(operations, additional_targets={"image2": "image"}, seed=0, strict=True)

    def __call__(self, images):
        import numpy as np
        if not images or set(images) - {"clinical", "dermoscopic"}:
            raise ValueError("Expected clinical and/or dermoscopic image inputs only")
        names = [name for name in ("clinical", "dermoscopic") if name in images]
        # Pre-resize each differently sized image before Compose's shape check.
        # Use the pipeline's resize operation so training and inference agree.
        arrays = [self.resize(image=np.array(images[name].convert("RGB")))["image"] for name in names]
        if self.training:
            self.pipeline.set_random_seed(random.getrandbits(32))
        transformed = self.pipeline(image=arrays[0], **({"image2": arrays[1]} if len(names) == 2 else {}))
        return {name: transformed[key].contiguous() for name, key in zip(names, ("image", "image2"))}
