from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


class AddGaussianNoise:
    def __init__(self, std: float = 0.0):
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


def build_eval_transform(image_size: int):
    """Preprocessing shared by validation and ONNX inference."""
    return transforms.Compose([
        transforms.Resize(
            (int(image_size), int(image_size)),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_transforms(cfg: dict):
    image_size = int(cfg["data"]["image_size"])
    aug = cfg["augmentation"]

    train_ops: list[Callable] = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=float(aug["horizontal_flip"])),
        transforms.RandomRotation(float(aug["rotation_degrees"])),
        transforms.ColorJitter(
            brightness=float(aug["brightness"]),
            contrast=float(aug["contrast"]),
            saturation=float(aug["saturation"]),
            hue=float(aug["hue"]),
        ),
        transforms.ToTensor(),
        AddGaussianNoise(float(aug["gaussian_noise_std"])),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    return transforms.Compose(train_ops), build_eval_transform(image_size)


def build_datasets(cfg: dict):
    root = Path(cfg["data"]["root"])
    train_tf, eval_tf = build_transforms(cfg)

    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

    # An unlabeled test directory may contain image files directly without
    # class subdirectories. ImageFolder cannot load that layout, so use the
    # labeled validation split as the fallback loader. The flat test folder
    # remains available through test.py --unlabeled-dir.
    has_test_class_dirs = test_dir.exists() and any(
        path.is_dir() for path in test_dir.iterdir()
    )
    if has_test_class_dirs:
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    else:
        test_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

    return train_ds, val_ds, test_ds


def build_loaders(cfg: dict):
    train_ds, val_ds, test_ds = build_datasets(cfg)
    batch_size = int(cfg["data"]["batch_size"])
    workers = int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"]["pin_memory"])

    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )

    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
        train_ds.classes,
    )
