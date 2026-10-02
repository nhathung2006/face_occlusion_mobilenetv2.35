from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import yaml
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install the project requirements in .venv first."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config_4class_single_logit.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
sys.path.insert(0, str(PROJECT_ROOT))
from src.datasets.dataset import AddGaussianNoise  # noqa: E402
from src.models.mobilenetv2 import MobileNetV2  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.training import build_optimizer, build_scheduler  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = (
        "model", "data", "task", "preprocessing", "training",
        "augmentation", "inference", "runtime", "evaluation", "paths",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config is missing sections: {missing}")
    names = config["data"].get("class_names", [])
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("This script requires four unique subclass names.")
    binary_names = config["task"].get("binary_class_names", [])
    mapping = config["task"].get("binary_label_by_subclass", {})
    if len(binary_names) != 2 or set(mapping) != set(names):
        raise ValueError("task must map every configured subclass to two binary class names.")
    if set(mapping.values()) != set(binary_names):
        raise ValueError("Every configured binary class must have at least one subclass.")
    if binary_names != ["clear", "occluded"]:
        raise ValueError("The binary output order must be [clear, occluded] for sigmoid-logit semantics.")
    for key in ("normalization_mean", "normalization_std"):
        if len(config["preprocessing"].get(key, [])) != 3:
            raise ValueError(f"preprocessing.{key} must contain three channel values.")
    return config


def image_files(folder: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_samples(config: dict) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], dict]:
    data_cfg = config["data"]
    root = Path(data_cfg["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    class_names = list(data_cfg["class_names"])
    manifest_value = data_cfg.get("split_manifest")
    split_root_value = data_cfg.get("split_root")
    if split_root_value and not manifest_value:
        raise ValueError("data.split_root requires data.split_manifest.")
    if manifest_value:
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["class_names"] != class_names:
            raise ValueError("Split manifest class order does not match config.")
        records = {r["sha256"]: r for r in manifest["images"]}
        if len(records) != len(manifest["images"]):
            raise ValueError("Duplicate image hashes in split manifest.")
        group_splits = {}
        for record in records.values():
            if record["split"] not in ("train", "val"):
                raise ValueError("Split manifest contains an invalid split name.")
            group = record["group_id"]
            if group in group_splits and group_splits[group] != record["split"]:
                raise ValueError("Similar-image group appears in both train and val.")
            group_splits[group] = record["split"]
        found = {}
        if split_root_value:
            split_root = Path(split_root_value)
            if not split_root.is_absolute():
                split_root = PROJECT_ROOT / split_root
            split_root = split_root.resolve()
            expected_paths = set()
            for record in manifest["images"]:
                class_name = record["class_name"]
                if class_name not in class_names:
                    raise ValueError(f"Unknown class in split manifest: {class_name}")
                source_relative = Path(record["path"])
                if source_relative.is_absolute() or not source_relative.parts or source_relative.parts[0] != class_name:
                    raise ValueError(f"Invalid image path in split manifest: {source_relative}")
                path = (split_root / record["split"] / source_relative).resolve()
                if not path.is_relative_to(split_root / record["split"] / class_name):
                    raise ValueError(f"Split image escapes its class folder: {path}")
                if path in expected_paths or not path.is_file():
                    raise ValueError(f"Missing or duplicate image in fixed split: {path}")
                if file_sha256(path) != record["sha256"]:
                    raise ValueError(f"Fixed split image changed: {path}")
                expected_paths.add(path)
                found[record["sha256"]] = (path, class_names.index(class_name))
            actual_paths = {
                path.resolve()
                for split in ("train", "val")
                for class_name in class_names
                for path in image_files(split_root / split / class_name)
            }
            if actual_paths != expected_paths:
                raise ValueError("Fixed split folders contain extra or missing images.")
        else:
            for label, class_name in enumerate(class_names):
                for path in image_files(root / class_name):
                    digest = file_sha256(path)
                    if digest in found:
                        raise ValueError(f"Duplicate file appeared after split preparation: {path}")
                    record = records.get(digest)
                    if record is None or record["class_name"] != class_name:
                        raise ValueError(f"Dataset or label changed since split preparation: {path}")
                    found[digest] = (path, label)
        if set(found) != set(records):
            raise ValueError("Dataset images are missing from the fixed split. Prepare a new experiment.")
        sets = {"train": [], "val": []}
        for record in manifest["images"]:
            sets[record["split"]].append(found[record["sha256"]])
        counts = {name: {"total_unique": 0, "train": 0, "val": 0} for name in class_names}
        for split, samples in sets.items():
            for _, label in samples:
                counts[class_names[label]][split] += 1
                counts[class_names[label]]["total_unique"] += 1
        if any(not row["train"] or not row["val"] for row in counts.values()):
            raise ValueError("Fixed split must contain train and val images for each class.")
        return sets["train"], sets["val"], counts
    val_fraction = float(data_cfg["validation_fraction"])
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = random.Random(int(config["training"].get("seed", 42)))
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    counts: dict[str, dict[str, int]] = {}
    digest_to_class: dict[str, str] = {}

    for label, class_name in enumerate(class_names):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        # Collapse byte-identical copies for training only; source files stay untouched.
        unique_files: list[Path] = []
        seen_in_class: set[str] = set()
        for path in image_files(class_dir):
            digest = file_sha256(path)
            previous_class = digest_to_class.get(digest)
            if previous_class is not None and previous_class != class_name:
                raise ValueError(
                    "The same exact image appears under two different labels: "
                    f"{previous_class!r} and {class_name!r}; review labels before training."
                )
            digest_to_class[digest] = class_name
            if digest not in seen_in_class:
                unique_files.append(path)
                seen_in_class.add(digest)

        if len(unique_files) < 2:
            raise ValueError(f"Need at least two unique images in {class_dir}.")
        rng.shuffle(unique_files)
        val_count = max(1, round(len(unique_files) * val_fraction))
        val_count = min(val_count, len(unique_files) - 1)
        val_files = unique_files[:val_count]
        train_files = unique_files[val_count:]
        train_samples.extend((path, label) for path in train_files)
        val_samples.extend((path, label) for path in val_files)
        counts[class_name] = {
            "total_unique": len(unique_files),
            "train": len(train_files),
            "val": len(val_files),
        }

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples, counts


class FaceClassDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def make_transforms(config: dict):
    data_cfg = config["data"]
    aug = config["augmentation"]
    prep = config["preprocessing"]
    size = int(data_cfg["image_size"])
    interpolation_name = str(prep["resize_interpolation"]).lower()
    interpolation = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }.get(interpolation_name)
    if interpolation is None:
        raise ValueError(f"Unsupported resize interpolation: {interpolation_name}")
    mean, std = tuple(prep["normalization_mean"]), tuple(prep["normalization_std"])
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=interpolation,
                antialias=bool(prep["antialias"]),
            ),
            transforms.RandomHorizontalFlip(p=float(aug["horizontal_flip"])),
            transforms.RandomRotation(float(aug["rotation_degrees"])),
            transforms.ColorJitter(
                brightness=float(aug["brightness"]),
                contrast=float(aug["contrast"]),
                saturation=float(aug["saturation"]),
                hue=float(aug.get("hue", 0.0)),
            ),
            transforms.ToTensor(),
            AddGaussianNoise(float(aug.get("gaussian_noise_std", 0.0))),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=interpolation,
                antialias=bool(prep["antialias"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, eval_transform


class BinaryWithSubtypeAuxiliary(nn.Module):
    """One binary head plus a four-way auxiliary head used only during training."""

    def __init__(
        self,
        backbone: nn.Module,
        dropout: float,
        auxiliary_classes: int,
        binary_logit_limit: float | None = None,
    ):
        super().__init__()
        self.features = backbone.features
        self.conv = backbone.conv
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        self.binary_logit_limit = (
            float(binary_logit_limit) if binary_logit_limit is not None else None
        )
        if self.binary_logit_limit is not None and self.binary_logit_limit <= 0:
            raise ValueError("binary_logit_limit must be positive when enabled.")
        self.classifier_aux = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.classifier[-1].in_features, auxiliary_classes),
        )

    def shared_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.conv(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared_features(images)
        binary_logits = self.classifier(features)
        # Keep the model's single scalar output in a finite, quantization-safe
        # range while retaining a differentiable path for training.  With
        # max_logit=3.8918203, sigmoid(logit) is strictly between 0.02 and 0.98.
        if self.binary_logit_limit is not None:
            limit = self.binary_logit_limit
            binary_logits = limit * torch.tanh(binary_logits / limit)
        return binary_logits, self.classifier_aux(features)


def build_model(config: dict, load_pretrained: bool = True) -> nn.Module:
    model_cfg = config["model"]
    width = float(model_cfg["width_mult"])
    model = MobileNetV2(num_classes=1000, width_mult=width)
    if load_pretrained and bool(model_cfg["pretrained"]):
        weights_path = Path(model_cfg["pretrained_path"])
        if not weights_path.is_absolute():
            weights_path = PROJECT_ROOT / weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(f"ImageNet weights not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=float(model_cfg["dropout"])),
        nn.Linear(in_features, 1),
    )
    logit_penalty_cfg = config.get("training", {}).get("logit_penalty", {})
    binary_logit_limit = (
        float(logit_penalty_cfg["max_logit"])
        if bool(logit_penalty_cfg.get("bounded_output", False))
        else None
    )
    return BinaryWithSubtypeAuxiliary(
        model,
        dropout=float(model_cfg["dropout"]),
        auxiliary_classes=len(config["data"]["class_names"]),
        binary_logit_limit=binary_logit_limit,
    )


def set_trainable(model: nn.Module, unfreeze_from_feature: int | None) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if unfreeze_from_feature is None:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        for parameter in model.classifier_aux.parameters():
            parameter.requires_grad = True
        return
    if not 0 <= unfreeze_from_feature < len(model.features):
        raise ValueError(
            f"unfreeze_from_feature must be in [0, {len(model.features) - 1}]."
        )
    for index, layer in enumerate(model.features):
        if index >= unfreeze_from_feature:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    for parameter in model.conv.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier_aux.parameters():
        parameter.requires_grad = True


def freeze_batch_norm_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def subclass_to_binary_ids(config: dict) -> list[int]:
    binary_names = list(config["task"]["binary_class_names"])
    mapping = config["task"]["binary_label_by_subclass"]
    return [binary_names.index(mapping[name]) for name in config["data"]["class_names"]]


def binary_targets_from_subclasses(labels: torch.Tensor, mapping_ids: list[int]) -> torch.Tensor:
    mapping = torch.tensor(mapping_ids, device=labels.device)
    return mapping[labels].to(dtype=torch.float32)


def macro_f1(labels: list[int], predictions: list[int], n_classes: int) -> tuple[float, float]:
    accuracy = sum(y == p for y, p in zip(labels, predictions)) / max(1, len(labels))
    class_scores = []
    for class_id in range(n_classes):
        tp = sum(y == class_id and p == class_id for y, p in zip(labels, predictions))
        fp = sum(y != class_id and p == class_id for y, p in zip(labels, predictions))
        fn = sum(y == class_id and p != class_id for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return accuracy, sum(class_scores) / n_classes


def build_auxiliary_class_weights(
    train_counts: torch.Tensor,
    weighting: str,
) -> torch.Tensor:
    """Build four-class CE weights from training counts only."""
    if train_counts.ndim != 1 or train_counts.numel() == 0:
        raise ValueError("train_counts must be a non-empty one-dimensional tensor.")
    if torch.any(train_counts <= 0):
        raise ValueError("Every auxiliary class must have at least one training image.")

    mode = str(weighting).strip().lower()
    if mode == "inverse_frequency":
        return train_counts.sum() / (train_counts.numel() * train_counts)
    if mode == "inverse_sqrt_frequency":
        # A common scale factor does not change weighted CE with mean reduction.
        # Normalize to the first configured class so the logged values are easy
        # to interpret (clear_full_face is 1.0 in the current configuration).
        weights = train_counts.rsqrt()
        return weights / weights[0]
    raise ValueError(
        "Supported auxiliary_class_weighting values: "
        "inverse_frequency, inverse_sqrt_frequency"
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    subclass_to_binary: list[int],
    auxiliary_class_weights: torch.Tensor,
    logit_penalty_cfg: dict,
    target_smoothing_cfg: dict,
    auxiliary_label_smoothing: float,
    auxiliary_loss_weight: float,
    binary_threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    batch_scheduler=None,
) -> tuple[float, float, float, float, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        freeze_batch_norm_stats(model)
    # Keep epoch aggregates and metric predictions on the active device. Calling
    # .item()/.cpu() in every batch forces CUDA to synchronize repeatedly.
    loss_sums = torch.zeros(2, dtype=torch.float64, device=device)
    count = 0
    metric_batches: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        binary_labels = binary_targets_from_subclasses(labels, subclass_to_binary)
        targets = binary_labels
        if bool(target_smoothing_cfg.get("enabled", False)):
            low = float(target_smoothing_cfg["target_low"])
            high = float(target_smoothing_cfg["target_high"])
            targets = torch.where(binary_labels > 0.5, high, low)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            binary_logits, auxiliary_logits = model(images)
            logits = binary_logits.squeeze(1)
            probabilities = torch.sigmoid(logits)
            per_sample_loss = F.binary_cross_entropy(
                probabilities, targets, reduction="none"
            )
            classification_loss = per_sample_loss.mean()
            auxiliary_loss = F.cross_entropy(
                auxiliary_logits,
                labels,
                weight=auxiliary_class_weights,
                label_smoothing=auxiliary_label_smoothing,
            )
            loss = classification_loss + auxiliary_loss_weight * auxiliary_loss
            # Include the same regularization term in train/validation reporting;
            # validation computes it without gradients, so both losses are comparable.
            if bool(logit_penalty_cfg.get("enabled", False)):
                max_logit = float(logit_penalty_cfg["max_logit"])
                margin_weight = float(logit_penalty_cfg["margin_weight"])
                l2_weight = float(logit_penalty_cfg["l2_weight"])
                if margin_weight > 0 and not bool(logit_penalty_cfg.get("bounded_output", False)):
                    excess = F.relu(logits.abs() - max_logit)
                    loss = loss + margin_weight * excess.square().mean()
                if l2_weight > 0:
                    loss = loss + l2_weight * logits.square().mean()
            if training:
                loss.backward()
                optimizer.step()
                if batch_scheduler is not None:
                    batch_scheduler.step()

        batch_count = int(labels.size(0))
        loss_sums[0] += loss.detach().to(torch.float64) * batch_count
        loss_sums[1] += auxiliary_loss.detach().to(torch.float64) * batch_count
        count += batch_count
        metric_batches.append(
            torch.stack(
                (
                    binary_labels.detach().long(),
                    (probabilities.detach() >= binary_threshold).long(),
                    labels.detach().long(),
                    auxiliary_logits.detach().argmax(dim=1).long(),
                )
            )
        )

    if metric_batches:
        metric_values = torch.cat(metric_batches, dim=1).reshape(-1).to(torch.float64)
    else:
        metric_values = torch.empty(0, dtype=torch.float64, device=device)
    # Transfer the epoch losses and all predictions/labels in one synchronization.
    epoch_values = torch.cat((loss_sums, metric_values)).cpu().tolist()
    loss_sum, auxiliary_loss_sum = epoch_values[:2]
    metric_values = [int(value) for value in epoch_values[2:]]
    labels_all = metric_values[:count]
    predictions_all = metric_values[count:2 * count]
    aux_labels_all = metric_values[2 * count:3 * count]
    aux_predictions_all = metric_values[3 * count:4 * count]

    accuracy, f1 = macro_f1(labels_all, predictions_all, n_classes=2)
    auxiliary_accuracy, auxiliary_f1 = macro_f1(
        aux_labels_all, aux_predictions_all, n_classes=auxiliary_class_weights.numel()
    )
    return (
        loss_sum / max(1, count),
        auxiliary_loss_sum / max(1, count),
        accuracy,
        f1,
        auxiliary_accuracy,
        auxiliary_f1,
    )


def checkpoint_payload(
    model: nn.Module,
    config: dict,
    epoch: int,
    val_f1: float,
    split_counts: dict,
    auxiliary_class_weights: torch.Tensor,
) -> dict:
    logit_penalty_cfg = config["training"].get("logit_penalty", {})
    return {
        "model_state_dict": model.state_dict(),
        "class_names": list(config["data"]["class_names"]),
        "binary_class_names": list(config["task"]["binary_class_names"]),
        "subclass_to_binary": subclass_to_binary_ids(config),
        "binary_threshold": float(config["inference"]["occluded_threshold"]),
        "auxiliary_loss_weight": float(config["training"]["auxiliary_loss_weight"]),
        "auxiliary_class_weighting": str(config["training"]["auxiliary_class_weighting"]),
        "auxiliary_class_weights": auxiliary_class_weights.detach().cpu().tolist(),
        "binary_logit_bounded": bool(logit_penalty_cfg.get("bounded_output", False)),
        "binary_logit_limit": (
            float(logit_penalty_cfg["max_logit"])
            if logit_penalty_cfg.get("max_logit") is not None
            else None
        ),
        "width_mult": float(config["model"]["width_mult"]),
        "dropout": float(config["model"]["dropout"]),
        "image_size": int(config["data"]["image_size"]),
        "normalization_mean": tuple(config["preprocessing"]["normalization_mean"]),
        "normalization_std": tuple(config["preprocessing"]["normalization_std"]),
        "epoch": epoch,
        "val_macro_f1": val_f1,
        "split_counts": split_counts,
    }


def train(config: dict) -> Path:
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    train_samples, val_samples, split_counts = split_samples(config)
    print("Per-class split after ignoring exact duplicate copies:")
    for name in config["data"]["class_names"]:
        row = split_counts[name]
        print(
            f"  {name}: total={row['total_unique']}, "
            f"train={row['train']}, val={row['val']}"
        )

    train_transform, eval_transform = make_transforms(config)
    workers = int(config["data"]["num_workers"])
    batch_size = int(config["data"]["batch_size"])
    configured_device = str(config["runtime"]["train_device"]).lower()
    if configured_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(configured_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("runtime.train_device requests CUDA, but CUDA is unavailable.")
    print(f"Device: {device}")
    train_loader = DataLoader(
        FaceClassDataset(train_samples, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=bool(config["data"]["pin_memory"]) and device.type == "cuda",
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        FaceClassDataset(val_samples, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config["data"]["pin_memory"]) and device.type == "cuda",
        persistent_workers=workers > 0,
    )

    class_names = list(config["data"]["class_names"])
    logit_penalty_cfg = config["training"]["logit_penalty"]
    target_smoothing_cfg = config["training"]["target_smoothing"]
    auxiliary_label_smoothing = float(config["training"]["auxiliary_label_smoothing"])
    auxiliary_loss_weight = float(config["training"]["auxiliary_loss_weight"])
    train_counts = torch.tensor(
        [split_counts[name]["train"] for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    auxiliary_weighting = str(config["training"]["auxiliary_class_weighting"])
    auxiliary_class_weights = build_auxiliary_class_weights(
        train_counts,
        auxiliary_weighting,
    )
    binary_threshold = float(config["inference"]["occluded_threshold"])
    subclass_to_binary = subclass_to_binary_ids(config)
    print("Main task: 0=clear, 1=occluded; auxiliary labels: " + ", ".join(
        f"{index}={name}" for index, name in enumerate(class_names)
    ))
    print(f"Auxiliary loss weight: {auxiliary_loss_weight:g}")
    print(f"Auxiliary class weighting: {auxiliary_weighting}")
    if bool(logit_penalty_cfg.get("bounded_output", False)):
        max_logit = float(logit_penalty_cfg["max_logit"])
        low_probability = float(torch.sigmoid(torch.tensor(-max_logit)))
        high_probability = float(torch.sigmoid(torch.tensor(max_logit)))
        print(
            "Binary output: one bounded scalar logit, no sigmoid in the model; "
            f"range=(-{max_logit:g}, {max_logit:g}), "
            f"sigmoid range=({low_probability:.6f}, {high_probability:.6f})"
        )
    for class_name, class_count, class_weight in zip(
        class_names,
        train_counts.detach().cpu().tolist(),
        auxiliary_class_weights.detach().cpu().tolist(),
    ):
        print(
            f"  {class_name}: train={int(class_count)} "
            f"weight={class_weight:.6f}"
        )
    model = build_model(config, load_pretrained=True).to(device)
    train_cfg = config["training"]
    epochs = int(train_cfg["epochs"])
    two_stage_cfg = train_cfg.get("two_stage", {})
    two_stage_enabled = bool(two_stage_cfg.get("enabled", False))
    stage1_epochs = int(two_stage_cfg["stage1_epochs"]) if two_stage_enabled else 0
    stage1_lr = float(two_stage_cfg["stage1_lr"])
    unfreeze_from_feature = int(two_stage_cfg["unfreeze_from_feature_index"])
    stage2_backbone_lr = float(two_stage_cfg["stage2_backbone_lr"])
    stage2_classifier_lr = float(two_stage_cfg["stage2_classifier_lr"])
    warmup_epochs = int(two_stage_cfg["warmup_epochs"])
    warmup_start_factor = float(two_stage_cfg["warmup_start_factor"])
    early_cfg = train_cfg["early_stopping"]
    patience = int(early_cfg["patience"])
    min_delta = float(early_cfg["min_delta"])
    early_enabled = bool(early_cfg["enabled"])
    early_monitor = str(early_cfg["monitor"]).lower()
    early_mode = str(early_cfg["mode"]).lower()
    if early_monitor not in ("val_f1", "val_macro_f1", "val_loss", "val_accuracy"):
        raise ValueError(f"Unsupported early-stopping monitor for this script: {early_monitor}")
    if early_mode not in ("min", "max"):
        raise ValueError("early_stopping.mode must be 'min' or 'max'.")
    scheduler_monitor = str(train_cfg["plateau"]["monitor"]).lower()
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / config["paths"]["best_checkpoint_name"]
    history_path = checkpoint_dir / config["paths"]["history_name"]
    history: list[dict] = []
    best_f1 = float("inf") if early_mode == "min" else float("-inf")
    best_epoch = 0
    waiting = 0
    optimizer = None
    scheduler = None
    scheduler_step_type = None

    for epoch in range(1, epochs + 1):
        if epoch == 1 and stage1_epochs > 0:
            set_trainable(model, None)
            optimizer = build_optimizer(model, config, lr=stage1_lr)
            scheduler, scheduler_step_type = build_scheduler(
                optimizer,
                config,
                len(train_loader),
                lr=stage1_lr,
                total_epochs=stage1_epochs,
            )
        elif epoch == stage1_epochs + 1:
            set_trainable(model, unfreeze_from_feature)
            remaining_epochs = max(1, epochs - stage1_epochs)
            optimizer = build_optimizer(
                model,
                config,
                backbone_lr=stage2_backbone_lr,
                classifier_lr=stage2_classifier_lr,
            )
            scheduler, scheduler_step_type = build_scheduler(
                optimizer,
                config,
                len(train_loader),
                lr=stage2_classifier_lr,
                total_epochs=remaining_epochs,
                warmup_epochs=warmup_epochs,
                warmup_start_factor=warmup_start_factor,
            )
        if optimizer is None:
            raise RuntimeError("Optimizer was not initialized; check stage1_epochs.")

        batch_scheduler = scheduler if scheduler_step_type == "batch" else None
        train_loss, train_aux_loss, train_acc, train_f1, train_aux_acc, train_aux_f1 = run_epoch(
            model,
            train_loader,
            device,
            subclass_to_binary,
            auxiliary_class_weights,
            logit_penalty_cfg,
            target_smoothing_cfg,
            auxiliary_label_smoothing,
            auxiliary_loss_weight,
            binary_threshold,
            optimizer,
            batch_scheduler,
        )
        val_loss, val_aux_loss, val_acc, val_f1, val_aux_acc, val_aux_f1 = run_epoch(
            model,
            val_loader,
            device,
            subclass_to_binary,
            auxiliary_class_weights,
            logit_penalty_cfg,
            target_smoothing_cfg,
            auxiliary_label_smoothing,
            auxiliary_loss_weight,
            binary_threshold,
        )
        if scheduler is not None and scheduler_step_type != "batch":
            if scheduler_step_type in ("plateau", "plateau_to_cosine", "plateau_then_cosine", "cosine_plateau", "hybrid"):
                scheduler_metric = {
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "val_macro_f1": val_f1,
                    "val_accuracy": val_acc,
                }.get(scheduler_monitor, val_loss)
                scheduler.step(scheduler_metric)
            else:
                scheduler.step()
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        stage_tag = "S1" if stage1_epochs > 0 and epoch <= stage1_epochs else "S2"
        monitored_value = {
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_macro_f1": val_f1,
            "val_accuracy": val_acc,
        }[early_monitor]
        improved = (
            monitored_value > best_f1 + min_delta
            if early_mode == "max"
            else monitored_value < best_f1 - min_delta
        )
        displayed_best = monitored_value if improved else best_f1
        displayed_best_epoch = epoch if improved else best_epoch
        monitor_label = "val F1" if early_monitor in ("val_f1", "val_macro_f1") else early_monitor
        best_marker = "NEW BEST" if improved else "best"
        print(
            f"Epoch {epoch:03d}/{epochs} [{stage_tag}] lr={learning_rates} | "
            f"train loss={train_loss:.4f} (sub={train_aux_loss:.4f}) "
            f"acc={train_acc:.3f} F1={train_f1:.3f} subAcc={train_aux_acc:.3f} "
            f"subF1={train_aux_f1:.3f} | "
            f"val loss={val_loss:.4f} (sub={val_aux_loss:.4f}) "
            f"acc={val_acc:.3f} F1={val_f1:.3f} subAcc={val_aux_acc:.3f} "
            f"subF1={val_aux_f1:.3f} | "
            f"{best_marker} {monitor_label}={displayed_best:.4f} "
            f"(epoch {displayed_best_epoch})"
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auxiliary_loss": train_aux_loss,
            "train_accuracy": train_acc,
            "train_auxiliary_accuracy": train_aux_acc,
            "train_macro_f1": train_f1,
            "train_auxiliary_macro_f1": train_aux_f1,
            "val_loss": val_loss,
            "val_auxiliary_loss": val_aux_loss,
            "val_accuracy": val_acc,
            "val_auxiliary_accuracy": val_aux_acc,
            "val_macro_f1": val_f1,
            "val_auxiliary_macro_f1": val_aux_f1,
            "stage": stage_tag,
            "learning_rates": str(learning_rates),
        }
        history.append(row)
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        payload = checkpoint_payload(
            model,
            config,
            epoch,
            val_f1,
            split_counts,
            auxiliary_class_weights,
        )
        if improved:
            best_f1, best_epoch, waiting = monitored_value, epoch, 0
            torch.save(payload, best_path)
        else:
            waiting += 1
        if early_enabled and (not two_stage_enabled or epoch > stage1_epochs) and waiting >= patience:
            print(
                f"Early stopping at best epoch {best_epoch}; "
                f"{early_monitor}={best_f1:.4f}"
            )
            break

    if not best_path.is_file():
        raise RuntimeError("Training did not produce a best checkpoint.")
    print(f"Best checkpoint: {best_path}")
    return best_path


def export_onnx(
    config: dict,
    checkpoint_path: Path,
    device_override: str | None = None,
) -> Path:
    runtime_cfg = config.get("runtime", {})
    configured_device = str(
        device_override or runtime_cfg.get("export_device", "auto")
    ).lower()
    if configured_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(configured_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "ONNX export requested CUDA, but torch.cuda.is_available() is false."
            )

    print(f"ONNX export device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("class_names") != list(config["data"]["class_names"]):
        raise ValueError("Checkpoint class order does not match this config.")
    if payload.get("binary_class_names") != list(config["task"]["binary_class_names"]):
        raise ValueError("Checkpoint does not contain the expected binary clear/occluded task.")
    model = build_model(config, load_pretrained=False).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    max_logit = float(config["training"]["logit_penalty"]["max_logit"])
    if not math.isfinite(max_logit) or max_logit <= 0:
        raise ValueError("training.logit_penalty.max_logit must be a positive finite number.")

    class BinaryLogitOnly(nn.Module):
        def __init__(self, trained_model: nn.Module, limit: float):
            super().__init__()
            self.trained_model = trained_model
            self.limit = limit

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            binary_logits, _ = self.trained_model(images)
            return torch.clamp(binary_logits, min=-self.limit, max=self.limit)

    export_model = BinaryLogitOnly(model, max_logit).to(device).eval()
    output_path = Path(config["paths"]["onnx_path"])
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = int(payload["image_size"])
    dummy = torch.zeros(1, 3, size, size, dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    torch.onnx.export(
        export_model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=int(config["paths"]["onnx_opset"]),
        do_constant_folding=True,
        input_names=[str(config["paths"]["onnx_input_name"])],
        output_names=[str(config["paths"]["onnx_output_name"])],
        dynamic_axes=(
            {str(config["paths"]["onnx_input_name"]): {0: "batch"},
             str(config["paths"]["onnx_output_name"]): {0: "batch"}}
            if bool(config["paths"]["onnx_dynamic_batch"]) else None
        ),
        dynamo=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    import onnx

    onnx.checker.check_model(onnx.load(str(output_path)))
    print(
        f"ONNX (scalar logit clamped to [-{max_logit}, {max_logit}], "
        f"no sigmoid): {output_path}"
    )
    return output_path


def classification_report(labels: list[int], predictions: list[int], class_names: list[str]) -> dict:
    matrix = [[0 for _ in class_names] for _ in class_names]
    for truth, prediction in zip(labels, predictions):
        matrix[truth][prediction] += 1
    per_class = {}
    f1_values = []
    for index, name in enumerate(class_names):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(class_names)) if row != index)
        fn = sum(matrix[index][column] for column in range(len(class_names)) if column != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(matrix[index])}
        f1_values.append(f1)
    return {
        "accuracy": sum(matrix[i][i] for i in range(len(class_names))) / max(1, len(labels)),
        "macro_f1": sum(f1_values) / len(class_names),
        "class_names": class_names,
        "confusion_matrix": matrix,
        "per_class": per_class,
        "sample_count": len(labels),
    }


def save_confusion_matrix(report: dict, output_path: Path, title: str, dpi: int) -> None:
    matrix = report["confusion_matrix"]
    names = report["class_names"]
    figure, axis = plt.subplots(figsize=(max(6, len(names) * 1.25), max(5, len(names))))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(names)), yticks=range(len(names)),
             xticklabels=names, yticklabels=names, xlabel="Predicted", ylabel="True", title=title)
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    threshold = max((max(row) for row in matrix), default=0) / 2
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center",
                      color="white" if matrix[row][column] > threshold else "black")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_validation_accuracy_chart(binary_report: dict, subclass_report: dict,
                                  output_path: Path, dpi: int) -> None:
    names = ["Clear / occluded (2 classes)", "Subtype (4 classes)"]
    values = [float(binary_report["accuracy"]), float(subclass_report["accuracy"])]
    figure, axis = plt.subplots(figsize=(8, 3.5))
    bars = axis.barh(names, values, color=["#4C78A8", "#F58518"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Validation accuracy")
    axis.set_title("Best checkpoint validation accuracy")
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(min(value + 0.015, 0.94), bar.get_y() + bar.get_height() / 2,
                  f"{value:.2%}", va="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_history_plot(history_path: Path, output_path: Path, title: str,
                      train_loss_key: str, val_loss_key: str,
                      train_f1_key: str, val_f1_key: str,
                      train_acc_key: str, val_acc_key: str,
                      width: float, height: float, dpi: int) -> None:
    if not history_path.is_file():
        return
    with history_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(width * 1.35, height))
    for key, label in ((train_loss_key, "Train"), (val_loss_key, "Validation")):
        if key in rows[0]:
            axes[0].plot(epochs, [float(row[key]) for row in rows], label=label)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for key, label in ((train_f1_key, "Train"), (val_f1_key, "Validation")):
        if key in rows[0]:
            axes[1].plot(epochs, [float(row[key]) for row in rows], label=label)
    axes[1].set(title="Macro F1", xlabel="Epoch", ylabel="F1", ylim=(0, 1.02))
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    if train_acc_key in rows[0] and val_acc_key in rows[0]:
        axes[2].plot(epochs, [float(row[train_acc_key]) for row in rows], label="Train")
        axes[2].plot(epochs, [float(row[val_acc_key]) for row in rows], label="Validation")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "Accuracy history not recorded", ha="center", va="center",
                     transform=axes[2].transAxes)
    axes[2].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    axes[2].grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def save_error_outputs(records: list[dict], output_dir: Path, prefix: str,
                       sheet_cfg: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}_misclassified.csv"
    fields = ["image_path", "true_label", "predicted_label", "confidence", "binary_prediction", "subclass_prediction"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    if not records:
        print(f"{prefix}: no validation errors; CSV saved at {csv_path}")
        return
    thumb_w = int(sheet_cfg["thumbnail_width"])
    thumb_h = int(sheet_cfg["thumbnail_height"])
    text_h = int(sheet_cfg["caption_height"])
    columns, rows = int(sheet_cfg["columns"]), int(sheet_cfg["rows"])
    per_page = columns * rows
    font = ImageFont.load_default()
    for page_start in range(0, len(records), per_page):
        page_records = records[page_start:page_start + per_page]
        sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + text_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(page_records):
            image_index = page_start + offset
            col, row = offset % columns, offset // columns
            x, y = col * thumb_w, row * (thumb_h + text_h)
            try:
                with Image.open(record["image_path"]) as source:
                    thumb = source.convert("RGB")
                    thumb.thumbnail((thumb_w - 8, thumb_h - 8))
                sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y + (thumb_h - thumb.height) // 2))
            except (OSError, ValueError):
                draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline="red")
                draw.text((x + 5, y + 5), "Image could not be opened", fill="red", font=font)
            caption = (f"{image_index + 1}. True: {record['true_label']}\n"
                       f"Pred: {record['predicted_label']} ({float(record['confidence']):.3f})")
            draw.text((x + 4, y + thumb_h), caption, fill="black", font=font)
        sheet.save(output_dir / f"{prefix}_errors_{page_start // per_page + 1:02d}.jpg",
                   quality=int(sheet_cfg["jpeg_quality"]))


def evaluate_checkpoint(config: dict, checkpoint_path: Path, output_root: Path) -> Path:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("class_names") != list(config["data"]["class_names"]):
        raise ValueError("Checkpoint class order does not match this config.")
    model = build_model(config, load_pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    device = torch.device(str(config["runtime"]["evaluation_device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.evaluation_device requests CUDA, but CUDA is unavailable.")
    model.to(device)

    _, validation_samples, split_counts = split_samples(config)
    _, eval_transform = make_transforms(config)
    loader = DataLoader(
        FaceClassDataset(validation_samples, eval_transform),
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    subclass_names = list(config["data"]["class_names"])
    binary_names = list(config["task"]["binary_class_names"])
    subclass_mapping = subclass_to_binary_ids(config)
    binary_truth: list[int] = []
    binary_pred: list[int] = []
    subclass_truth: list[int] = []
    subclass_pred: list[int] = []
    binary_confidence: list[float] = []
    subclass_confidence: list[float] = []
    with torch.inference_mode():
        for images, labels in loader:
            binary_logits, subclass_logits = model(images.to(device))
            probabilities = torch.sigmoid(binary_logits.squeeze(1))
            label_map = torch.tensor(subclass_mapping, dtype=torch.long)
            binary_truth.extend(label_map[labels].tolist())
            threshold = float(config["inference"]["occluded_threshold"])
            binary_pred.extend((probabilities >= threshold).long().tolist())
            binary_confidence.extend(torch.where(probabilities >= threshold, probabilities, 1 - probabilities).tolist())
            subclass_probs = torch.softmax(subclass_logits, dim=1)
            subclass_pred.extend(subclass_probs.argmax(dim=1).tolist())
            subclass_truth.extend(labels.tolist())
            subclass_confidence.extend(subclass_probs.max(dim=1).values.tolist())

    binary_report = classification_report(binary_truth, binary_pred, binary_names)
    subclass_report = classification_report(subclass_truth, subclass_pred, subclass_names)
    # This directory always represents the current best checkpoint. Remove only
    # evaluator-owned entries so stale plots/error pages cannot survive a rerun.
    run_dir = output_root
    run_dir.mkdir(parents=True, exist_ok=True)
    for generated_dir in (run_dir / "binary_2class", run_dir / "subtype_4class"):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    for generated_file in (
        run_dir / "validation_accuracy.png",
        run_dir / "evaluation_summary.json",
        run_dir / "latest_evaluation.json",
    ):
        generated_file.unlink(missing_ok=True)
    binary_dir, subclass_dir = run_dir / "binary_2class", run_dir / "subtype_4class"
    binary_dir.mkdir(parents=True, exist_ok=True)
    subclass_dir.mkdir(parents=True, exist_ok=True)
    (binary_dir / "metrics.json").write_text(json.dumps(binary_report, indent=2), encoding="utf-8")
    (subclass_dir / "metrics.json").write_text(json.dumps(subclass_report, indent=2), encoding="utf-8")
    plot_cfg = config["evaluation"]["plots"]
    save_validation_accuracy_chart(binary_report, subclass_report,
                                   run_dir / "validation_accuracy.png",
                                   int(plot_cfg["history_dpi"]))
    save_confusion_matrix(binary_report, binary_dir / "confusion_matrix.png",
                          "Validation confusion matrix — 2 classes",
                          int(plot_cfg["confusion_matrix_dpi"]))
    save_confusion_matrix(subclass_report, subclass_dir / "confusion_matrix.png",
                          "Validation confusion matrix — 4 subclasses",
                          int(plot_cfg["confusion_matrix_dpi"]))
    history_path = checkpoint_path.parent / "history.csv"
    save_history_plot(history_path, binary_dir / "training_curves.png", "Main binary task",
                      "train_loss", "val_loss", "train_macro_f1", "val_macro_f1",
                      "train_accuracy", "val_accuracy",
                      float(plot_cfg["history_width"]), float(plot_cfg["history_height"]),
                      int(plot_cfg["history_dpi"]))
    save_history_plot(history_path, subclass_dir / "training_curves.png", "Auxiliary 4-class task",
                      "train_auxiliary_loss", "val_auxiliary_loss",
                      "train_auxiliary_macro_f1", "val_auxiliary_macro_f1",
                      "train_auxiliary_accuracy", "val_auxiliary_accuracy",
                      float(plot_cfg["history_width"]), float(plot_cfg["history_height"]),
                      int(plot_cfg["history_dpi"]))

    binary_errors, subclass_errors = [], []
    for i, (path, subclass_id) in enumerate(validation_samples):
        binary_id = subclass_mapping[subclass_id]
        binary_conf = binary_confidence[i]
        subclass_conf = subclass_confidence[i]
        common = {
            "image_path": str(path),
            "binary_prediction": binary_names[binary_pred[i]],
            "subclass_prediction": subclass_names[subclass_pred[i]],
        }
        if binary_pred[i] != binary_id:
            binary_errors.append({**common, "true_label": binary_names[binary_id],
                                  "predicted_label": binary_names[binary_pred[i]],
                                  "confidence": f"{binary_conf:.6f}"})
        if subclass_pred[i] != subclass_id:
            subclass_errors.append({**common, "true_label": subclass_names[subclass_id],
                                    "predicted_label": subclass_names[subclass_pred[i]],
                                    "confidence": f"{subclass_conf:.6f}"})
    sheet_cfg = config["evaluation"]["contact_sheet"]
    save_error_outputs(binary_errors, binary_dir / "errors", "binary", sheet_cfg)
    save_error_outputs(subclass_errors, subclass_dir / "errors", "subtype_4class", sheet_cfg)
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "validation_split_counts": split_counts,
        "binary_2class": binary_report,
        "subtype_4class": subclass_report,
        "binary_subclass_prediction_disagreements": sum(
            binary_pred[i] != subclass_mapping[subclass_pred[i]]
            for i in range(len(binary_pred))
        ),
    }
    (run_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_pointer = {
        "run_dir": str(run_dir.resolve()),
        "summary": str((run_dir / "evaluation_summary.json").resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    (run_dir / "latest_evaluation.json").write_text(
        json.dumps(latest_pointer, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Best-checkpoint evaluation output: {run_dir.resolve()}")
    print(f"Binary: accuracy={binary_report['accuracy']:.4f}, macro-F1={binary_report['macro_f1']:.4f}, errors={len(binary_errors)}")
    print(f"4-class: accuracy={subclass_report['accuracy']:.4f}, macro-F1={subclass_report['macro_f1']:.4f}, errors={len(subclass_errors)}")
    print(f"Prediction disagreements between binary and grouped 4-class heads: {summary['binary_subclass_prediction_disagreements']}")
    return run_dir


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description=(
            "Train binary clear/occluded classification with four-way auxiliary supervision. "
            "The four folders are grouped into two main classes; ONNX export is separate."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("train", "export", "evaluate"),
        default="train",
        help="train, export a binary ONNX model, or evaluate both heads and save plots/errors.",
    )
    parser.add_argument("--check-data", action="store_true", help="Print the deterministic 80/20 split and exit.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to export in --mode export; defaults to the best checkpoint.")
    parser.add_argument(
        "--device",
        default=None,
        help="ONNX export device override, e.g. cuda or cpu; default comes from runtime.export_device.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Evaluation output root for --mode evaluate.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    if args.check_data:
        _, _, counts = split_samples(config)
        total_train = total_val = 0
        for name in config["data"]["class_names"]:
            row = counts[name]
            total_train += row["train"]
            total_val += row["val"]
            print(
                f"{name}: unique={row['total_unique']} "
                f"train={row['train']} val={row['val']}"
            )
        print(f"TOTAL: train={total_train}, val={total_val}, all={total_train + total_val}")
        print("Source image files were not modified.")
        return

    if args.mode == "train":
        checkpoint_path = train(config)
        if bool(config["paths"].get("auto_export_after_training", True)):
            onnx_path = export_onnx(config, checkpoint_path)
            print(f"Updated inference model: {onnx_path}")
        else:
            print("Training complete. Automatic ONNX export is disabled.")
        if bool(config["evaluation"].get("auto_evaluate_after_training", True)):
            output_root = Path(config["evaluation"]["output_dir"])
            if not output_root.is_absolute():
                output_root = PROJECT_ROOT / output_root
            evaluate_checkpoint(config, checkpoint_path, output_root)
        print(f"To export this checkpoint later, run with --mode export --checkpoint \"{checkpoint_path}\".")
        return

    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint_path is None:
        checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = PROJECT_ROOT / checkpoint_dir
        checkpoint_path = checkpoint_dir / config["paths"]["best_checkpoint_name"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.mode == "export":
        export_onnx(config, checkpoint_path, device_override=args.device)
    else:
        output_root = args.output_dir or Path(config["evaluation"]["output_dir"])
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        evaluate_checkpoint(config, checkpoint_path, output_root)


if __name__ == "__main__":
    main()
