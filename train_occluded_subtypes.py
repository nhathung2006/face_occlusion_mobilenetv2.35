from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import yaml
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
except ModuleNotFoundError as exc:
    missing = exc.name or "unknown"
    raise SystemExit(
        f"Missing Python dependency: {missing}. Activate the project's Python environment "
        "and install its requirements before running this script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(r"C:\Thực tập LUMI\model_BaiToan\Deepleaning\data\class")
CLASS_NAMES = ("occluded_object", "occluded_pose")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.mobilenetv2 import MobileNetV2  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


class OccludedSubtypeDataset(Dataset):
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


def list_images(folder: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def stratified_split(
    data_root: Path, validation_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Training class folder not found: {class_dir}")
        files = list_images(class_dir)
        if len(files) < 2:
            raise ValueError(
                f"At least 2 images are required in {class_dir}; found {len(files)}."
            )

        rng.shuffle(files)
        val_count = max(1, round(len(files) * validation_fraction))
        val_count = min(val_count, len(files) - 1)
        val_samples.extend((path, label) for path in files[:val_count])
        train_samples.extend((path, label) for path in files[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def load_project_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return config


def make_model(config: dict, pretrained: bool) -> nn.Module:
    model_cfg = config["model"]
    width = float(model_cfg["width_mult"])
    dropout = float(model_cfg.get("dropout", 0.3))
    model = MobileNetV2(num_classes=1000, width_mult=width)

    if pretrained:
        weights_path = Path(model_cfg["pretrained_path"])
        if not weights_path.is_absolute():
            weights_path = PROJECT_ROOT / weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(f"ImageNet pretrained weights not found: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    return model


def set_trainable(model: nn.Module, unfreeze_from_feature: int | None) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if unfreeze_from_feature is None:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if unfreeze_from_feature < 0 or unfreeze_from_feature >= len(model.features):
        raise ValueError(
            f"unfreeze-from-feature must be in [0, {len(model.features) - 1}]"
        )
    for index, layer in enumerate(model.features):
        if index >= unfreeze_from_feature:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    for parameter in model.conv.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def freeze_batch_norm_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def compute_metrics(labels: list[int], predictions: list[int]) -> tuple[float, float]:
    accuracy = sum(y == p for y, p in zip(labels, predictions)) / max(1, len(labels))
    class_f1 = []
    for class_id in range(len(CLASS_NAMES)):
        tp = sum(y == class_id and p == class_id for y, p in zip(labels, predictions))
        fp = sum(y != class_id and p == class_id for y, p in zip(labels, predictions))
        fn = sum(y == class_id and p != class_id for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        class_f1.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return accuracy, sum(class_f1) / len(class_f1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        freeze_batch_norm_stats(model)

    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())

    accuracy, macro_f1 = compute_metrics(labels_all, predictions_all)
    return total_loss / max(1, len(labels_all)), accuracy, macro_f1


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict,
    image_size: int,
    epoch: int,
    val_macro_f1: float,
) -> None:
    model_cfg = config["model"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "width_mult": float(model_cfg["width_mult"]),
            "dropout": float(model_cfg.get("dropout", 0.3)),
            "image_size": image_size,
            "epoch": epoch,
            "val_macro_f1": val_macro_f1,
        },
        path,
    )


def train(args, config: dict, device: torch.device) -> Path:
    seed_everything(args.seed)
    train_samples, val_samples = stratified_split(
        args.data_root, args.validation_fraction, args.seed
    )
    print(
        "Samples (object, pose): "
        f"train=({sum(y == 0 for _, y in train_samples)}, {sum(y == 1 for _, y in train_samples)}), "
        f"val=({sum(y == 0 for _, y in val_samples)}, {sum(y == 1 for _, y in val_samples)})"
    )

    image_size = int(args.image_size or config["data"]["image_size"])
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    workers = int(config["data"].get("num_workers", 0))
    train_loader = DataLoader(
        OccludedSubtypeDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        OccludedSubtypeDataset(val_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(config, pretrained=True).to(device)
    counts = torch.tensor(
        [sum(label == i for _, label in train_samples) for i in range(len(CLASS_NAMES))],
        dtype=torch.float32,
        device=device,
    )
    class_weights = counts.sum() / (len(CLASS_NAMES) * counts.clamp_min(1.0))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_dir = args.checkpoint_dir
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    history_path = checkpoint_dir / "history.csv"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = float("-inf")
    best_epoch = 0
    waiting = 0
    optimizer = None
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.stage1_epochs > 0:
            set_trainable(model, None)
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.classifier_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=args.weight_decay,
            )
        elif epoch == args.stage1_epochs + 1:
            set_trainable(model, args.unfreeze_from_feature)
            backbone_params = [
                p for name, p in model.named_parameters()
                if p.requires_grad and not name.startswith("classifier")
            ]
            classifier_params = [
                p for name, p in model.named_parameters()
                if p.requires_grad and name.startswith("classifier")
            ]
            optimizer = torch.optim.SGD(
                [
                    {"params": backbone_params, "lr": args.backbone_lr},
                    {"params": classifier_params, "lr": args.classifier_lr / 2},
                ],
                momentum=0.9,
                nesterov=True,
                weight_decay=args.weight_decay,
            )
        if optimizer is None:
            raise RuntimeError("Optimizer was not initialized; check stage settings.")

        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} macroF1={train_f1:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} macroF1={val_f1:.3f}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        save_checkpoint(last_path, model, config, image_size, epoch, val_f1)
        if val_f1 > best_f1 + args.min_delta:
            best_f1, best_epoch, waiting = val_f1, epoch, 0
            save_checkpoint(best_path, model, config, image_size, epoch, val_f1)
        else:
            waiting += 1

        if epoch > args.stage1_epochs and waiting >= args.patience:
            print(f"Early stopping; best epoch={best_epoch}, val macroF1={best_f1:.4f}")
            break

    if not best_path.is_file():
        raise RuntimeError("Training ended without producing a best checkpoint.")
    print(f"Best checkpoint: {best_path}")
    return best_path


def classify(args, checkpoint_path: Path, device: torch.device) -> None:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("class_names") != list(CLASS_NAMES):
        raise ValueError(f"Checkpoint labels must be {list(CLASS_NAMES)}")
    config = load_project_config(args.config)
    config["model"]["width_mult"] = float(payload["width_mult"])
    config["model"]["dropout"] = float(payload["dropout"])
    model = make_model(config, pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()

    files = list_images(args.input_dir)
    if not files:
        raise ValueError(f"No supported images found under {args.input_dir}")
    size = int(payload["image_size"])
    transform = transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    for folder in (*CLASS_NAMES, "review"):
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "classification_manifest.csv"
    counts = {name: 0 for name in (*CLASS_NAMES, "review")}

    with torch.inference_mode(), manifest_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["source", "predicted_class", "confidence", "destination", "needs_review"],
        )
        writer.writeheader()
        for start in range(0, len(files), args.batch_size):
            batch_paths = files[start : start + args.batch_size]
            tensors = []
            for path in batch_paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            probabilities = torch.softmax(model(torch.stack(tensors).to(device)), dim=1)
            confidences, labels = probabilities.max(dim=1)

            for path, confidence_tensor, label_tensor in zip(
                batch_paths, confidences.cpu(), labels.cpu()
            ):
                label = int(label_tensor.item())
                confidence = float(confidence_tensor.item())
                predicted = CLASS_NAMES[label]
                destination_class = (
                    predicted if confidence >= args.review_threshold else "review"
                )
                relative_path = path.relative_to(args.input_dir)
                destination = args.output_dir / destination_class / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                counts[destination_class] += 1
                writer.writerow(
                    {
                        "source": str(relative_path),
                        "predicted_class": predicted,
                        "confidence": f"{confidence:.6f}",
                        "destination": str(destination.relative_to(args.output_dir)),
                        "needs_review": confidence < args.review_threshold,
                    }
                )

    print(f"Classified {len(files)} images from {args.input_dir}")
    print(f"Counts: {counts}")
    print(f"Manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 to distinguish occlusion by object vs pose, then classify images."
    )
    parser.add_argument("--mode", choices=("train", "classify", "all"), default="all")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--input-dir", type=Path, default=DATA_ROOT / "occluded")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_ROOT / "phan_loai" / "phan_loai_occluded",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "occluded_subtypes",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--unfreeze-from-feature", type=int, default=14)
    parser.add_argument("--classifier-lr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", type=float, default=0.00002)
    parser.add_argument("--weight-decay", type=float, default=0.0003)
    parser.add_argument("--review-threshold", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_classification_paths(args) -> None:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {args.output_dir}. "
            "Move/rename prior results first; this script will not overwrite them."
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    args.config = args.config.resolve()
    args.data_root = args.data_root.resolve()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else args.checkpoint_dir / "best.pth"
    config = load_project_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Class mapping: 0={CLASS_NAMES[0]}, 1={CLASS_NAMES[1]}")

    # Check before training so --mode all cannot spend hours training and then
    # fail because the input/output paths are invalid or the output is occupied.
    if args.mode in ("classify", "all"):
        validate_classification_paths(args)
    if args.mode in ("train", "all"):
        checkpoint_path = train(args, config, device)
    if args.mode in ("classify", "all"):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        classify(args, checkpoint_path, device)


if __name__ == "__main__":
    main()
