from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    from src.datasets.dataset import build_eval_transform
    from src.utils.config import load_config
    from src.utils.model import build_model
    from src.utils.seed import seed_everything
except ModuleNotFoundError as exc:
    missing = exc.name or "unknown"
    raise SystemExit(
        f"Missing Python dependency: {missing}.\n"
        "Use a project virtual environment and install the project requirements first:\n"
        "  py -3.14 -m venv .venv\n"
        "  .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
        "  .\\.venv\\Scripts\\python.exe train_clear_pose.py --mode all"
    ) from exc


CLASS_NAMES = ("clear_full_face", "clear_side_face")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FacePoseDataset(Dataset):
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


def image_files(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).casefold(),
    )


def make_stratified_split(
    data_root: Path, validation_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    rng = random.Random(seed)

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Training class folder not found: {class_dir}")
        files = image_files(class_dir)
        if len(files) < 2:
            raise ValueError(
                f"Need at least 2 images in {class_dir}; found {len(files)}. "
                "Add and review labeled examples before training."
            )
        rng.shuffle(files)
        val_count = max(1, round(len(files) * validation_fraction))
        val_count = min(val_count, len(files) - 1)
        val_samples.extend((p, label) for p in files[:val_count])
        train_samples.extend((p, label) for p in files[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_pose_model(
    config_path: Path,
    project_root: Path,
    pretrained: bool,
    width_mult: float | None = None,
    dropout: float | None = None,
) -> nn.Module:
    cfg = load_config(config_path)
    cfg["model"]["num_classes"] = len(CLASS_NAMES)
    cfg["model"]["pretrained"] = pretrained
    if width_mult is not None:
        cfg["model"]["width_mult"] = width_mult
    if dropout is not None:
        cfg["model"]["dropout"] = dropout
    pretrained_path = Path(cfg["model"]["pretrained_path"])
    if not pretrained_path.is_absolute():
        cfg["model"]["pretrained_path"] = str(project_root / pretrained_path)
    return build_model(cfg)


def freeze_for_stage(model: nn.Module, unfreeze_from_feature: int | None) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if unfreeze_from_feature is None:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    for index, layer in enumerate(model.features):
        if index >= unfreeze_from_feature:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    for parameter in model.conv.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def keep_batch_norm_frozen(model: nn.Module) -> None:
    # Preserve pretrained running statistics on this relatively small dataset.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def classification_metrics(labels: list[int], predictions: list[int]) -> tuple[float, float]:
    correct = sum(y == p for y, p in zip(labels, predictions))
    accuracy = correct / max(1, len(labels))
    f1_by_class = []
    for class_id in range(len(CLASS_NAMES)):
        tp = sum(y == class_id and p == class_id for y, p in zip(labels, predictions))
        fp = sum(y != class_id and p == class_id for y, p in zip(labels, predictions))
        fn = sum(y == class_id and p != class_id for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_by_class.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return accuracy, sum(f1_by_class) / len(f1_by_class)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        keep_batch_norm_frozen(model)

    loss_sum = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())

    accuracy, macro_f1 = classification_metrics(labels_all, predictions_all)
    return loss_sum / max(1, len(labels_all)), accuracy, macro_f1


def save_checkpoint(path: Path, model: nn.Module, args, epoch: int, val_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "width_mult": args.width_mult,
            "dropout": args.dropout,
            "image_size": args.image_size,
            "best_epoch": epoch,
            "val_macro_f1": val_f1,
        },
        path,
    )


def train(args, project_root: Path, data_root: Path) -> Path:
    seed_everything(args.seed)
    train_samples, val_samples = make_stratified_split(
        data_root, args.validation_fraction, args.seed
    )
    print(
        f"Split: train={len(train_samples)} ("
        f"{sum(y == 0 for _, y in train_samples)}/{sum(y == 1 for _, y in train_samples)}), "
        f"val={len(val_samples)} ("
        f"{sum(y == 0 for _, y in val_samples)}/{sum(y == 1 for _, y in val_samples)})"
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (args.image_size, args.image_size),
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
    eval_transform = build_eval_transform(args.image_size)
    train_loader = DataLoader(
        FacePoseDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FacePoseDataset(val_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_pose_model(
        args.config,
        project_root,
        pretrained=True,
        width_mult=args.width_mult,
        dropout=args.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_counts = torch.tensor(
        [sum(y == i for _, y in train_samples) for i in range(len(CLASS_NAMES))],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_counts.sum() / (len(CLASS_NAMES) * class_counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_dir = project_root / "checkpoints" / "clear_pose"
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"
    best_f1 = -1.0
    best_epoch = 0
    patience_count = 0

    print(f"Device: {device}; ImageNet pretrained MobileNetV2 width={args.width_mult}")
    print(f"BatchNorm running statistics are frozen in both stages.")
    optimizer = None
    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.stage1_epochs > 0:
            freeze_for_stage(model, None)
            stage = "head"
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.classifier_lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=args.weight_decay,
            )
        elif epoch == args.stage1_epochs + 1 or (epoch == 1 and args.stage1_epochs == 0):
            freeze_for_stage(model, args.unfreeze_from_feature)
            stage = "fine-tune"
            optimizer = torch.optim.SGD(
                [
                    {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier")], "lr": args.backbone_lr},
                    {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("classifier")], "lr": args.classifier_lr / 2},
                ],
                momentum=0.9,
                nesterov=True,
                weight_decay=args.weight_decay,
            )
        elif epoch <= args.stage1_epochs:
            stage = "head"
        else:
            stage = "fine-tune"

        if optimizer is None:
            raise RuntimeError("Optimizer was not initialized.")

        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device, None)
        print(
            f"Epoch {epoch:03d}/{args.epochs} [{stage}] "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} macroF1={train_f1:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} macroF1={val_f1:.3f}"
        )
        save_checkpoint(last_path, model, args, epoch, val_f1)
        if val_f1 > best_f1 + args.min_delta:
            best_f1 = val_f1
            best_epoch = epoch
            patience_count = 0
            save_checkpoint(best_path, model, args, epoch, val_f1)
        else:
            patience_count += 1
        if epoch > args.stage1_epochs and patience_count >= args.patience:
            print(f"Early stopping; best epoch={best_epoch}, val macroF1={best_f1:.3f}")
            break

    if not best_path.exists():
        raise RuntimeError("Training did not produce a best checkpoint.")
    print(f"Best checkpoint: {best_path}")
    return best_path


def classify_clear(args, project_root: Path, clear_dir: Path, output_dir: Path, checkpoint: Path) -> None:
    if not clear_dir.is_dir():
        raise FileNotFoundError(f"Classification input folder not found: {clear_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {output_dir}. Choose a new folder or review its contents; "
            "the script will not overwrite previous results."
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("class_names") != list(CLASS_NAMES):
        raise ValueError(f"Checkpoint class order is not {list(CLASS_NAMES)}")
    model = build_pose_model(
        args.config,
        project_root,
        pretrained=False,
        width_mult=float(payload["width_mult"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    files = image_files(clear_dir)
    if not files:
        raise ValueError(f"No supported images found under {clear_dir}")
    transform = build_eval_transform(int(payload["image_size"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (*CLASS_NAMES, "review"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "classification_manifest.csv"
    manifest_rows = []
    with torch.inference_mode(), manifest_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["source", "predicted_class", "confidence", "destination", "needs_review"],
        )
        writer.writeheader()
        for start in range(0, len(files), args.batch_size):
            batch_paths = files[start : start + args.batch_size]
            batch = []
            for path in batch_paths:
                with Image.open(path) as image:
                    batch.append(transform(image.convert("RGB")))
            inputs = torch.stack(batch).to(device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            for path, conf_t, pred_t in zip(batch_paths, confidence.cpu(), predictions.cpu()):
                pred_idx = int(pred_t.item())
                conf = float(conf_t.item())
                needs_review = conf < args.review_threshold
                target_class = "review" if needs_review else CLASS_NAMES[pred_idx]
                relative = path.relative_to(clear_dir)
                destination = output_dir / target_class / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                row = {
                    "source": str(relative),
                    "predicted_class": CLASS_NAMES[pred_idx],
                    "confidence": f"{conf:.6f}",
                    "destination": str(destination.relative_to(output_dir)),
                    "needs_review": needs_review,
                }
                writer.writerow(row)
                manifest_rows.append(row)

    totals = {name: sum(row["destination"].startswith(name + "\\") for row in manifest_rows) for name in (*CLASS_NAMES, "review")}
    print(f"Classified {len(files)} images from {clear_dir}")
    print(f"Output: {output_dir}")
    print(f"Counts: {totals}")
    print(f"Low-confidence threshold: {args.review_threshold:.2f}; see {manifest_path}")


def parse_args():
    project_root = Path(__file__).resolve().parent
    default_data_root = Path(r"C:\Thực tập LUMI\model_BaiToan\Deepleaning\data\class")
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV2 full-face vs side-face classifier and classify clear images."
    )
    parser.add_argument("--mode", choices=("train", "classify", "all"), default="all")
    parser.add_argument("--config", type=Path, default=project_root / "config" / "config.yaml")
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--clear-dir", type=Path, default=default_data_root / "clear")
    parser.add_argument("--output-dir", type=Path, default=default_data_root / "phan_loai_clear")
    parser.add_argument("--checkpoint", type=Path, default=project_root / "checkpoints" / "clear_pose" / "best.pth")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--width-mult", type=float, default=0.35)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze-from-feature", type=int, default=14)
    parser.add_argument("--classifier-lr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", type=float, default=0.00002)
    parser.add_argument("--weight-decay", type=float, default=0.0003)
    parser.add_argument("--review-threshold", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_root = args.data_root.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else project_root / args.checkpoint
    if args.mode in ("train", "all"):
        checkpoint = train(args, project_root, data_root)
    if args.mode in ("classify", "all"):
        classify_clear(args, project_root, args.clear_dir.resolve(), args.output_dir.resolve(), checkpoint.resolve())


if __name__ == "__main__":
    main()
