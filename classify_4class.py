from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 printing
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.models.mobilenetv2 import MobileNetV2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class BinaryWithSubtypeAuxiliary(nn.Module):
    def __init__(self, backbone: nn.Module, dropout: float, auxiliary_classes: int):
        super().__init__()
        self.features = backbone.features
        self.conv = backbone.conv
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
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
        return self.classifier(features), self.classifier_aux(features)


def build_4class_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", ["clear_full_face", "clear_side_face", "occluded_object", "occluded_pose"])
    width = float(ckpt.get("width_mult", 0.35))
    dropout = float(ckpt.get("dropout", 0.3))
    image_size = int(ckpt.get("image_size", 112))
    mean = tuple(ckpt.get("normalization_mean", (0.485, 0.456, 0.406)))
    std = tuple(ckpt.get("normalization_std", (0.229, 0.224, 0.225)))

    base = MobileNetV2(num_classes=1000, width_mult=width)
    base.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(base.classifier.in_features, 1),
    )
    model = BinaryWithSubtypeAuxiliary(base, dropout=dropout, auxiliary_classes=len(class_names))
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, class_names, eval_transform


def _render_sheet(items: list[dict], save_path: Path) -> None:
    columns = 5
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.6 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, item in zip(axes, items):
        try:
            image = Image.open(item["source_path"]).convert("RGB")
            ax.imshow(image)
            ax.set_title(
                f"{item['filename']}\n{item['pred_class']} ({item['confidence']:.3f})",
                fontsize=8,
            )
        except Exception:
            pass
        ax.axis("off")

    for ax in axes[len(items) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_contact_sheets(results: list[dict], output_dir: Path, class_names: list[str], sheet_size: int = 25) -> None:
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    for cat in class_names:
        cat_items = [r for r in results if r["pred_class"] == cat][: sheet_size * 2]
        if cat_items:
            for sheet_index, start in enumerate(range(0, len(cat_items), sheet_size), start=1):
                sheet = cat_items[start : start + sheet_size]
                _render_sheet(sheet, sheets_dir / f"sample_{cat}_sheet_{sheet_index:03d}.png")


def classify_4class_folder(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    action: str = "copy",
    make_contact_sheets: bool = True,
    batch_size: int = 64,
    use_tta: bool = False,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Test-Time Augmentation (TTA): {'ENABLED (2-pass Horizontal Flip)' if use_tta else 'DISABLED'}")

    model, class_names, eval_transform = build_4class_model(checkpoint_path, device)
    print(f"Classes ({len(class_names)}): {class_names}")

    image_paths = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    total_images = len(image_paths)
    if total_images == 0:
        print(f"No supported images found in: {input_dir}")
        return

    print(f"Found {total_images} images to classify in {input_dir}")
    print(f"Destination root: {output_dir.resolve()}")

    dirs = {name: output_dir / name for name in class_names}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    results = []
    stats = {name: 0 for name in class_names}
    stats["failed"] = 0

    start_time = time.time()

    with torch.inference_mode():
        for start_idx in range(0, total_images, batch_size):
            batch_paths = image_paths[start_idx : start_idx + batch_size]
            batch_tensors = []
            valid_paths = []

            for img_path in batch_paths:
                try:
                    with Image.open(img_path) as img:
                        batch_tensors.append(eval_transform(img.convert("RGB")))
                    valid_paths.append(img_path)
                except Exception as exc:
                    print(f"Error loading {img_path}: {exc}")
                    stats["failed"] += 1

            if not batch_tensors:
                continue

            inputs = torch.stack(batch_tensors).to(device)
            if use_tta:
                inputs_flipped = torch.flip(inputs, dims=[3])
                _, aux_orig = model(inputs)
                _, aux_flip = model(inputs_flipped)
                aux_logits = (aux_orig + aux_flip) * 0.5
            else:
                _, aux_logits = model(inputs)

            probabilities = torch.softmax(aux_logits, dim=-1).cpu()
            confidences, predictions = probabilities.max(dim=-1)

            for img_path, pred_idx, conf in zip(valid_paths, predictions, confidences):
                conf_val = float(conf)
                pred_label = class_names[int(pred_idx)]
                stats[pred_label] += 1

                dest_file = dirs[pred_label] / img_path.name
                if action == "move":
                    shutil.move(str(img_path), str(dest_file))
                else:
                    shutil.copy2(str(img_path), str(dest_file))

                results.append(
                    {
                        "filename": img_path.name,
                        "source_path": str(img_path),
                        "pred_class": pred_label,
                        "confidence": f"{conf_val:.4f}",
                        "destination_path": str(dest_file),
                    }
                )

            processed_so_far = min(start_idx + batch_size, total_images)
            print(f"Processed [{processed_so_far}/{total_images}] images...", end="\r", flush=True)

    elapsed_time = time.time() - start_time
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "source_path", "pred_class", "confidence", "destination_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if make_contact_sheets and results:
        print("Generating contact sheets for review...")
        save_contact_sheets(results, output_dir, class_names)

    print("=" * 60)
    print("4-CLASS CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total images scanned:   {total_images}")
    print(f"Total processed:        {len(results)}")
    for name in class_names:
        pct = (stats[name] / total_images * 100) if total_images > 0 else 0
        print(f"  - {name:<20}: {stats[name]:>5} ({pct:5.1f}%)")
    if stats["failed"] > 0:
        print(f"  - Failed/Corrupt:     {stats['failed']}")
    print(f"Time elapsed:           {elapsed_time:.2f}s ({len(results)/max(elapsed_time, 0.001):.1f} img/s)")
    print(f"Predictions CSV:        {csv_path.resolve()}")
    if make_contact_sheets:
        print(f"Contact sheets folder:  {(output_dir / 'contact_sheets').resolve()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Classify face images into 4 classes")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to input directory containing crop images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory for organized images",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/4class_config_yaml/best.pth",
        help="Path to 4-class model checkpoint (.pth)",
    )
    parser.add_argument(
        "--action",
        choices=["copy", "move"],
        default="copy",
        help="Whether to copy or move images (default: copy)",
    )
    parser.add_argument(
        "--no-sheets",
        action="store_true",
        help="Disable contact sheets generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable 2-pass horizontal flip Test-Time Augmentation (TTA)",
    )
    args = parser.parse_args()

    classify_4class_folder(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        checkpoint_path=Path(args.checkpoint),
        action=args.action,
        make_contact_sheets=not args.no_sheets,
        batch_size=args.batch_size,
        use_tta=args.tta,
    )


if __name__ == "__main__":
    main()
