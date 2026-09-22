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
from PIL import Image

from src.datasets.dataset import build_transforms
from src.utils.config import load_config
from src.utils.model import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def _render_sheet(items: list[dict], save_path: Path) -> None:
    columns = 5
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.6 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, item in zip(axes, items):
        try:
            image = Image.open(item["source_path"]).convert("RGB")
            ax.imshow(image)
            tag = f" [{item['category'].upper()}]" if item["category"] == "low_confidence" else ""
            ax.set_title(
                f"{item['filename']}\n{item['pred_class']} ({item['confidence']:.3f}){tag}",
                fontsize=8,
                color="red" if item["category"] == "low_confidence" else "black",
            )
        except Exception:
            pass
        ax.axis("off")

    for ax in axes[len(items) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_contact_sheets(results: list[dict], output_dir: Path, sheet_size: int = 25) -> None:
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate sheets for ALL low_confidence images
    low_conf_items = [r for r in results if r["category"] == "low_confidence"]
    if low_conf_items:
        print(f"Generating contact sheets for {len(low_conf_items)} low-confidence images...")
        for sheet_index, start in enumerate(range(0, len(low_conf_items), sheet_size), start=1):
            sheet = low_conf_items[start : start + sheet_size]
            _render_sheet(sheet, sheets_dir / f"low_conf_sheet_{sheet_index:03d}.png")

    # 2. Generate sample sheets for clear & occluded (up to 2 sheets each)
    for cat in ["clear", "occluded"]:
        cat_items = [r for r in results if r["category"] == cat][: sheet_size * 2]
        if cat_items:
            for sheet_index, start in enumerate(range(0, len(cat_items), sheet_size), start=1):
                sheet = cat_items[start : start + sheet_size]
                _render_sheet(sheet, sheets_dir / f"sample_{cat}_sheet_{sheet_index:03d}.png")


def classify_and_organize(
    input_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    threshold: float,
    action: str = "copy",
    make_contact_sheets: bool = True,
    batch_size: int | None = None,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_names = list(cfg["data"]["class_names"])
    bs = batch_size or int(cfg["data"].get("batch_size", 64))
    _, eval_transform = build_transforms(cfg)

    # Initialize model
    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    # Find all images
    image_paths = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    total_images = len(image_paths)
    if total_images == 0:
        print(f"No supported images found in: {input_dir}")
        return

    # Determine effective confidence threshold for binary uncertainty
    # If user specifies 0.4 for binary classification, uncertainty zone is [0.40, 0.60] (max confidence < 0.60)
    effective_threshold = (1.0 - threshold) if 0.0 < threshold < 0.5 else threshold

    print(f"Found {total_images} images to classify in {input_dir}")
    print(f"Confidence threshold for low_confidence: < {effective_threshold:.2f} (input threshold: {threshold:.2f})")
    print(f"Destination root: {output_dir.resolve()}")

    # Prepare subdirectories
    dirs = {
        "clear": output_dir / "clear",
        "occluded": output_dir / "occluded",
        "low_confidence": output_dir / "low_confidence",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    results = []
    stats = {"clear": 0, "occluded": 0, "low_confidence": 0, "failed": 0}

    start_time = time.time()

    with torch.inference_mode():
        for start_idx in range(0, total_images, bs):
            batch_paths = image_paths[start_idx : start_idx + bs]
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
            logits = model(inputs)
            num_classes = int(cfg["model"].get("num_classes", 1))
            if num_classes == 1:
                logits_1d = logits.view(-1)
                probs_occ = torch.sigmoid(logits_1d).cpu()
                predictions = (probs_occ >= 0.5).long()
                confidences = torch.where(predictions == 1, probs_occ, 1.0 - probs_occ)
            else:
                probabilities = torch.softmax(logits, dim=1).cpu()
                confidences, predictions = probabilities.max(dim=1)

            for img_path, pred_idx, conf in zip(valid_paths, predictions, confidences):
                conf_val = float(conf)
                pred_label = class_names[int(pred_idx)]

                if conf_val < effective_threshold:
                    category = "low_confidence"
                else:
                    category = pred_label

                stats[category] += 1

                dest_file = dirs[category] / img_path.name
                if action == "move":
                    shutil.move(str(img_path), str(dest_file))
                else:
                    shutil.copy2(str(img_path), str(dest_file))

                results.append(
                    {
                        "filename": img_path.name,
                        "source_path": str(img_path),
                        "pred_class": pred_label,
                        "confidence": conf_val,
                        "category": category,
                        "destination_path": str(dest_file),
                    }
                )

            processed_so_far = min(start_idx + bs, total_images)
            print(f"Processed [{processed_so_far}/{total_images}] images...", end="\r", flush=True)

    elapsed_time = time.time() - start_time
    print()

    # Write predictions CSV
    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "source_path", "pred_class", "confidence", "category", "destination_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Generate contact sheets if requested
    if make_contact_sheets and results:
        print("Generating contact sheets for review...")
        save_contact_sheets(results, output_dir)

    print("=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total images scanned:   {total_images}")
    print(f"Total processed:        {len(results)}")
    print(f"  - Clear:              {stats['clear']} ({stats['clear']/total_images*100:.1f}%)")
    print(f"  - Occluded:           {stats['occluded']} ({stats['occluded']/total_images*100:.1f}%)")
    print(f"  - Low Confidence:     {stats['low_confidence']} ({stats['low_confidence']/total_images*100:.1f}%)")
    if stats["failed"] > 0:
        print(f"  - Failed/Corrupt:     {stats['failed']}")
    print(f"Time elapsed:           {elapsed_time:.2f}s ({len(results)/elapsed_time:.1f} img/s)")
    print(f"Predictions CSV:        {csv_path.resolve()}")
    if make_contact_sheets:
        print(f"Contact sheets folder:  {(output_dir / 'contact_sheets').resolve()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Classify and organize face images into folders")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"C:\Thực tập LUMI\model_BaiToan\Deepleaning\data\detect scrfd\crops_square",
        help="Path to folder containing face images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/classified_crops",
        help="Path to output directory for organized images",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold below which images go to low_confidence/",
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
    args = parser.parse_args()

    classify_and_organize(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        threshold=args.threshold,
        action=args.action,
        make_contact_sheets=not args.no_sheets,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
