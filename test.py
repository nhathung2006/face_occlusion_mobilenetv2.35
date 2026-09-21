from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score

from src.datasets.dataset import build_loaders, build_transforms
from src.utils.config import load_config
from src.utils.model import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_checkpoint(model, checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def save_contact_sheets_paged(items: list[dict], output_dir: Path, prefix: str = "misclassified", sheet_size: int = 25, max_sheets: int = 5) -> None:
    sheets_to_generate = min(max_sheets, math.ceil(len(items) / sheet_size))
    for sheet_idx in range(sheets_to_generate):
        start = sheet_idx * sheet_size
        chunk = items[start:start + sheet_size]
        columns = 5
        rows = math.ceil(len(chunk) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(15, 3.6 * rows))
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for ax, item in zip(axes, chunk):
            try:
                img = Image.open(item["path"]).convert("RGB")
                ax.imshow(img)
                conf = item.get("confidence", 0.0) * 100
                ax.set_title(
                    f"{Path(item['path']).name}\nTrue: {item['true_class']} | Pred: {item['pred_class']} ({conf:.1f}%)",
                    fontsize=8,
                    color="#d63031" if item["true_class"] != item["pred_class"] else "#00b894",
                )
            except Exception:
                pass
            ax.axis("off")

        for ax in axes[len(chunk):]:
            ax.axis("off")

        fig.tight_layout()
        sheet_name = f"{prefix}_sheet_{sheet_idx + 1:03d}.png"
        fig.savefig(output_dir / sheet_name, dpi=120)
        plt.close(fig)


def evaluate_dataset_dir(cfg: dict, checkpoint_path: Path, dataset_dir: Path, output_dir: Path, batch_size: int = 64) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating dataset from: {dataset_dir.resolve()}")
    print(f"Loading checkpoint: {checkpoint_path.resolve()}")

    class_names = list(cfg["data"]["class_names"])
    _, eval_transform = build_transforms(cfg)

    # Use ImageFolder for labeled dataset
    test_ds = datasets.ImageFolder(dataset_dir, transform=eval_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    y_true, y_pred = [], []
    misclassified = []
    sample_offset = 0

    start_time = time.time()
    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            confidences, preds = probs.max(dim=1)

            preds_list = preds.tolist()
            targets_list = targets.tolist()

            for i, (target, pred, conf) in enumerate(zip(targets_list, preds_list, confidences)):
                global_idx = sample_offset + i
                image_path, _ = test_ds.samples[global_idx]
                true_cls = class_names[target]
                pred_cls = class_names[pred]

                if target != pred:
                    misclassified.append({
                        "index": global_idx,
                        "path": str(image_path),
                        "filename": Path(image_path).name,
                        "true_id": target,
                        "true_class": true_cls,
                        "pred_id": pred,
                        "pred_class": pred_cls,
                        "confidence": float(conf),
                    })

            y_pred.extend(preds_list)
            y_true.extend(targets_list)
            sample_offset += len(targets_list)

            if (batch_idx + 1) % 20 == 0 or (sample_offset == len(test_ds)):
                print(f"Processed [{sample_offset}/{len(test_ds)}] images...", end="\r", flush=True)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s ({len(test_ds)/elapsed:.1f} images/sec)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 65)
    print("                    EVALUATION REPORT")
    print("=" * 65)
    print(f"Total Samples:       {len(y_true)}")
    print(f"Overall Accuracy:    {acc * 100:.2f}%")
    print(f"Macro F1-Score:      {macro_f1 * 100:.2f}%")
    print(f"Misclassified Total: {len(misclassified)} / {len(y_true)} ({len(misclassified)/len(y_true)*100:.2f}%)")
    print("-" * 65)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    print("=" * 65)

    # Save Confusion Matrix
    cm_path = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix (Acc: {acc*100:.2f}%)", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion Matrix saved to: {cm_path.resolve()}")

    # Sort misclassified by confidence descending (most obvious errors first)
    misclassified.sort(key=lambda x: -x["confidence"])

    # Save misclassified CSV
    csv_path = output_dir / "misclassified.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "filename", "true_class", "pred_class", "confidence", "path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in misclassified:
            writer.writerow({
                "index": item["index"],
                "filename": item["filename"],
                "true_class": item["true_class"],
                "pred_class": item["pred_class"],
                "confidence": f"{item['confidence']:.4f}",
                "path": item["path"],
            })
    print(f"Misclassified CSV saved to: {csv_path.resolve()}")

    # Save contact sheets for misclassified
    if misclassified:
        print(f"Generating contact sheets for {len(misclassified)} misclassified images...")
        save_contact_sheets_paged(misclassified, output_dir, prefix="misclassified", sheet_size=25, max_sheets=8)
        print(f"Contact sheets saved to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV2 Face Occlusion Model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to an arbitrary labeled dataset directory containing class subfolders (e.g. clear/ and occluded/)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory to save evaluation metrics, CSVs, and plots",
    )
    parser.add_argument("--misclassified-dir", default="outputs/misclassified")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Labeled split to evaluate from config.yaml",
    )
    parser.add_argument(
        "--unlabeled-dir",
        default=None,
        help="Run inference on a flat directory of images without ground-truth labels",
    )
    parser.add_argument("--unlabeled-output-dir", default="outputs/unlabeled_test")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint or cfg["checkpoint"]["best_path"])

    if args.dataset_dir:
        out_dir = Path(args.output_dir or "outputs/eval_custom")
        evaluate_dataset_dir(cfg, checkpoint_path, Path(args.dataset_dir), out_dir, batch_size=args.batch_size)
        return

    if args.unlabeled_dir:
        from test import run_unlabeled_inference
        run_unlabeled_inference(
            cfg,
            checkpoint_path,
            Path(args.unlabeled_dir),
            Path(args.unlabeled_output_dir),
        )
        return

    # Default flow using config data.root
    data_root = Path(cfg["data"]["root"])
    split_dir = data_root / args.split
    out_dir = Path(args.output_dir or f"outputs/eval_{args.split}")
    evaluate_dataset_dir(cfg, checkpoint_path, split_dir, out_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
