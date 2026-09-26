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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score, precision_score, recall_score

from src.datasets.dataset import build_loaders, build_transforms
from src.training.losses import apply_target_smoothing, build_loss_criterion
from src.utils.config import load_config
from src.utils.model import build_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ExplicitClassImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, class_names: list[str], transform=None):
        self.samples = []
        self.classes = list(class_names)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.transform = transform

        for cls_name in self.classes:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            idx = self.class_to_idx[cls_name]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((str(img_path), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target


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


def evaluate_dataset_dir(
    cfg: dict,
    checkpoint_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
    use_tta: bool = False,
) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating dataset from: {dataset_dir.resolve()}")
    print(f"Loading checkpoint: {checkpoint_path.resolve()}")
    print(f"Test-Time Augmentation (TTA): {'ENABLED (2-pass Horizontal Flip)' if use_tta else 'DISABLED'}")

    class_names = list(cfg["data"]["class_names"])
    _, eval_transform = build_transforms(cfg)

    # Use ExplicitClassImageDataset for labeled dataset
    test_ds = ExplicitClassImageDataset(dataset_dir, class_names=class_names, transform=eval_transform)
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
    all_logits, all_probs = [], []
    misclassified = []
    sample_offset = 0
    num_classes = int(cfg["model"].get("num_classes", 1))
    criterion = build_loss_criterion(cfg, device)
    criterion.eval()
    target_smoothing_cfg = cfg.get("training", {}).get("target_smoothing", {})
    target_smoothing_enabled = bool(target_smoothing_cfg.get("enabled", False))
    target_low = float(target_smoothing_cfg.get("target_low", 0.02))
    target_high = float(target_smoothing_cfg.get("target_high", 0.98))
    total_loss = 0.0

    start_time = time.time()
    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            
            if use_tta:
                images_flipped = torch.flip(images, dims=[3])
                logits_orig = model(images)
                logits_flip = model(images_flipped)
                logits = (logits_orig + logits_flip) * 0.5
            else:
                logits = model(images)

            if num_classes == 1:
                logits_1d = logits.view(-1)
                loss_targets = targets_device.float()
                if target_smoothing_enabled:
                    loss_targets = apply_target_smoothing(loss_targets, target_low, target_high)
                loss = criterion(logits_1d, loss_targets)
                probs_occ = torch.sigmoid(logits_1d).cpu()
                preds = (probs_occ >= 0.5).long()
                confidences = torch.where(preds == 1, probs_occ, 1.0 - probs_occ)
                all_logits.extend(logits_1d.cpu().tolist())
                all_probs.extend(probs_occ.tolist())
            else:
                loss = criterion(logits, targets_device)
                probs = torch.softmax(logits, dim=1).cpu()
                confidences, preds = probs.max(dim=1)

            total_loss += float(loss.item()) * len(targets)

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
    avg_loss = total_loss / len(y_true) if y_true else 0.0
    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 65)
    print(f"                    EVALUATION REPORT {'(TTA: 2-PASS FLIP)' if use_tta else ''}")
    print("=" * 65)
    print(f"TTA Enabled:         {use_tta}")
    print(f"Total Samples:       {len(y_true)}")
    print(f"Cross-Entropy Loss:  {avg_loss:.4f}")
    print(f"Overall Accuracy:    {acc * 100:.2f}%")
    print(f"Macro Precision:     {macro_precision * 100:.2f}%")
    print(f"Macro Recall:        {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score:      {macro_f1 * 100:.2f}%")
    print(f"Misclassified Total: {len(misclassified)} / {len(y_true)} ({len(misclassified)/len(y_true)*100:.2f}%)")
    if all_logits:
        import numpy as np
        print(f"Logits Range:        [{np.min(all_logits):.3f}, {np.max(all_logits):.3f}] (Mean: {np.mean(all_logits):.3f}, Std: {np.std(all_logits):.3f})")
        print(f"Sigmoid Prob Range:  [{np.min(all_probs):.4f}, {np.max(all_probs):.4f}] (Mean: {np.mean(all_probs):.4f})")
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
    parser.add_argument(
        "pos_dataset_dir",
        nargs="?",
        default=None,
        help="Optional positional path to labeled dataset folder (containing clear/ and occluded/)",
    )
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
        choices=["val", "test", "auto"],
        default="auto",
        help="Labeled split to evaluate from config.yaml: val, test, or auto (default: auto)",
    )
    parser.add_argument(
        "--unlabeled-dir",
        default=None,
        help="Run inference on a flat directory of images without ground-truth labels",
    )
    parser.add_argument("--unlabeled-output-dir", default="outputs/unlabeled_test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--tta",
        dest="tta",
        action="store_true",
        default=None,
        help="Enable 2-pass horizontal flip Test-Time Augmentation (TTA)",
    )
    parser.add_argument(
        "--no-tta",
        dest="tta",
        action="store_false",
        help="Disable Test-Time Augmentation (TTA)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    use_tta = args.tta if args.tta is not None else bool(cfg.get("evaluation", {}).get("use_tta", False))
    checkpoint_path = Path(args.checkpoint or cfg["checkpoint"]["best_path"])

    target_dataset_dir = args.dataset_dir or args.pos_dataset_dir
    if target_dataset_dir:
        target_path = Path(target_dataset_dir)
        out_dir = Path(args.output_dir or (target_path / "eval_results"))
        evaluate_dataset_dir(cfg, checkpoint_path, target_path, out_dir, batch_size=args.batch_size, use_tta=use_tta)
        return

    # Default flow using config data.root
    data_root = Path(cfg["data"]["root"])
    if args.split == "auto":
        split_name = "test" if (data_root / "test").exists() else "val"
    else:
        split_name = args.split

    split_dir = data_root / split_name
    if not split_dir.exists():
        if (data_root / "val").exists():
            print(f"Directory {split_dir} not found. Falling back to validation set: {data_root / 'val'}")
            split_dir = data_root / "val"
            split_name = "val"
        else:
            raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

    out_dir = Path(args.output_dir or f"outputs/eval_{split_name}")
    evaluate_dataset_dir(cfg, checkpoint_path, split_dir, out_dir, batch_size=args.batch_size, use_tta=use_tta)


if __name__ == "__main__":
    main()
