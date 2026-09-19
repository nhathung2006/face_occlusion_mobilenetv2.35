from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.datasets.dataset import build_loaders, build_transforms
from src.utils.config import load_config
from src.utils.model import build_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_checkpoint(model, checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def save_unlabeled_contact_sheets(results, output_dir: Path, sheet_size: int = 25) -> None:
    for sheet_index, start in enumerate(range(0, len(results), sheet_size), start=1):
        sheet = results[start:start + sheet_size]
        columns = 5
        rows = math.ceil(len(sheet) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(15, 3.6 * rows))
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for ax, item in zip(axes, sheet):
            image = Image.open(item["source_path"]).convert("RGB")
            ax.imshow(image)
            low_confidence = " | LOW CONF" if item["below_threshold"] else ""
            ax.set_title(
                f"{item['filename']}\n{item['pred_class']} "
                f"({item['confidence']:.3f}){low_confidence}",
                fontsize=8,
            )
            ax.axis("off")

        for ax in axes[len(sheet):]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / f"contact_sheet_{sheet_index:03d}.png", dpi=150)
        plt.close(fig)


def run_unlabeled_inference(cfg: dict, checkpoint_path: Path, input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Unlabeled input directory not found: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = list(cfg["data"]["class_names"])
    image_size = int(cfg["data"]["image_size"])
    batch_size = int(cfg["data"].get("batch_size", 32))
    confidence_threshold = float(cfg.get("inference", {}).get("confidence_threshold", 0.5))
    _, eval_transform = build_transforms(cfg)

    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    results = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch_images = []
            valid_paths = []

            for image_path in batch_paths:
                try:
                    with Image.open(image_path) as image:
                        batch_images.append(eval_transform(image.convert("RGB")))
                    valid_paths.append(image_path)
                except (OSError, ValueError) as exc:
                    print(f"Skipped unreadable image: {image_path} ({exc})")

            if not batch_images:
                continue

            logits = model(torch.stack(batch_images).to(device))
            probabilities = torch.softmax(logits, dim=1).cpu()
            confidences, predictions = probabilities.max(dim=1)

            for image_path, prediction, confidence in zip(valid_paths, predictions, confidences):
                confidence_value = float(confidence)
                results.append(
                    {
                        "filename": image_path.name,
                        "source_path": str(image_path),
                        "pred_id": int(prediction),
                        "pred_class": class_names[int(prediction)],
                        "confidence": confidence_value,
                        "below_threshold": confidence_value < confidence_threshold,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename",
            "source_path",
            "pred_id",
            "pred_class",
            "confidence",
            "below_threshold",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    save_unlabeled_contact_sheets(results, output_dir)

    print(f"Unlabeled images processed: {len(results)}")
    print(f"Predictions CSV: {csv_path.resolve()}")
    print(f"Contact sheets: {output_dir.resolve()}")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    for item in results:
        print(
            f"  {item['filename']} -> {item['pred_class']} "
            f"({item['confidence']:.4f})"
            f"{' [LOW CONF]' if item['below_threshold'] else ''}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--misclassified-dir", default="outputs/misclassified")
    parser.add_argument(
        "--unlabeled-dir",
        default=None,
        help="Run inference on a flat directory of images without ground-truth labels",
    )
    parser.add_argument("--unlabeled-output-dir", default="outputs/unlabeled_test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint or cfg["checkpoint"]["best_path"])

    if args.unlabeled_dir:
        run_unlabeled_inference(
            cfg,
            checkpoint_path,
            Path(args.unlabeled_dir),
            Path(args.unlabeled_output_dir),
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = build_loaders(cfg)
    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    y_true, y_pred = [], []
    misclassified = []
    sample_offset = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            targets_list = targets.tolist()

            for batch_index, (target, pred) in enumerate(zip(targets_list, preds)):
                if target != pred:
                    image_path, folder_target = test_loader.dataset.samples[sample_offset + batch_index]
                    misclassified.append(
                        {
                            "index": sample_offset + batch_index,
                            "path": str(image_path),
                            "true_id": int(target),
                            "true_class": class_names[int(target)],
                            "pred_id": int(pred),
                            "pred_class": class_names[int(pred)],
                            "folder_target": int(folder_target),
                        }
                    )

            y_pred.extend(preds)
            y_true.extend(targets_list)
            sample_offset += len(targets_list)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    Path("outputs/confusion_matrix").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, ax=ax, cmap="Blues")
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix/test_confusion_matrix.png", dpi=150)
    plt.close(fig)

    misclassified_dir = Path(args.misclassified_dir)
    misclassified_dir.mkdir(parents=True, exist_ok=True)

    csv_path = misclassified_dir / "misclassified.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "source_path", "true_id", "true_class", "pred_id", "pred_class"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in misclassified:
            copied_name = (
                f"{item['index']:04d}_true-{item['true_class']}"
                f"_pred-{item['pred_class']}_{Path(item['path']).name}"
            )
            destination = misclassified_dir / copied_name
            shutil.copy2(item["path"], destination)
            writer.writerow(
                {
                    "index": item["index"],
                    "source_path": item["path"],
                    "true_id": item["true_id"],
                    "true_class": item["true_class"],
                    "pred_id": item["pred_id"],
                    "pred_class": item["pred_class"],
                }
            )
            item["copied_path"] = str(destination)

    if misclassified:
        columns = 4
        rows = math.ceil(len(misclassified) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for ax, item in zip(axes, misclassified):
            image = Image.open(item["path"]).convert("RGB")
            ax.imshow(image)
            ax.set_title(
                f"True: {item['true_class']}\nPred: {item['pred_class']}",
                fontsize=9,
            )
            ax.axis("off")
        for ax in axes[len(misclassified):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(misclassified_dir / "contact_sheet.png", dpi=150)
        plt.close(fig)

    print(f"Misclassified: {len(misclassified)} / {len(y_true)}")
    print(f"Misclassified CSV: {csv_path}")
    if misclassified:
        print(f"Misclassified images: {misclassified_dir.resolve()}")
        for item in misclassified:
            print(
                f"  [{item['index']:04d}] true={item['true_class']} | "
                f"pred={item['pred_class']} | {item['path']}"
            )
    else:
        print("No misclassified images found.")


if __name__ == "__main__":
    main()
