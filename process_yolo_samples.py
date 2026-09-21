from __future__ import annotations

import csv
from pathlib import Path
import shutil
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

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from src.datasets.dataset import build_transforms
from src.utils.config import load_config
from src.utils.model import build_model

# Visual audit mapping for the 45 misclassified images in yolo_non_occluded_1000
# These 8 files were determined to have real occlusions (hand, sunglasses, rescue/hockey helmet, object in mouth, baby head)
MANUAL_OCCLUDED_FILES = {
    "61--Street_Battle_61_Street_Battle_streetfight_61_868.jpg_b147062.jpg",
    "27--Spa_27_Spa_Spa_27_400.jpg_b92950.jpg",
    "0--Parade_0_Parade_Parade_0_431.jpg_b10342.jpg",
    "11--Meeting_11_Meeting_Meeting_11_Meeting_Meeting_11_471.jpg_b22848.jpg",
    "23--Shoppers_23_Shoppers_Shoppers_23_115.jpg_b89690.jpg",
    "54--Rescue_54_Rescue_rescuepeople_54_817.jpg_b35088.jpg",
    "58--Hockey_58_Hockey_icehockey_puck_58_330.jpg_b36421.jpg",
    "53--Raid_53_Raid_policeraid_53_318.jpg_b136504.jpg",
}


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def main():
    config_path = Path("config/config.yaml")
    cfg = load_config(config_path)

    checkpoint_path = Path(cfg["checkpoint"]["best_path"])
    yolo_dataset_dir = Path(r"C:\Thực tập LUMI\model_BaiToan\Deepleaning\data\yolo_non_occluded_1000")
    train_dest_root = Path(r"C:\Thực tập LUMI\model_BaiToan\Deepleaning\data\dataset\train")
    output_log_dir = Path("outputs/eval_yolo_test")
    output_log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Source: {yolo_dataset_dir.resolve()}")
    print(f"Destination: {train_dest_root.resolve()}")

    class_names = list(cfg["data"]["class_names"])  # ['clear', 'occluded']
    _, eval_transform = build_transforms(cfg)

    dataset = datasets.ImageFolder(yolo_dataset_dir, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    all_results = []
    sample_offset = 0

    print(f"\n--- 1. Scanning {len(dataset)} images in {yolo_dataset_dir} ---")
    start_time = time.time()
    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            confidences, preds = probs.max(dim=1)

            preds_list = preds.tolist()
            targets_list = targets.tolist()

            for i, (target, pred, conf) in enumerate(zip(targets_list, preds_list, confidences)):
                global_idx = sample_offset + i
                img_path, _ = dataset.samples[global_idx]
                img_path = Path(img_path)
                true_folder_cls = class_names[target]
                pred_cls = class_names[pred]
                conf_val = float(conf)
                is_correct = (target == pred)

                all_results.append({
                    "path": img_path,
                    "filename": img_path.name,
                    "folder_label": true_folder_cls,
                    "pred_class": pred_cls,
                    "confidence": conf_val,
                    "is_correct": is_correct,
                })

            sample_offset += len(targets_list)

    elapsed = time.time() - start_time
    print(f"Scan completed in {elapsed:.2f}s ({len(dataset)/elapsed:.1f} images/sec)")

    # 2. Filter images
    misclassified_items = [r for r in all_results if not r["is_correct"]]
    correct_low_conf_items = [r for r in all_results if r["is_correct"] and r["confidence"] < 0.80]

    print(f"\n--- 2. Filter Summary ---")
    print(f"Total scanned:                        {len(all_results)}")
    print(f"Total misclassified:                  {len(misclassified_items)}")
    print(f"Total correct with confidence < 80%:  {len(correct_low_conf_items)}")

    # Assign target labels
    to_copy_list = []

    # Process Misclassified items
    for item in misclassified_items:
        # Determine actual assigned label
        if item["filename"] in MANUAL_OCCLUDED_FILES:
            assigned_label = "occluded"
            reason = "misclassified_manual_occluded"
        else:
            assigned_label = "clear"
            reason = "misclassified_manual_clear"

        to_copy_list.append({
            "source_path": item["path"],
            "filename": item["filename"],
            "folder_label": item["folder_label"],
            "pred_class": item["pred_class"],
            "confidence": item["confidence"],
            "assigned_label": assigned_label,
            "reason": reason,
        })

    # Process Correct Low-Confidence items (< 80%)
    for item in correct_low_conf_items:
        assigned_label = item["folder_label"]
        reason = f"correct_low_conf_{item['confidence']:.3f}"
        to_copy_list.append({
            "source_path": item["path"],
            "filename": item["filename"],
            "folder_label": item["folder_label"],
            "pred_class": item["pred_class"],
            "confidence": item["confidence"],
            "assigned_label": assigned_label,
            "reason": reason,
        })

    # 3. Copy files to train dataset
    print(f"\n--- 3. Copying {len(to_copy_list)} images to train dataset ---")
    clear_train_dir = train_dest_root / "clear"
    occluded_train_dir = train_dest_root / "occluded"
    clear_train_dir.mkdir(parents=True, exist_ok=True)
    occluded_train_dir.mkdir(parents=True, exist_ok=True)

    copied_clear = 0
    copied_occluded = 0
    copied_records = []

    for item in to_copy_list:
        assigned_label = item["assigned_label"]
        dest_folder = clear_train_dir if assigned_label == "clear" else occluded_train_dir
        dest_path = dest_folder / item["filename"]

        # Ensure file copy
        shutil.copy2(item["source_path"], dest_path)

        if assigned_label == "clear":
            copied_clear += 1
        else:
            copied_occluded += 1

        copied_records.append({
            "filename": item["filename"],
            "assigned_label": assigned_label,
            "reason": item["reason"],
            "pred_class": item["pred_class"],
            "confidence": f"{item['confidence']:.4f}",
            "source_path": str(item["source_path"]),
            "dest_path": str(dest_path),
        })

    # 4. Save audit log
    audit_csv = output_log_dir / "added_to_train_summary.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "assigned_label", "reason", "pred_class", "confidence", "source_path", "dest_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(copied_records)

    print(f"\nSuccessfully copied {len(copied_records)} images to train dataset:")
    print(f"  • Added to 'train/clear':    {copied_clear}")
    print(f"  • Added to 'train/occluded': {copied_occluded}")
    print(f"Audit log saved to: {audit_csv.resolve()}")


if __name__ == "__main__":
    main()
