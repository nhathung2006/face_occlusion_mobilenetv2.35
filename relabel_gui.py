from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, List, Optional

import torch
from PIL import Image

from src.datasets.dataset import build_transforms
from src.utils.config import load_config
from src.utils.model import build_model

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


class DatasetScanner:
    """Scans dataset split with the model and detects mislabeled/uncertain images."""

    def __init__(self, config_path: str | Path, checkpoint_path: str | Path, device: Optional[str] = None):
        self.cfg = load_config(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = list(self.cfg["data"]["class_names"])
        self.checkpoint_path = Path(checkpoint_path)
        self.model = build_model(self.cfg).to(self.device)
        load_checkpoint(self.model, self.checkpoint_path, self.device)
        self.model.eval()
        _, self.eval_transform = build_transforms(self.cfg)

    def scan(
        self,
        target_dir: Path,
        confidence_threshold: float = 0.60,
        only_misclassified: bool = True,
        batch_size: int = 64,
    ) -> List[dict[str, Any]]:
        items_to_scan = []
        for cls_name in self.class_names:
            cls_dir = target_dir / cls_name
            if not cls_dir.exists():
                continue
            for f in sorted(cls_dir.glob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    items_to_scan.append({
                        "path": f,
                        "name": f.name,
                        "true_class": cls_name,
                    })

        if not items_to_scan:
            return []

        results = []
        with torch.inference_mode():
            for i in range(0, len(items_to_scan), batch_size):
                batch = items_to_scan[i:i + batch_size]
                tensors = []
                valid_items = []
                for it in batch:
                    try:
                        with Image.open(it["path"]) as img:
                            w, h = img.size
                            tensor = self.eval_transform(img.convert("RGB"))
                            tensors.append(tensor)
                            it["width"] = w
                            it["height"] = h
                            valid_items.append(it)
                    except Exception as e:
                        print(f"Error opening image {it['path']}: {e}")

                if not tensors:
                    continue

                inputs = torch.stack(tensors).to(self.device)
                logits = self.model(inputs)
                probs = torch.softmax(logits, dim=1).cpu()
                confidences, preds = probs.max(dim=1)

                for it, pred_idx, conf in zip(valid_items, preds, confidences):
                    pred_class = self.class_names[int(pred_idx)]
                    conf_val = float(conf)
                    is_mis = (pred_class != it["true_class"])
                    is_unc = (conf_val < confidence_threshold)

                    it["pred_class"] = pred_class
                    it["confidence"] = conf_val
                    it["is_misclassified"] = is_mis
                    it["is_uncertain"] = is_unc

                    if only_misclassified:
                        if is_mis or is_unc:
                            results.append(it)
                    else:
                        results.append(it)

        # Sort results: misclassified with high confidence first
        results.sort(key=lambda x: (not x["is_misclassified"], -x["confidence"]))
        return results


class RelabelApp:
    """Tkinter-based GUI for reviewing and relabeling face images with single key presses."""

    def __init__(
        self,
        items: List[dict[str, Any]],
        data_root: Path,
        target_dir: Path,
        output_csv: Path,
    ):
        self.items = items
        self.data_root = data_root
        self.target_dir = target_dir
        self.raw_dir = data_root / "raw"
        self.trash_dir = data_root / "_trash"
        self.output_csv = output_csv

        self.current_idx = 0
        self.history_actions: List[dict[str, Any]] = []
        self.audit_log: List[dict[str, Any]] = []

        import tkinter as tk
        from tkinter import ttk, messagebox
        from PIL import ImageTk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.ImageTk = ImageTk

        self.root = tk.Tk()
        self.root.title("Face Occlusion Rapid Relabeling Tool")
        self.root.geometry("820x680")
        self.root.minsize(720, 600)

        self._setup_ui()
        self._bind_keys()
        self._load_current_image()

    def _setup_ui(self):
        # Top Header Frame
        header_frame = self.tk.Frame(self.root, bg="#2c3e50", padx=16, pady=10)
        header_frame.pack(side="top", fill="x")

        self.lbl_progress = self.tk.Label(
            header_frame,
            text="Item 0 / 0",
            font=("Segoe UI", 12, "bold"),
            fg="#ecf0f1",
            bg="#2c3e50",
        )
        self.lbl_progress.pack(side="left")

        self.lbl_stats = self.tk.Label(
            header_frame,
            text="Relabeled: 0 | Deleted: 0 | Skipped: 0",
            font=("Segoe UI", 10),
            fg="#bdc3c7",
            bg="#2c3e50",
        )
        self.lbl_stats.pack(side="right")

        # Center Main Area
        main_frame = self.tk.Frame(self.root, padx=20, pady=15)
        main_frame.pack(side="top", fill="both", expand=True)

        # Image Canvas Frame
        img_container = self.tk.Frame(main_frame, bg="#f5f6fa", bd=2, relief="groove")
        img_container.pack(side="top", fill="both", expand=True, pady=(0, 10))

        self.lbl_image = self.tk.Label(img_container, bg="#f5f6fa")
        self.lbl_image.pack(fill="both", expand=True, padx=10, pady=10)

        # Info Box Frame
        info_frame = self.tk.Frame(main_frame, bg="#ffffff", bd=1, relief="solid", padx=12, pady=8)
        info_frame.pack(side="top", fill="x", pady=(0, 10))

        self.lbl_filename = self.tk.Label(
            info_frame, text="Filename: -", font=("Segoe UI", 10, "bold"), bg="#ffffff", anchor="w"
        )
        self.lbl_filename.pack(fill="x")

        badges_frame = self.tk.Frame(info_frame, bg="#ffffff")
        badges_frame.pack(fill="x", pady=(4, 0))

        self.lbl_true = self.tk.Label(
            badges_frame,
            text="Current: CLEAR",
            font=("Segoe UI", 11, "bold"),
            bg="#3498db",
            fg="white",
            padx=10,
            pady=4,
        )
        self.lbl_true.pack(side="left", padx=(0, 10))

        self.lbl_pred = self.tk.Label(
            badges_frame,
            text="Model: OCCLUDED (95.0%)",
            font=("Segoe UI", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=10,
            pady=4,
        )
        self.lbl_pred.pack(side="left", padx=(0, 10))

        self.lbl_dimensions = self.tk.Label(
            badges_frame, text="Size: 112x112", font=("Segoe UI", 10), bg="#ffffff", fg="#7f8c8d"
        )
        self.lbl_dimensions.pack(side="right")

        # Action Buttons Frame
        btn_frame = self.tk.Frame(self.root, padx=15, pady=12, bg="#ecf0f1")
        btn_frame.pack(side="bottom", fill="x")

        btn_style_clear = {"font": ("Segoe UI", 10, "bold"), "bg": "#27ae60", "fg": "white", "padx": 14, "pady": 6}
        btn_style_occ = {"font": ("Segoe UI", 10, "bold"), "bg": "#d35400", "fg": "white", "padx": 14, "pady": 6}
        btn_style_del = {"font": ("Segoe UI", 10, "bold"), "bg": "#c0392b", "fg": "white", "padx": 12, "pady": 6}
        btn_style_skip = {"font": ("Segoe UI", 10), "bg": "#7f8c8d", "fg": "white", "padx": 12, "pady": 6}
        btn_style_undo = {"font": ("Segoe UI", 10), "bg": "#95a5a6", "fg": "white", "padx": 12, "pady": 6}

        self.btn_clear = self.tk.Button(
            btn_frame, text="[1] Set CLEAR", command=lambda: self._apply_action("clear"), **btn_style_clear
        )
        self.btn_clear.pack(side="left", padx=5)

        self.btn_occ = self.tk.Button(
            btn_frame, text="[2] Set OCCLUDED", command=lambda: self._apply_action("occluded"), **btn_style_occ
        )
        self.btn_occ.pack(side="left", padx=5)

        self.btn_del = self.tk.Button(
            btn_frame, text="[D] Delete / Trash", command=lambda: self._apply_action("delete"), **btn_style_del
        )
        self.btn_del.pack(side="left", padx=5)

        self.btn_undo = self.tk.Button(
            btn_frame, text="[Z] Undo", command=self._undo_last_action, **btn_style_undo
        )
        self.btn_undo.pack(side="right", padx=5)

        self.btn_skip = self.tk.Button(
            btn_frame, text="[Space] Skip / Keep", command=lambda: self._apply_action("skip"), **btn_style_skip
        )
        self.btn_skip.pack(side="right", padx=5)

    def _bind_keys(self):
        self.root.bind("1", lambda e: self._apply_action("clear"))
        self.root.bind("2", lambda e: self._apply_action("occluded"))
        self.root.bind("<Delete>", lambda e: self._apply_action("delete"))
        self.root.bind("d", lambda e: self._apply_action("delete"))
        self.root.bind("D", lambda e: self._apply_action("delete"))
        self.root.bind("<space>", lambda e: self._apply_action("skip"))
        self.root.bind("<Right>", lambda e: self._apply_action("skip"))
        self.root.bind("z", lambda e: self._undo_last_action())
        self.root.bind("Z", lambda e: self._undo_last_action())
        self.root.bind("<Left>", lambda e: self._undo_last_action())
        self.root.bind("<Escape>", lambda e: self._on_close())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_current_image(self):
        if self.current_idx >= len(self.items):
            self._show_completion()
            return

        item = self.items[self.current_idx]
        img_path = Path(item["path"])

        # Update Progress & Info
        self.lbl_progress.config(text=f"Item {self.current_idx + 1} / {len(self.items)}")
        self._update_stats_display()

        self.lbl_filename.config(text=f"File: {item['name']}")
        self.lbl_dimensions.config(text=f"Size: {item.get('width', '-')}x{item.get('height', '-')}")

        true_cls = item["true_class"]
        pred_cls = item["pred_class"]
        conf = item["confidence"] * 100

        # Colors for badges
        true_color = "#27ae60" if true_cls == "clear" else "#d35400"
        self.lbl_true.config(text=f"Current: {true_cls.upper()}", bg=true_color)

        pred_color = "#27ae60" if pred_cls == "clear" else "#d35400"
        self.lbl_pred.config(text=f"Model: {pred_cls.upper()} ({conf:.1f}%)", bg=pred_color)

        # Load & resize image for preview
        try:
            pil_img = Image.open(img_path).convert("RGB")
            # Fit inside canvas box ~360x360
            max_size = (380, 380)
            pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
            tk_img = self.ImageTk.PhotoImage(pil_img)
            self.lbl_image.config(image=tk_img)
            self.lbl_image.image = tk_img
        except Exception as e:
            self.lbl_image.config(text=f"Cannot load image:\n{e}", image="")

    def _apply_action(self, action: str):
        if self.current_idx >= len(self.items):
            return

        item = self.items[self.current_idx]
        src_path = Path(item["path"])
        filename = item["name"]
        old_cls = item["true_class"]

        undo_record: dict[str, Any] = {
            "index": self.current_idx,
            "item": item,
            "action": action,
            "moves": [],
        }

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if action in ["clear", "occluded"]:
            target_cls = action
            if target_cls != old_cls:
                # Move file in target dir
                dest_dir = src_path.parent.parent / target_cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / filename
                if src_path.exists():
                    shutil.move(str(src_path), str(dest_path))
                    undo_record["moves"].append((dest_path, src_path))
                    item["path"] = dest_path
                    item["true_class"] = target_cls

                # Sync with raw dir if exists
                if self.raw_dir.exists():
                    raw_src = self.raw_dir / old_cls / filename
                    raw_dst_dir = self.raw_dir / target_cls
                    raw_dst = raw_dst_dir / filename
                    if raw_src.exists():
                        raw_dst_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(raw_src), str(raw_dst))
                        undo_record["moves"].append((raw_dst, raw_src))

                self.audit_log.append({
                    "timestamp": timestamp,
                    "filename": filename,
                    "old_class": old_cls,
                    "new_class": target_cls,
                    "model_pred": item["pred_class"],
                    "confidence": f"{item['confidence']:.4f}",
                    "action": "relabel",
                })
            else:
                action = "skip"

        elif action == "delete":
            self.trash_dir.mkdir(parents=True, exist_ok=True)
            trash_path = self.trash_dir / f"{old_cls}_{filename}"
            if src_path.exists():
                shutil.move(str(src_path), str(trash_path))
                undo_record["moves"].append((trash_path, src_path))

            # Sync delete from raw
            if self.raw_dir.exists():
                raw_src = self.raw_dir / old_cls / filename
                raw_trash = self.trash_dir / f"raw_{old_cls}_{filename}"
                if raw_src.exists():
                    shutil.move(str(raw_src), str(raw_trash))
                    undo_record["moves"].append((raw_trash, raw_src))

            self.audit_log.append({
                "timestamp": timestamp,
                "filename": filename,
                "old_class": old_cls,
                "new_class": "DELETED",
                "model_pred": item["pred_class"],
                "confidence": f"{item['confidence']:.4f}",
                "action": "delete",
            })

        elif action == "skip":
            self.audit_log.append({
                "timestamp": timestamp,
                "filename": filename,
                "old_class": old_cls,
                "new_class": old_cls,
                "model_pred": item["pred_class"],
                "confidence": f"{item['confidence']:.4f}",
                "action": "keep",
            })

        self.history_actions.append(undo_record)
        self.current_idx += 1
        self._load_current_image()

    def _undo_last_action(self):
        if not self.history_actions:
            self.messagebox.showinfo("Undo", "Không còn thao tác nào để hoàn tác.")
            return

        last_action = self.history_actions.pop()
        for curr_path, orig_path in reversed(last_action["moves"]):
            if curr_path.exists():
                orig_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(curr_path), str(orig_path))

        # Revert item properties
        item = last_action["item"]
        if last_action["action"] in ["clear", "occluded"]:
            item["true_class"] = last_action["moves"][0][1].parent.name if last_action["moves"] else item["true_class"]
            item["path"] = last_action["moves"][0][1] if last_action["moves"] else item["path"]

        if self.audit_log:
            self.audit_log.pop()

        self.current_idx = last_action["index"]
        self._load_current_image()

    def _update_stats_display(self):
        relabeled = sum(1 for a in self.audit_log if a["action"] == "relabel")
        deleted = sum(1 for a in self.audit_log if a["action"] == "delete")
        skipped = sum(1 for a in self.audit_log if a["action"] == "keep")
        self.lbl_stats.config(text=f"Relabeled: {relabeled} | Deleted: {deleted} | Skipped: {skipped}")

    def _show_completion(self):
        self._save_audit_log()
        relabeled = sum(1 for a in self.audit_log if a["action"] == "relabel")
        deleted = sum(1 for a in self.audit_log if a["action"] == "delete")
        skipped = sum(1 for a in self.audit_log if a["action"] == "keep")

        msg = (
            f"🎉 Đã duyệt hết {len(self.items)} ảnh cần xem xét!\n\n"
            f"• Đã sửa nhãn: {relabeled}\n"
            f"• Đã xóa ảnh: {deleted}\n"
            f"• Giữ nguyên: {skipped}\n\n"
            f"Lịch sử đã lưu tại: {self.output_csv}\n\n"
            f"Bước tiếp theo: Bạn hãy chạy lại 'split_dataset.py' để cập nhật tập train/val mới!"
        )
        self.messagebox.showinfo("Hoàn tất duyệt nhãn", msg)
        self.root.destroy()

    def _save_audit_log(self):
        if not self.audit_log:
            return
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.output_csv.exists()
        with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["timestamp", "filename", "old_class", "new_class", "model_pred", "confidence", "action"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.audit_log)
        print(f"Audit log written to: {self.output_csv.resolve()}")

    def _on_close(self):
        self._save_audit_log()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Rapid Face Occlusion Relabeling Tool")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--split",
        choices=["train", "val", "raw"],
        default="train",
        help="Dataset split to scan for errors (default: train)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Confidence threshold for uncertainty (default: 0.60)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review ALL images in split, not just errors/uncertain ones",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/relabel_history.csv",
        help="Path to save relabeling audit CSV",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only scan and output error statistics to terminal/CSV without opening GUI",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["data"]["root"])
    target_dir = data_root / args.split

    print("=" * 60)
    print("      RAPID FACE OCCLUSION RELABELING TOOL")
    print("=" * 60)
    print(f"Data root:   {data_root}")
    print(f"Target split:{args.split} ({target_dir})")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Threshold:   {args.threshold}")
    print(f"Mode:        {'All images' if args.all else 'Errors + Low-confidence only'}")
    print("=" * 60)

    scanner = DatasetScanner(args.config, args.checkpoint)
    items = scanner.scan(
        target_dir=target_dir,
        confidence_threshold=args.threshold,
        only_misclassified=not args.all,
    )

    mis_count = sum(1 for x in items if x["is_misclassified"])
    unc_count = sum(1 for x in items if not x["is_misclassified"] and x["is_uncertain"])

    print(f"\nScan completed:")
    print(f"  • Total suspicious images found: {len(items)}")
    print(f"    - Misclassified:  {mis_count}")
    print(f"    - Low confidence: {unc_count}")

    if not items:
        print("\n🎉 Không phát hiện ảnh nào bị đoán sai hoặc có độ tự tin thấp trong tập này!")
        return

    if args.scan_only:
        scan_csv = Path("outputs/scan_errors.csv")
        scan_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(scan_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["filename", "true_class", "pred_class", "confidence", "is_misclassified", "path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for it in items:
                writer.writerow({
                    "filename": it["name"],
                    "true_class": it["true_class"],
                    "pred_class": it["pred_class"],
                    "confidence": f"{it['confidence']:.4f}",
                    "is_misclassified": it["is_misclassified"],
                    "path": str(it["path"]),
                })
        print(f"Saved scan results to: {scan_csv.resolve()}")
        return

    print("\nLaunching Relabel GUI...")
    app = RelabelApp(
        items=items,
        data_root=data_root,
        target_dir=target_dir,
        output_csv=Path(args.output_csv),
    )
    app.run()


if __name__ == "__main__":
    main()
