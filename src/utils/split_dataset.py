from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any

from PIL import Image
import yaml

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


def get_group_id(filename: str) -> str:
    """Group crops belonging to the same source image or recording session."""
    # Pattern 1: WIDER FACE base image (e.g. 0--Parade_...jpg_b1234.jpg)
    m_wider = re.match(r"^(.*\.jpg)_b\d+\.jpg$", filename, re.IGNORECASE)
    if m_wider:
        return f"wider_{m_wider.group(1)}"

    # Pattern 2: Face dataset aligned/crop IDs (e.g. face_0028_aligned112.jpg, face_0028_crop.jpg)
    m_face = re.match(r"^(face_\d+)", filename, re.IGNORECASE)
    if m_face:
        return f"face_id_{m_face.group(1)}"

    # Pattern 3: Lumi staff recording session (e.g. PhongLT_20260415_090449_774.jpg -> PhongLT_20260415_0904)
    m_person_session = re.match(r"^([A-Za-z]+)_(\d{8}_\d{4})", filename)
    if m_person_session:
        return f"person_session_{m_person_session.group(1)}_{m_person_session.group(2)}"

    # Pattern 4: Lumi staff person generic
    m_person = re.match(r"^([A-Za-z]+)_[0-9]+", filename)
    if m_person:
        return f"person_{m_person.group(1)}"

    # Pattern 5: Single file
    return f"single_{filename}"


def split_data(
    data_root: Path,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_dir = data_root / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    random.seed(seed)
    items: list[dict[str, Any]] = []

    for cls_name in ["clear", "occluded"]:
        cls_dir = raw_dir / cls_name
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob("*")):
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                with Image.open(f) as img:
                    w, h = img.size
                is_sq = (w == h)
                ar = round(w / h, 2)
                size_bin = "small" if max(w, h) < 75 else ("medium" if max(w, h) < 112 else "large")
                items.append({
                    "path": f,
                    "name": f.name,
                    "class": cls_name,
                    "width": w,
                    "height": h,
                    "aspect_ratio": ar,
                    "is_square": is_sq,
                    "size_bin": size_bin,
                    "group": get_group_id(f.name),
                })

    total_count = len(items)
    if total_count == 0:
        raise ValueError(f"No valid images found in {raw_dir}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)

    # Multi-criteria Stratification Signature
    def group_signature(g: list[dict[str, Any]]) -> tuple[str, str, bool]:
        c_clear = sum(1 for x in g if x["class"] == "clear")
        c_occ = sum(1 for x in g if x["class"] == "occluded")
        dom_cls = "clear" if c_clear >= c_occ else "occluded"
        sizes = Counter(x["size_bin"] for x in g)
        dom_size = sizes.most_common(1)[0][0]
        has_non_sq = any(not x["is_square"] for x in g)
        return (dom_cls, dom_size, has_non_sq)

    strata: dict[tuple[str, str, bool], list[list[dict[str, Any]]]] = defaultdict(list)
    for g in groups.values():
        strata[group_signature(g)].append(g)

    train_items: list[dict[str, Any]] = []
    val_items: list[dict[str, Any]] = []

    for sig, g_list in sorted(strata.items(), key=lambda x: str(x[0])):
        random.shuffle(g_list)
        n_groups = len(g_list)
        n_val = int(round(n_groups * val_ratio))
        for i, g in enumerate(g_list):
            if i < n_val:
                val_items.extend(g)
            else:
                train_items.extend(g)

    return train_items, val_items


def apply_split(
    data_root: Path,
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
) -> None:
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    # Ensure clean target folders
    for split_dir in [train_dir, val_dir]:
        for cls_name in ["clear", "occluded"]:
            target_cls = split_dir / cls_name
            if target_cls.exists():
                shutil.rmtree(target_cls)
            target_cls.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    print(f"Copying {len(train_items)} train images...")
    for it in train_items:
        dst = train_dir / it["class"] / it["name"]
        shutil.copy2(it["path"], dst)
        manifest_rows.append({
            "filename": it["name"],
            "split": "train",
            "class": it["class"],
            "width": it["width"],
            "height": it["height"],
            "aspect_ratio": it["aspect_ratio"],
            "is_square": it["is_square"],
            "size_bin": it["size_bin"],
            "group_id": it["group"],
            "source_path": str(it["path"]),
        })

    print(f"Copying {len(val_items)} validation images...")
    for it in val_items:
        dst = val_dir / it["class"] / it["name"]
        shutil.copy2(it["path"], dst)
        manifest_rows.append({
            "filename": it["name"],
            "split": "val",
            "class": it["class"],
            "width": it["width"],
            "height": it["height"],
            "aspect_ratio": it["aspect_ratio"],
            "is_square": it["is_square"],
            "size_bin": it["size_bin"],
            "group_id": it["group"],
            "source_path": str(it["path"]),
        })

    manifest_path = data_root / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename",
            "split",
            "class",
            "width",
            "height",
            "aspect_ratio",
            "is_square",
            "size_bin",
            "group_id",
            "source_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Manifest written to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Stratified Group Split for Face Occlusion Dataset")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation ratio (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["root"])
    seed = int(args.seed if args.seed is not None else cfg["training"].get("seed", 42))
    val_ratio = float(args.val_ratio)

    print("=" * 60)
    print("       DATASET STRATIFIED GROUP SPLIT (80/20)")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Seed:      {seed}")
    print(f"Val ratio: {val_ratio:.2f}")

    train_items, val_items = split_data(data_root, val_ratio=val_ratio, seed=seed)
    total_imgs = len(train_items) + len(val_items)

    apply_split(data_root, train_items, val_items)

    tr_clear = sum(1 for x in train_items if x["class"] == "clear")
    tr_occ = sum(1 for x in train_items if x["class"] == "occluded")
    val_clear = sum(1 for x in val_items if x["class"] == "clear")
    val_occ = sum(1 for x in val_items if x["class"] == "occluded")

    print("\n" + "=" * 60)
    print("                   SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total Images: {total_imgs}")
    print(f"Train Set:    {len(train_items):4d} ({len(train_items)/total_imgs*100:.1f}%) [clear: {tr_clear:3d}, occluded: {tr_occ:3d}]")
    print(f"Val Set:      {len(val_items):4d} ({len(val_items)/total_imgs*100:.1f}%) [clear: {val_clear:3d}, occluded: {val_occ:3d}]")
    print("-" * 60)
    print("Train Size Distribution:", dict(Counter(x["size_bin"] for x in train_items)))
    print("Val Size Distribution:  ", dict(Counter(x["size_bin"] for x in val_items)))
    print("-" * 60)
    print("Train Square Distribution: True=", sum(1 for x in train_items if x["is_square"]), "False=", sum(1 for x in train_items if not x["is_square"]))
    print("Val Square Distribution:   True=", sum(1 for x in val_items if x["is_square"]), "False=", sum(1 for x in val_items if not x["is_square"]))
    print("=" * 60)


if __name__ == "__main__":
    main()
