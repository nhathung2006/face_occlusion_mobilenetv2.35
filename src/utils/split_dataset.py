from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any

import numpy as np
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


def get_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of image file content to detect binary duplicates."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_group_id(filename: str) -> str:
    """Group crops belonging to the same source image, face identity, or recording session to avoid leakage."""
    # Pattern 1: WIDER FACE base image (e.g. 0--Parade_...jpg_b1234_masked.jpg, 0--Parade_...jpg_b1234.jpg)
    m_wider = re.match(r"^(.*\.jpg)_b\d+(?:_masked)?(?:\.jpg)?$", filename, re.IGNORECASE)
    if m_wider:
        return f"wider_{m_wider.group(1)}"

    # Pattern 2: Camera stream crop session (e.g. 01a0b348-9ab8-73e3-9797-86234c0f24be_nvr-llWGaUl9M9E771xokkJUiNr5fMyJBigI_1789714275642_img_face1.jpg)
    m_cam = re.match(r"^([a-f0-9\-]+_[a-zA-Z0-9\-]+_\d+)", filename, re.IGNORECASE)
    if m_cam:
        return f"cam_{m_cam.group(1)}"

    # Pattern 3: Numeric face ID with optional mask/aligned (e.g. 000159_face.jpg, 000159_face_masked.jpg, 000159_face1.jpg)
    m_face_num = re.match(r"^(\d+)_face", filename, re.IGNORECASE)
    if m_face_num:
        return f"facenum_{m_face_num.group(1)}"

    # Pattern 4: Face dataset aligned/crop IDs (e.g. face_0028_aligned112.jpg, face_0028_crop.jpg)
    m_face = re.match(r"^(face_\d+)", filename, re.IGNORECASE)
    if m_face:
        return f"face_id_{m_face.group(1)}"

    # Pattern 5: Lumi staff recording session (e.g. PhongLT_20260415_090449_774.jpg -> PhongLT_20260415_0904)
    m_person_session = re.match(r"^([A-Za-z]+)_(\d{8}_\d{4})", filename)
    if m_person_session:
        return f"person_session_{m_person_session.group(1)}_{m_person_session.group(2)}"

    # Pattern 6: Lumi staff person generic
    m_person = re.match(r"^([A-Za-z]+)_[0-9]+", filename)
    if m_person:
        return f"person_{m_person.group(1)}"

    # Pattern 7: Single file
    return f"single_{filename}"


def get_source_type(filename: str) -> str:
    """Classify the origin / domain / occlusion style of the face crop."""
    if "nvr-" in filename or filename.startswith(("01a", "01b", "01c", "01d", "01e", "01f")):
        return "cam_stream"
    if "--" in filename:
        if "_masked" in filename.lower():
            return "wider_synth_mask"
        return "wider_face"
    if filename.lower().startswith("with_mask_"):
        return "real_mask"
    if "_masked" in filename.lower():
        return "synth_mask"
    if any(filename.startswith(k) for k in ["PhongLT", "DoanNV", "MaiNT", "Hung", "Tuan"]):
        return "lumi_staff"
    if re.match(r"^\d+_face", filename) or filename.startswith("face_"):
        return "face_crops"
    return "other"


def image_quality_features(img: Image.Image) -> tuple[float, float, float]:
    """Return approximate sharpness, contrast, and brightness features."""
    gray = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(laplacian.var()), float(gray.std()), float(gray.mean())


def get_aspect_ratio_bin(aspect_ratio: float) -> str:
    if aspect_ratio < 0.80:
        return "portrait(<0.80)"
    if aspect_ratio <= 1.25:
        return "square(0.80-1.25)"
    if aspect_ratio <= 1.80:
        return "landscape(1.25-1.80)"
    return "wide(>1.80)"


def split_data(
    data_root: Path,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_dir = data_root / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    for cls_name in ["clear", "occluded"]:
        cls_dir = raw_dir / cls_name
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.glob("*")):
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                with Image.open(f) as img:
                    w, h = img.size
                    sharpness, contrast, brightness = image_quality_features(img)
                is_sq = (w == h)
                ar = round(w / h, 2)
                size_bin = "small(<75)" if max(w, h) < 75 else ("medium(75-111)" if max(w, h) < 112 else "large(>=112)")
                source_type = get_source_type(f.name)
                name_grp = get_group_id(f.name)
                sha256_hash = get_file_sha256(f)
                items.append({
                    "path": f,
                    "name": f.name,
                    "class": cls_name,
                    "width": w,
                    "height": h,
                    "aspect_ratio": ar,
                    "aspect_ratio_bin": get_aspect_ratio_bin(ar),
                    "is_square": is_sq,
                    "size_bin": size_bin,
                    "sharpness": sharpness,
                    "contrast": contrast,
                    "brightness": brightness,
                    "source_type": source_type,
                    "name_group": name_grp,
                    "sha256": sha256_hash,
                })

    total_count = len(items)
    if total_count == 0:
        raise ValueError(f"No valid images found in {raw_dir}")

    sharpness_values = np.asarray([it["sharpness"] for it in items], dtype=np.float64)
    contrast_values = np.asarray([it["contrast"] for it in items], dtype=np.float64)
    sharpness_cutoffs = np.quantile(sharpness_values, [1 / 3, 2 / 3])
    contrast_cutoffs = np.quantile(contrast_values, [1 / 3, 2 / 3])
    for it in items:
        it["sharpness_bin"] = (
            "soft/blurry" if it["sharpness"] <= sharpness_cutoffs[0]
            else "moderate" if it["sharpness"] <= sharpness_cutoffs[1]
            else "sharp"
        )
        it["contrast_bin"] = (
            "low_contrast" if it["contrast"] <= contrast_cutoffs[0]
            else "moderate" if it["contrast"] <= contrast_cutoffs[1]
            else "high_contrast"
        )
        difficulty_score = 0.0
        difficulty_score += {"small(<75)": 1.5, "medium(75-111)": 0.6, "large(>=112)": 0.0}[it["size_bin"]]
        difficulty_score += {"soft/blurry": 1.0, "moderate": 0.35, "sharp": 0.0}[it["sharpness_bin"]]
        difficulty_score += {"low_contrast": 0.8, "moderate": 0.25, "high_contrast": 0.0}[it["contrast_bin"]]
        if it["brightness"] < 50 or it["brightness"] > 210:
            difficulty_score += 0.8
        it["difficulty_score"] = round(difficulty_score, 3)
        it["difficulty_bin"] = (
            "easy" if difficulty_score < 0.8
            else "medium" if difficulty_score < 1.8
            else "hard"
        )

    # Disjoint Set / Union-Find: Merge groups sharing the same binary hash or name_group
    parent: dict[int, int] = {i: i for i in range(total_count)}

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    hash_to_indices = defaultdict(list)
    name_group_to_indices = defaultdict(list)

    for i, it in enumerate(items):
        hash_to_indices[it["sha256"]].append(i)
        name_group_to_indices[it["name_group"]].append(i)

    # Union all identical file contents (zero duplicate hash across splits)
    for indices in hash_to_indices.values():
        for idx in indices[1:]:
            union(indices[0], idx)

    # Union all items sharing same metadata group
    for grp_name, indices in name_group_to_indices.items():
        if not grp_name.startswith("single_"):
            for idx in indices[1:]:
                union(indices[0], idx)

    for i, it in enumerate(items):
        it["group"] = f"group_{find(i)}"

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)

    # Balance marginal distributions while keeping duplicate/source groups intact.
    # Features include class, source domain, resolution, aspect ratio, sharpness,
    # contrast, and an image-quality difficulty proxy.
    feature_fields = {
        "class": 5.0,
        "aspect_ratio_bin": 3.0,
        "difficulty_bin": 3.0,
        "size_bin": 2.0,
        "sharpness_bin": 2.0,
        "contrast_bin": 2.0,
        "source_type": 1.0,
    }
    group_list = list(groups.values())
    group_vectors: list[Counter[str]] = []
    total_features: Counter[str] = Counter()
    for group_items in group_list:
        vector: Counter[str] = Counter()
        for item in group_items:
            for field in feature_fields:
                vector[f"{field}={item[field]}"] += 1
        group_vectors.append(vector)
        total_features.update(vector)

    target_val_count = total_count * val_ratio
    target_features = {key: value * val_ratio for key, value in total_features.items()}

    def split_cost(val_count: int, val_features: Counter[str]) -> float:
        count_error = (val_count - target_val_count) / max(target_val_count, 1.0)
        cost = 10.0 * count_error * count_error
        for key, target in target_features.items():
            error = (val_features[key] - target) / max(target, 1.0)
            field = key.split("=", 1)[0]
            cost += feature_fields[field] * error * error
        return cost

    best_val_indices: set[int] | None = None
    best_cost = float("inf")
    # Several seeded starts plus local swaps reduce dependence on input ordering.
    for _ in range(4):
        order = list(range(len(group_list)))
        rng.shuffle(order)
        val_indices: set[int] = set()
        val_count = 0
        val_features: Counter[str] = Counter()
        for index in order:
            group_size = len(group_list[index])
            if val_count < target_val_count:
                val_indices.add(index)
                val_count += group_size
                val_features.update(group_vectors[index])

        train_indices = set(range(len(group_list))) - val_indices
        current_cost = split_cost(val_count, val_features)
        for _step in range(12000):
            if not val_indices or not train_indices:
                break
            val_index = rng.choice(tuple(val_indices))
            train_index = rng.choice(tuple(train_indices))
            next_count = val_count - len(group_list[val_index]) + len(group_list[train_index])
            next_features = val_features.copy()
            next_features.subtract(group_vectors[val_index])
            next_features.update(group_vectors[train_index])
            next_cost = split_cost(next_count, next_features)
            if next_cost + 1e-12 < current_cost:
                val_indices.remove(val_index)
                val_indices.add(train_index)
                train_indices.remove(train_index)
                train_indices.add(val_index)
                val_count = next_count
                val_features = next_features
                current_cost = next_cost

        if current_cost < best_cost:
            best_cost = current_cost
            best_val_indices = val_indices

    if best_val_indices is None:
        raise RuntimeError("Could not create train/validation split")

    val_items = [item for index, group_items in enumerate(group_list) if index in best_val_indices for item in group_items]
    train_items = [item for index, group_items in enumerate(group_list) if index not in best_val_indices for item in group_items]

    return train_items, val_items


def apply_split(
    data_root: Path,
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
) -> None:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    staging_dir = data_root / f".split_staging_{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    train_stage = staging_dir / "train"
    val_stage = staging_dir / "val"
    for split_dir in (train_stage, val_stage):
        for cls_name in ("clear", "occluded"):
            (split_dir / cls_name).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    try:
        print(f"Copying {len(train_items)} train images...")
        for split_name, split_items, split_dir in (
            ("train", train_items, train_stage),
            ("val", val_items, val_stage),
        ):
            for it in split_items:
                dst = split_dir / it["class"] / it["name"]
                shutil.copy2(it["path"], dst)
                manifest_rows.append({
                    "filename": it["name"],
                    "split": split_name,
                    "class": it["class"],
                    "width": it["width"],
                    "height": it["height"],
                    "aspect_ratio": it["aspect_ratio"],
                    "aspect_ratio_bin": it["aspect_ratio_bin"],
                    "is_square": it["is_square"],
                    "size_bin": it["size_bin"],
                    "sharpness": round(it["sharpness"], 4),
                    "sharpness_bin": it["sharpness_bin"],
                    "contrast": round(it["contrast"], 4),
                    "contrast_bin": it["contrast_bin"],
                    "brightness": round(it["brightness"], 4),
                    "difficulty_score": it["difficulty_score"],
                    "difficulty_bin": it["difficulty_bin"],
                    "source_type": it["source_type"],
                    "group_id": it["group"],
                    "source_path": str(it["path"]),
                })
            if split_name == "train":
                print(f"Copying {len(val_items)} validation images...")

        manifest_path = data_root / "split_manifest.csv"
        manifest_temp = staging_dir / "split_manifest.csv"
        fieldnames = list(manifest_rows[0].keys())
        with manifest_temp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        # Replace the old splits only after the complete new split was copied.
        for old_dir in (train_dir, val_dir):
            if old_dir.exists():
                shutil.rmtree(old_dir)
        shutil.move(str(train_stage), str(train_dir))
        shutil.move(str(val_stage), str(val_dir))
        if manifest_path.exists():
            manifest_path.unlink()
        shutil.move(str(manifest_temp), str(manifest_path))
        print(f"Manifest written to: {manifest_path}")
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


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

    print("=" * 70)
    print("       GROUP-AWARE DATASET SPLIT (CLASS / QUALITY / ASPECT RATIO)")
    print("=" * 70)
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

    print("\n" + "=" * 70)
    print("                         SPLIT SUMMARY")
    print("=" * 70)
    print(f"Total Images: {total_imgs}")
    print(f"Train Set:    {len(train_items):4d} ({len(train_items)/total_imgs*100:.1f}%) [clear: {tr_clear:4d} ({tr_clear/len(train_items)*100:.1f}%), occluded: {tr_occ:4d} ({tr_occ/len(train_items)*100:.1f}%)]")
    print(f"Val Set:      {len(val_items):4d} ({len(val_items)/total_imgs*100:.1f}%) [clear: {val_clear:4d} ({val_clear/len(val_items)*100:.1f}%), occluded: {val_occ:4d} ({val_occ/len(val_items)*100:.1f}%)]")
    print("-" * 70)
    print("Train Size (Difficulty) Distribution:")
    for k, v in sorted(Counter(x["size_bin"] for x in train_items).items()):
        print(f"  - {k:<15}: {v:4d} ({v/len(train_items)*100:.1f}%)")
    print("Val Size (Difficulty) Distribution:")
    for k, v in sorted(Counter(x["size_bin"] for x in val_items).items()):
        print(f"  - {k:<15}: {v:4d} ({v/len(val_items)*100:.1f}%)")
    print("-" * 70)
    for title, field in (
        ("Aspect ratio", "aspect_ratio_bin"),
        ("Image difficulty", "difficulty_bin"),
        ("Sharpness", "sharpness_bin"),
    ):
        print(f"{title} distribution (train / val):")
        train_counts = Counter(x[field] for x in train_items)
        val_counts = Counter(x[field] for x in val_items)
        for key in sorted(set(train_counts) | set(val_counts)):
            tr_pct = train_counts[key] / max(len(train_items), 1) * 100
            va_pct = val_counts[key] / max(len(val_items), 1) * 100
            print(f"  - {key:<22}: {train_counts[key]:4d} ({tr_pct:5.1f}%) / {val_counts[key]:4d} ({va_pct:5.1f}%)")
    print("-" * 70)
    print("Train Domain/Source Distribution:")
    for k, v in sorted(Counter(x["source_type"] for x in train_items).items()):
        print(f"  - {k:<18}: {v:4d} ({v/len(train_items)*100:.1f}%)")
    print("Val Domain/Source Distribution:")
    for k, v in sorted(Counter(x["source_type"] for x in val_items).items()):
        print(f"  - {k:<18}: {v:4d} ({v/len(val_items)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
