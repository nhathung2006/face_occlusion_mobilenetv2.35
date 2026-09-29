"""Copy the current four-class dataset into an immutable train/val snapshot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.fft import dctn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import train_4class_single_logit as trainer  # noqa: E402
from src.utils.split_dataset import get_group_id  # noqa: E402


def _pixel_and_perceptual_hash(path: Path) -> tuple[str, int]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixel_hash = hashlib.sha256(str(rgb.size).encode() + b"\0" + rgb.tobytes()).hexdigest()
        gray = np.asarray(rgb.convert("L").resize((32, 32)), dtype=float)
    dct = dctn(gray, norm="ortho")[:8, :8].flatten()
    bits = dct > np.median(dct[1:])
    bits[0] = False
    return pixel_hash, int.from_bytes(np.packbits(bits).tobytes(), "big")


def _group_and_assign(records: list[dict], class_names: list[str], fraction: float,
                      seed: int, max_distance: int) -> dict[str, dict[str, int]]:
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    by_pixel, by_name = {}, {}
    for i, record in enumerate(records):
        pixel = record["pixel_sha256"]
        if pixel in by_pixel:
            other = records[by_pixel[pixel]]
            if other["class_name"] != record["class_name"]:
                raise ValueError(f"Identical pixels have conflicting labels: {other['path']} and {record['path']}")
            union(i, by_pixel[pixel])
        else:
            by_pixel[pixel] = i
        name_group = get_group_id(Path(record["path"]).name)
        if name_group in by_name:
            union(i, by_name[name_group])
        else:
            by_name[name_group] = i

    for i, record in enumerate(records):
        for j in range(i):
            if (record["phash"] ^ records[j]["phash"]).bit_count() <= max_distance:
                union(i, j)

    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record)
    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)
    targets = {name: round(sum(r["class_name"] == name for r in records) * fraction)
               for name in class_names}
    val_counts = Counter()
    for group in group_list:
        group_counts = Counter(record["class_name"] for record in group)
        improvement = sum(
            abs(targets[name] - val_counts[name] - group_counts[name])
            - abs(targets[name] - val_counts[name]) for name in class_names
        )
        split = "val" if improvement < 0 else "train"
        if split == "val":
            val_counts.update(group_counts)
        group_id = min(record["sha256"] for record in group)
        for record in group:
            record["split"] = split
            record["group_id"] = group_id
            del record["phash"]
    counts = {name: {"train": sum(r["class_name"] == name and r["split"] == "train" for r in records),
                     "val": val_counts[name],
                     "total": sum(r["class_name"] == name for r in records)}
              for name in class_names}
    if any(not row["train"] or not row["val"] for row in counts.values()):
        raise ValueError(f"Similarity grouping left a class without train or val images: {counts}")
    return counts


def prepare(config_path: Path, output_root: Path) -> dict:
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists() or output_root.with_name(output_root.name + ".incomplete").exists():
        raise FileExistsError(f"Split output already exists; choose a new folder: {output_root}")
    config = trainer.load_config(config_path)
    data_cfg = config["data"]
    if data_cfg.get("split_manifest") or data_cfg.get("split_root"):
        raise ValueError("Build a new split from the original four-class config, not another frozen split.")
    source_root = Path(data_cfg["root"])
    if not source_root.is_absolute():
        source_root = PROJECT_ROOT / source_root
    source_root = source_root.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("The frozen folder must not be inside the source dataset.")
    class_names = list(data_cfg["class_names"])
    fraction = float(data_cfg["validation_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("data.validation_fraction must be between zero and one.")
    max_distance = int(config.get("experiments", {}).get("near_duplicate_phash_distance", 4))
    if not 0 <= max_distance <= 63:
        raise ValueError("near_duplicate_phash_distance must be between 0 and 63.")

    records, seen_hashes = [], set()
    for class_name in class_names:
        class_dir = (source_root / class_name).resolve()
        if class_dir.parent != source_root or not class_dir.is_dir():
            raise ValueError(f"Missing direct class folder: {class_dir}")
        for path in trainer.image_files(class_dir):
            path = path.resolve()
            if not path.is_relative_to(class_dir):
                raise ValueError(f"Image escapes its class folder: {path}")
            digest = trainer.file_sha256(path)
            if digest in seen_hashes:
                raise ValueError(f"Byte-identical image appears more than once: {path}")
            seen_hashes.add(digest)
            pixel_hash, phash = _pixel_and_perceptual_hash(path)
            records.append({"path": path.relative_to(source_root).as_posix(),
                            "class_name": class_name, "sha256": digest,
                            "pixel_sha256": pixel_hash, "phash": phash})
    counts = _group_and_assign(records, class_names, fraction,
                               int(config["training"].get("seed", 42)), max_distance)
    records.sort(key=lambda record: record["sha256"])

    staging = output_root.with_name(output_root.name + ".incomplete")
    staging.mkdir(parents=True)
    for record in records:
        source = source_root / record["path"]
        if trainer.file_sha256(source) != record["sha256"]:
            raise ValueError(f"Source image changed while freezing: {source}")
        destination = staging / record["split"] / record["path"]
        if destination.exists():
            raise FileExistsError(f"Split destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if trainer.file_sha256(destination) != record["sha256"]:
            raise ValueError(f"Copied image does not match source: {destination}")
    # A relabel during copying must not silently produce a mixed-time snapshot.
    for class_name in class_names:
        actual = {path.relative_to(source_root).as_posix(): trainer.file_sha256(path)
                  for path in trainer.image_files(source_root / class_name)}
        expected = {r["path"]: r["sha256"] for r in records if r["class_name"] == class_name}
        if actual != expected:
            raise ValueError("Source dataset changed while freezing. The incomplete folder is preserved; retry with a new output name.")

    manifest = {"version": 1, "data_root": str(source_root), "class_names": class_names,
                "seed": int(config["training"].get("seed", 42)),
                "validation_fraction": fraction, "near_duplicate_phash_distance": max_distance,
                "created_at": datetime.now().astimezone().isoformat(), "counts": counts,
                "images": records}
    (staging / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    frozen_config = dict(config)
    frozen_config["data"] = dict(data_cfg)
    frozen_config["data"]["split_root"] = str(output_root)
    frozen_config["data"]["split_manifest"] = str(output_root / "split_manifest.json")
    frozen_config["paths"] = dict(config["paths"])
    frozen_config["paths"]["checkpoint_dir"] = str(PROJECT_ROOT / "checkpoints" / output_root.name)
    frozen_config["evaluation"] = dict(config["evaluation"])
    frozen_config["evaluation"]["output_dir"] = str(PROJECT_ROOT / "outputs" / output_root.name / "evaluation")
    frozen_config.pop("relabel_web", None)  # Continue relabeling from the original config, not this snapshot.
    (staging / "train_config.yaml").write_text(
        yaml.safe_dump(frozen_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    staging.rename(output_root)
    train, val, verified_counts = trainer.split_samples(frozen_config)
    if verified_counts != {name: {"total_unique": row["total"], "train": row["train"], "val": row["val"]}
                           for name, row in counts.items()}:
        raise RuntimeError("Frozen split counts failed verification.")
    print(f"Frozen split: {len(train)} train / {len(val)} val", flush=True)
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    print(f"Folder: {output_root}", flush=True)
    print(f"Train config: {output_root / 'train_config.yaml'}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze four-class train/val image folders without moving source files.")
    parser.add_argument("--config", type=Path, default=trainer.DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "dataset" /
                        f"fixed_4class_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    prepare(args.config, args.output)


if __name__ == "__main__":
    main()
