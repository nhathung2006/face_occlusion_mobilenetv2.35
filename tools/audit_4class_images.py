"""Read-only image audit for the active four-class training split.

Creates review sheets and CSVs under outputs; never moves source images.
Run: .venv/Scripts/python.exe tools/audit_4class_images.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.fft import dctn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_4class_single_logit as trainer  # noqa: E402

OUT = ROOT / "outputs" / "dataset_review_20260929"
CHECKPOINT = ROOT / "checkpoints" / "4class_config_yaml" / "best.pth"
NEAR_DISTANCE = 4


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def short_name(path: Path, limit: int = 37) -> str:
    name = path.name
    return name if len(name) <= limit else name[:limit - 3] + "..."


def draw_sheet(records: list[dict], path: Path, columns: int, tile: tuple[int, int],
               caption_lines: int = 2) -> None:
    width, height = tile
    caption_height = 18 * caption_lines + 10
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, record in enumerate(records):
        x = (i % columns) * width
        y = (i // columns) * (height + caption_height)
        source_path = Path(record["path"])
        try:
            with Image.open(source_path) as original:
                thumb = ImageOps.exif_transpose(original).convert("RGB")
                thumb.thumbnail((width - 8, height - 8))
                sheet.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))
        except Exception as exc:
            draw.text((x + 4, y + 4), f"READ ERROR: {exc}", fill="red", font=font)
        for line, text in enumerate(record["caption"]):
            draw.text((x + 4, y + height + 3 + 18 * line), text, fill="black", font=font)
    sheet.save(path, quality=93, subsampling=0)


def paginate(records: list[dict], name: str, columns: int = 7,
             per_page: int = 49, tile: tuple[int, int] = (190, 175)) -> list[str]:
    names = []
    for start in range(0, len(records), per_page):
        path = OUT / f"{name}_{start // per_page + 1:02d}.jpg"
        draw_sheet(records[start:start + per_page], path, columns, tile)
        names.append(path.name)
    return names


def side_probabilities(config: dict, side_samples: list[tuple[Path, int, str]]) -> list[dict]:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = trainer.build_model(config, load_pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    _, eval_transform = trainer.make_transforms(config)
    dataset = trainer.FaceClassDataset([(path, label) for path, label, _ in side_samples], eval_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    outputs = []
    with torch.inference_mode():
        for images, _ in loader:
            logits, aux_logits = model(images)
            probabilities = aux_logits.softmax(dim=-1)
            for binary_logit, aux_prob in zip(logits.squeeze(1), probabilities):
                outputs.append((float(binary_logit), [float(x) for x in aux_prob]))
    rows = []
    for (path, _, split), (logit, probabilities) in zip(side_samples, outputs, strict=True):
        rows.append({
            "split": split,
            "path": str(path),
            "filename": path.name,
            "pose_probability": probabilities[3],
            "side_probability": probabilities[1],
            "aux_prediction": config["data"]["class_names"][max(range(4), key=probabilities.__getitem__)],
            "binary_logit": logit,
            "binary_probability_occluded": float(torch.sigmoid(torch.tensor(logit))),
        })
    return rows


def image_fingerprints(config: dict, split_by_path: dict[Path, str]) -> tuple[list[dict], list[dict], list[dict]]:
    root = Path(config["data"]["root"])
    records = []
    for class_name in config["data"]["class_names"]:
        for path in trainer.image_files(root / class_name):
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                pixels = hashlib.sha256(str(rgb.size).encode() + b"\0" + rgb.tobytes()).hexdigest()
                gray = np.asarray(rgb.convert("L").resize((32, 32)), dtype=float)
            dct = dctn(gray, norm="ortho")[:8, :8].flatten()
            bits = dct > np.median(dct[1:])
            bits[0] = False
            records.append({
                "path": str(path), "class_name": class_name,
                "split": split_by_path.get(path.resolve(), "not_in_active_split"),
                "sha256": trainer.file_sha256(path), "pixel_sha256": pixels,
                "phash": int.from_bytes(np.packbits(bits).tobytes(), "big"),
            })

    byte_groups, pixel_groups = {}, {}
    for rec in records:
        byte_groups.setdefault(rec["sha256"], []).append(rec)
        pixel_groups.setdefault(rec["pixel_sha256"], []).append(rec)
    exact = []
    for match_type, groups in (("same_bytes", byte_groups), ("same_decoded_pixels", pixel_groups)):
        for group in groups.values():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i):
                        exact.append({"match_type": match_type, "a": group[j], "b": group[i]})

    near = []
    for i, record in enumerate(records):
        for other in records[:i]:
            distance = (record["phash"] ^ other["phash"]).bit_count()
            if distance <= NEAR_DISTANCE:
                near.append({"a": other, "b": record, "distance": distance,
                             "cross_split": other["split"] != record["split"],
                             "cross_class": other["class_name"] != record["class_name"]})
    return records, exact, near


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = trainer.load_config(trainer.DEFAULT_CONFIG)
    train, val, counts = trainer.split_samples(config)
    split_by_path = {path.resolve(): split for split, samples in (("train", train), ("val", val))
                     for path, _ in samples}
    side_label = config["data"]["class_names"].index("clear_side_face")
    side_samples = [(path, label, split) for split, samples in (("train", train), ("val", val))
                    for path, label in samples if label == side_label]
    side = side_probabilities(config, side_samples)
    side.sort(key=lambda rec: (-rec["pose_probability"], rec["filename"].casefold()))
    side_columns = ["split", "path", "filename", "pose_probability", "side_probability",
                    "aux_prediction", "binary_logit", "binary_probability_occluded"]
    save_csv(OUT / "clear_side_all.csv", side, side_columns)
    side_pages = {}
    candidate_pages = {}
    candidates = [rec for rec in side if rec["pose_probability"] >= 0.20]
    save_csv(OUT / "clear_side_pose_candidates.csv", candidates, side_columns)
    for split in ("train", "val"):
        split_rows = [r for r in side if r["split"] == split]
        tiles = [{"path": r["path"], "caption": [f"{split}  pose={r['pose_probability']:.2f}",
                  short_name(Path(r["path"]))]} for r in split_rows]
        side_pages[split] = paginate(tiles, f"clear_side_all_{split}")
        candidate_tiles = [tile for tile, rec in zip(tiles, split_rows) if rec in candidates]
        candidate_pages[split] = paginate(candidate_tiles, f"pose_candidates_{split}",
                                          columns=5, per_page=25, tile=(215, 195))

    _, exact, near = image_fingerprints(config, split_by_path)
    pair_rows = []
    for pair in near:
        a, b = pair["a"], pair["b"]
        pair_rows.append({"distance": pair["distance"], "cross_split": pair["cross_split"],
                          "cross_class": pair["cross_class"], "a_split": a["split"],
                          "a_class": a["class_name"], "a_path": a["path"],
                          "b_split": b["split"], "b_class": b["class_name"],
                          "b_path": b["path"]})
    pair_rows.sort(key=lambda p: (not p["cross_split"], p["distance"], p["a_path"]))
    save_csv(OUT / "near_duplicate_pairs.csv", pair_rows,
             ["distance", "cross_split", "cross_class", "a_split", "a_class", "a_path",
              "b_split", "b_class", "b_path"])
    def pair_tiles(pairs: list[dict]) -> list[dict]:
        tiles = []
        for index, pair in enumerate(pairs, start=1):
            for letter in ("a", "b"):
                tiles.append({"path": pair[f"{letter}_path"], "caption": [
                    f"PAIR {index:02d} {letter.upper()} {pair[f'{letter}_split']} d={pair['distance']}",
                    f"{pair[f'{letter}_class']}: {short_name(Path(pair[f'{letter}_path']), 25)}",
                ]})
        return tiles

    duplicate_tiles = pair_tiles(pair_rows)
    duplicate_pages = paginate(duplicate_tiles, "near_duplicate_pairs", columns=4,
                               per_page=24, tile=(245, 210))
    cross_pages = paginate(pair_tiles([p for p in pair_rows if p["cross_split"]]),
                           "cross_split_pairs", columns=4, per_page=24, tile=(245, 210))
    within_pages = paginate(pair_tiles([p for p in pair_rows if not p["cross_split"]]),
                            "within_split_pairs", columns=4, per_page=24, tile=(245, 210))
    exact_rows = [{"match_type": item["match_type"], "a_path": item["a"]["path"],
                   "a_split": item["a"]["split"], "b_path": item["b"]["path"],
                   "b_split": item["b"]["split"]} for item in exact]
    save_csv(OUT / "exact_duplicates.csv", exact_rows,
             ["match_type", "a_path", "a_split", "b_path", "b_split"])
    summary = {
        "checkpoint": str(CHECKPOINT), "active_config": str(trainer.DEFAULT_CONFIG),
        "split_counts": counts, "split_note": "Reconstructed from current files and seed; checkpoint does not store membership.",
        "side_total": len(side), "candidate_rule": "auxiliary P(occluded_pose) >= 0.20; visual review required",
        "side_candidates": {split: sum(r["split"] == split for r in candidates) for split in ("train", "val")},
        "exact_duplicate_pair_records": len(exact), "near_pair_count": len(near),
        "near_cross_split": sum(p["cross_split"] for p in near),
        "near_within_split": sum(not p["cross_split"] for p in near),
        "side_pages": side_pages, "candidate_pages": candidate_pages,
        "near_pair_pages": duplicate_pages,
        "cross_split_pages": cross_pages, "within_split_pages": within_pages,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
