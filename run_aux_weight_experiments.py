"""Prepare recoverable deduplication and run matched auxiliary-loss experiments."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.fft import dctn

import train_4class_single_logit as trainer

ROOT = Path(__file__).resolve().parent


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def resolved(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def prepare(config_path, run_dir, apply):
    if (run_dir / 'split_manifest.json').exists():
        raise FileExistsError('This experiment was already prepared. Use run mode or a new run directory.')
    cfg = trainer.load_config(config_path)
    root = resolved(cfg['data']['root'])
    names = cfg['data']['class_names']
    for name in names:
        folder = (root / name).resolve()
        if folder.parent != root or not folder.is_dir():
            raise ValueError(f'Class must be an existing direct child of data.root: {name}')
    if not 0 < float(cfg['data']['validation_fraction']) < 1:
        raise ValueError('validation_fraction must be between zero and one.')
    records, copies, pixels, conflicts = [], [], {}, []
    raw_counts = {}
    for name in names:
        files = trainer.image_files(root / name)
        raw_counts[name] = len(files)
        for path in files:
            path = path.resolve()
            if not path.is_relative_to(root / name):
                raise ValueError(f'Image escapes its class folder: {path}')
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + b'\0' + rgb.tobytes()).hexdigest()
                gray = np.asarray(rgb.convert('L').resize((32, 32)), dtype=float)
            record = {'path': path.relative_to(root).as_posix(), 'class_name': name,
                      'sha256': trainer.file_sha256(path), 'pixel_sha256': pixel_hash}
            if pixel_hash in pixels:
                keep = pixels[pixel_hash]
                if keep['class_name'] != name:
                    conflicts.append({'keep': keep, 'other': record})
                else:
                    copies.append({**record, 'keep_path': keep['path'],
                                   'match': 'bytes' if record['sha256'] == keep['sha256'] else 'decoded_pixels'})
                continue
            pixels[pixel_hash] = record
            dct = dctn(gray, norm='ortho')[:8, :8].flatten()
            bits = dct > np.median(dct[1:]); bits[0] = False
            record['phash'] = int.from_bytes(np.packbits(bits).tobytes(), 'big')
            records.append(record)

    # Conservative similarity grouping prevents near-copies crossing train/val.
    # Only identical decoded RGB pixels are candidates for physical removal.
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    near_pairs = []
    max_distance = int(cfg['experiments']['near_duplicate_phash_distance'])
    if not 0 <= max_distance <= 63:
        raise ValueError('near_duplicate_phash_distance must be in [0, 63].')
    for i, record in enumerate(records):
        for j in range(i):
            distance = (record['phash'] ^ records[j]['phash']).bit_count()
            if distance <= max_distance:
                parent[find(i)] = find(j)
                near_pairs.append({'a': record['path'], 'b': records[j]['path'], 'distance': distance})
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record)
    group_list = list(groups.values())
    random.Random(int(cfg['training']['seed'])).shuffle(group_list)
    totals = Counter(r['class_name'] for r in records)
    targets = {n: round(totals[n] * cfg['data']['validation_fraction']) for n in names}
    val_counts = Counter()
    for group in group_list:
        group_counts = Counter(r['class_name'] for r in group)
        before = sum(abs(targets[n] - val_counts[n]) for n in names)
        after = sum(abs(targets[n] - val_counts[n] - group_counts[n]) for n in names)
        split = 'val' if after < before else 'train'
        if split == 'val':
            val_counts.update(group_counts)
        group_id = min(r['sha256'] for r in group)
        for r in group:
            r['split'] = split
            r['group_id'] = group_id
            r.pop('phash', None)
    # Stable order and seeded batch sampling are identical for every trial.
    records.sort(key=lambda r: r['sha256'])
    counts = {n: {'train': totals[n] - val_counts[n], 'val': val_counts[n], 'total': totals[n]} for n in names}
    if any(row['train'] == 0 or row['val'] == 0 for row in counts.values()):
        raise ValueError('A class has no train/val samples after grouping.')
    quarantine = (root / '_dedup_quarantine' / run_dir.name).resolve()
    if not quarantine.is_relative_to(root) or quarantine == root:
        raise ValueError('Invalid quarantine directory.')
    for record in copies:
        record['quarantine_path'] = str(quarantine / record['path'])
    audit = {'data_root': str(root), 'raw_counts': raw_counts, 'retained_counts': counts,
             'duplicates': copies, 'conflicting_pixel_labels': conflicts, 'near_pairs': near_pairs,
             'quarantine': str(quarantine), 'applied': False}
    save_json(run_dir / 'dedup_audit.json', audit)
    print(json.dumps({'raw_counts': raw_counts, 'duplicate_files': len(copies),
                      'duplicate_types': dict(Counter(r['match'] for r in copies)),
                      'conflicting_labels': len(conflicts), 'near_pairs': len(near_pairs),
                      'retained_split': counts}, ensure_ascii=False), flush=True)
    if conflicts:
        raise ValueError('Identical pixels have conflicting labels; review dedup_audit.json before moving files.')
    if not apply:
        return
    moves = []
    try:
        for record in copies:
            source = (root / record['path']).resolve()
            dest = Path(record['quarantine_path']).resolve()
            if not source.is_relative_to(root / record['class_name']) or not dest.is_relative_to(quarantine):
                raise ValueError('Deduplication target escapes the configured folders.')
            if trainer.file_sha256(source) != record['sha256'] or dest.exists():
                raise ValueError(f'File changed or quarantine target exists: {source}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            source.rename(dest)
            moves.append((source, dest))
        audit['applied'] = True
        save_json(run_dir / 'dedup_audit.json', audit)
    except Exception:
        for source, dest in reversed(moves):
            if source.exists():
                raise RuntimeError(f'Rollback destination unexpectedly exists: {source}')
            dest.rename(source)
        raise
    manifest = {'version': 1, 'data_root': str(root), 'class_names': names,
                'seed': cfg['training']['seed'], 'validation_fraction': cfg['data']['validation_fraction'],
                'near_duplicate_phash_distance': max_distance, 'images': records}
    save_json(run_dir / 'split_manifest.json', manifest)
    cfg['data']['split_manifest'] = str(run_dir / 'split_manifest.json')
    save_json(run_dir / 'base_config.json', cfg)
    # Assert no exact or similarity-group overlap across the two splits.
    train, val, _ = trainer.split_samples(cfg)
    assert {r['group_id'] for r in records if r['split'] == 'train'}.isdisjoint(
        {r['group_id'] for r in records if r['split'] == 'val'})
    print(f'Prepared {len(train)} train / {len(val)} val; recoverable duplicate moves: {len(moves)}', flush=True)


def clean_train_metrics(cfg, checkpoint):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = trainer.build_model(cfg, load_pretrained=False)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    train, _, _ = trainer.split_samples(cfg)
    _, tf = trainer.make_transforms(cfg)
    mapping = torch.tensor(trainer.subclass_to_binary_ids(cfg))
    truth, predictions = [], []
    with torch.inference_mode():
        for x, y in trainer.DataLoader(trainer.FaceClassDataset(train, tf), batch_size=64, num_workers=0):
            z, _ = model(x)
            truth.extend(mapping[y].tolist())
            predictions.extend((z.squeeze(1).sigmoid() >= cfg['inference']['occluded_threshold']).long().tolist())
    acc, f1 = trainer.macro_f1(truth, predictions, 2)
    return {'accuracy': acc, 'macro_f1': f1}


def run(run_dir):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to silently run long experiments on CPU.')
    cfg = json.loads((run_dir / 'base_config.json').read_text(encoding='utf-8'))
    trainer.split_samples(cfg)
    results = []
    for weight in cfg['experiments']['auxiliary_loss_weights']:
        trial = run_dir / f'aux_{weight:.2f}'.replace('.', '_')
        trial.mkdir(parents=True, exist_ok=True)
        c = copy.deepcopy(cfg)
        c['training']['auxiliary_loss_weight'] = float(weight)
        c['runtime']['train_device'] = 'cuda'
        c['paths']['checkpoint_dir'] = str(trial / 'checkpoints')
        c['evaluation']['output_dir'] = str(trial / 'evaluation')
        c['evaluation']['auto_evaluate_after_training'] = True
        config_path = trial / 'config.yaml'
        if (trial / 'result.json').exists():
            saved_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            if saved_config != c:
                raise ValueError(f'Completed trial settings differ from this experiment: {trial}')
        if not (trial / 'result.json').exists():
            if (trial / 'checkpoints').exists():
                raise FileExistsError(f'Incomplete trial present; preserve it and start a new experiment: {trial}')
            config_path.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding='utf-8')
            print(f'START auxiliary_loss_weight={weight:.2f}', flush=True)
            start = time.monotonic()
            with (trial / 'train.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'train_4class_single_logit.py'),
                                            '--config', str(config_path), '--mode', 'train'],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        time.sleep(20)
                        lines = (trial / 'train.log').read_text(encoding='utf-8', errors='replace').splitlines()
                        if lines:
                            print(f'[{weight:.2f}] {lines[-1]}', flush=True)
                except BaseException:
                    process.terminate(); process.wait()
                    raise
            if process.returncode:
                raise RuntimeError(f'Training failed: see {trial / "train.log"}')
            latest = json.loads((trial / 'evaluation' / 'latest_evaluation.json').read_text(encoding='utf-8'))
            report = json.loads(Path(latest['summary']).read_text(encoding='utf-8'))
            checkpoint = trial / 'checkpoints' / c['paths']['best_checkpoint_name']
            train_metrics = clean_train_metrics(c, checkpoint)
            row = {'auxiliary_loss_weight': weight, 'epoch': report['checkpoint_epoch'],
                   'checkpoint': str(checkpoint), 'config': str(config_path),
                   'val_binary': report['binary_2class'], 'train_binary': train_metrics,
                   'accuracy_gap': train_metrics['accuracy'] - report['binary_2class']['accuracy'],
                   'val_subtype': report['subtype_4class'] if weight > 0 else None,
                   'elapsed_seconds': round(time.monotonic() - start, 1)}
            save_json(trial / 'result.json', row)
        row = json.loads((trial / 'result.json').read_text(encoding='utf-8'))
        results.append(row)
        save_json(run_dir / 'comparison.json', {'results': results})
        print(f'FINISHED weight={weight:.2f}, val F1={row["val_binary"]["macro_f1"]:.6f}', flush=True)
    results.sort(key=lambda r: (r['val_binary']['macro_f1'], r['val_binary']['accuracy']), reverse=True)
    save_json(run_dir / 'comparison.json', {'selection': 'binary validation macro-F1, then accuracy',
                                          'seed': cfg['training']['seed'], 'winner': results[0], 'results': results})
    print(f'WINNER: auxiliary_loss_weight={results[0]["auxiliary_loss_weight"]:.2f}', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=trainer.DEFAULT_CONFIG)
    parser.add_argument('--mode', choices=['prepare', 'run'], required=True)
    parser.add_argument('--run-dir', type=Path, help='Existing run for run mode; optional new directory for prepare.')
    parser.add_argument('--apply-dedup', action='store_true')
    args = parser.parse_args()
    if args.run_dir is None:
        if args.mode == 'run':
            parser.error('--run-dir is required for run mode.')
        cfg = trainer.load_config(args.config.resolve())
        run_dir = resolved(cfg['experiments']['output_root']) / datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        run_dir = args.run_dir.resolve()
    print(f'Experiment directory: {run_dir}', flush=True)
    if args.mode == 'prepare':
        prepare(args.config.resolve(), run_dir, args.apply_dedup)
    else:
        run(run_dir)


if __name__ == '__main__':
    main()
