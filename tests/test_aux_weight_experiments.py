"""Small synthetic-data checks; never touch the real face dataset."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image, PngImagePlugin
import torch
import yaml

import run_aux_weight_experiments as experiment
import train_4class_single_logit as trainer
from src.utils.training import build_scheduler


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='face-aux-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.config = trainer.load_config(trainer.DEFAULT_CONFIG)
        self.root = self.base / 'data'
        self.config['data']['root'] = str(self.root)
        self.config['data'].pop('split_manifest', None)
        rng = np.random.default_rng(123)
        for name in self.config['data']['class_names']:
            folder = self.root / name
            folder.mkdir(parents=True)
            for i in range(10):
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / f'{i:02}.png')
        self.config_path = self.base / 'config.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config), encoding='utf-8')
        self.run_dir = self.base / 'run'

    def prepare(self, apply):
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.prepare(self.config_path, self.run_dir, apply)

    def test_recoverable_byte_and_pixel_dedup_and_fixed_split(self):
        folder = self.root / self.config['data']['class_names'][0]
        original = folder / '00.png'
        shutil.copy2(original, folder / 'byte-copy.png')
        info = PngImagePlugin.PngInfo()
        info.add_text('note', 'Same pixels, different encoded bytes')
        with Image.open(original) as image:
            image.save(folder / 'pixel-copy.png', pnginfo=info)
        self.prepare(False)
        self.assertEqual(len(trainer.image_files(folder)), 12)
        self.prepare(True)
        audit = json.loads((self.run_dir / 'dedup_audit.json').read_text(encoding='utf-8'))
        self.assertTrue(audit['applied'])
        self.assertEqual({r['match'] for r in audit['duplicates']}, {'bytes', 'decoded_pixels'})
        for record in audit['duplicates']:
            self.assertFalse((self.root / record['path']).exists())
            self.assertEqual(trainer.file_sha256(Path(record['quarantine_path'])), record['sha256'])
        cfg = json.loads((self.run_dir / 'base_config.json').read_text(encoding='utf-8'))
        train, val, counts = trainer.split_samples(cfg)
        self.assertEqual((len(train), len(val)), (32, 8))
        self.assertEqual(trainer.split_samples(cfg), (train, val, counts))
        self.assertTrue({trainer.file_sha256(p) for p, _ in train}.isdisjoint(
            {trainer.file_sha256(p) for p, _ in val}))
        saved_audit = (self.run_dir / 'dedup_audit.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare(True)
        self.assertEqual(saved_audit, (self.run_dir / 'dedup_audit.json').read_bytes())
        (folder / '01.png').rename(folder / 'renamed.png')
        self.assertEqual(len(trainer.split_samples(cfg)[0]), 32)
        shutil.copy2(original, folder / 'new-duplicate.png')
        with self.assertRaisesRegex(ValueError, 'Duplicate file'):
            trainer.split_samples(cfg)

    def test_conflicting_labels_stop_before_any_moves(self):
        names = self.config['data']['class_names']
        shutil.copy2(self.root / names[0] / '00.png', self.root / names[1] / 'conflict.png')
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            self.prepare(True)
        self.assertEqual(len(trainer.image_files(self.root)), 41)
        self.assertFalse((self.root / '_dedup_quarantine').exists())

    def test_fixed_split_rejects_similarity_group_leakage(self):
        self.prepare(True)
        path = self.run_dir / 'split_manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        train_record = next(r for r in manifest['images'] if r['split'] == 'train')
        val_record = next(r for r in manifest['images'] if r['split'] == 'val')
        val_record['group_id'] = train_record['group_id']
        experiment.save_json(path, manifest)
        cfg = json.loads((self.run_dir / 'base_config.json').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ValueError, 'both train and val'):
            trainer.split_samples(cfg)

    def test_f1_scheduler_reduces_and_switches(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.004)
        scheduler, kind = build_scheduler(optimizer, self.config, steps_per_epoch=1, total_epochs=102)
        self.assertEqual(kind, 'plateau_to_cosine')
        self.assertEqual(scheduler.mode, 'max')
        scheduler.step(0.8)
        scheduler.step(0.9)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.004)
        for _ in range(4):
            scheduler.step(0.89)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.002)
        for epoch in range(7, 31):
            scheduler.step(0.90 + epoch * 0.001)
        self.assertEqual(scheduler.phase, 'cosine')
        self.assertEqual(scheduler.cosine_start_epoch, 30)

if __name__ == '__main__':
    unittest.main()
