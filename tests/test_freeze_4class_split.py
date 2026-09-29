"""Check that a frozen split is physically separate and remains fixed."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import yaml

import train_4class_single_logit as trainer
from tools.freeze_4class_split import prepare


class FrozenSplitTests(unittest.TestCase):
    def test_snapshot_is_disjoint_and_detects_later_changes(self):
        with tempfile.TemporaryDirectory(prefix="face-fixed-split-test-") as temp:
            base = Path(temp)
            source = base / "source"
            config = trainer.load_config(trainer.DEFAULT_CONFIG)
            config["data"]["root"] = str(source)
            config["data"]["validation_fraction"] = 0.2
            config["experiments"]["near_duplicate_phash_distance"] = 0
            rng = np.random.default_rng(1234)
            for class_name in config["data"]["class_names"]:
                folder = source / class_name
                folder.mkdir(parents=True)
                for index in range(8):
                    image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                    Image.fromarray(image).save(folder / f"{class_name}_{index}.png")
            config_path = base / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            output = base / "fixed"
            prepare(config_path, output)
            frozen_config = trainer.load_config(output / "train_config.yaml")
            train, val, counts = trainer.split_samples(frozen_config)
            self.assertEqual(len(train) + len(val), 32)
            self.assertEqual(len(trainer.image_files(source)), 32)
            self.assertTrue(all(path.is_relative_to(output / "train") for path, _ in train))
            self.assertTrue(all(path.is_relative_to(output / "val") for path, _ in val))
            self.assertTrue({path for path, _ in train}.isdisjoint(path for path, _ in val))
            self.assertTrue(all(counts[name]["val"] > 0 for name in config["data"]["class_names"]))
            with self.assertRaises(FileExistsError):
                prepare(config_path, output)
            first = train[0][0]
            first.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Fixed split image changed"):
                trainer.split_samples(frozen_config)


if __name__ == "__main__":
    unittest.main()
