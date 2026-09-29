"""Relabel operations use temporary images, never the real face dataset."""

import json
from pathlib import Path
import tempfile
import threading
import unittest

from relabel_web import RelabelService


class RelabelWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="face-relabel-test-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.service = RelabelService.__new__(RelabelService)
        self.service.data_root = root / "class"
        self.service.class_names = ["clear_side_face", "occluded_pose"]
        self.service.move_immediately = True
        self.service.quarantine_dir = self.service.data_root / "_relabel_deleted"
        self.service.manifest_path = root / "review.json"
        self.service.lock = threading.RLock()
        self.service.session = {"audit": []}
        source = self.service.data_root / "clear_side_face" / "face.jpg"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"synthetic-image")
        self.service.items = [{
            "id": 0, "candidate_id": "sample", "split": "val",
            "original_label": "clear_side_face", "current_label": "clear_side_face",
            "original_path": "clear_side_face/face.jpg",
            "current_path": "clear_side_face/face.jpg",
            "absolute_path": source, "predicted_label": "occluded_pose",
            "confidence": 0.9, "reviewed": False, "deleted": False,
        }]

    def test_relabel_quarantine_restore_updates_actual_files_and_manifest(self):
        service = self.service
        old_path = service.items[0]["absolute_path"]
        service.set_label(0, "occluded_pose")
        moved = service.data_root / "occluded_pose" / "face.jpg"
        self.assertFalse(old_path.exists())
        self.assertEqual(moved.read_bytes(), b"synthetic-image")
        self.assertEqual(service.item_result(0)["path"], "occluded_pose/face.jpg")
        service.delete_candidate(0)
        quarantined = service.quarantine_dir / "occluded_pose" / "face.jpg"
        self.assertFalse(moved.exists())
        self.assertEqual(quarantined.read_bytes(), b"synthetic-image")
        self.assertTrue(service.item_result(0)["deleted"])
        service.restore_candidate(0)
        self.assertFalse(quarantined.exists())
        self.assertEqual(moved.read_bytes(), b"synthetic-image")
        self.assertFalse(service.item_result(0)["deleted"])
        manifest = json.loads(service.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual([event["action"] for event in manifest["audit"]],
                         ["label", "quarantine", "restore"])
        self.assertEqual(manifest["candidates"][0]["current_path"], "occluded_pose/face.jpg")

    def test_existing_target_is_not_overwritten(self):
        service = self.service
        target = service.data_root / "occluded_pose" / "face.jpg"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"another-image")
        with self.assertRaises(FileExistsError):
            service.set_label(0, "occluded_pose")
        self.assertEqual(target.read_bytes(), b"another-image")
        self.assertTrue(service.items[0]["absolute_path"].exists())
        self.assertEqual(service.session["audit"], [])


if __name__ == "__main__":
    unittest.main()
