from __future__ import annotations

import argparse
import sys

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

from src.inference.classifier import ONNXFaceOcclusionClassifier
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run single-image ONNX inference")
    parser.add_argument("image")
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    onnx_path = (
        args.onnx
        or cfg.get("export", {}).get("onnx_path")
        or cfg.get("paths", {}).get("onnx_path")
    )
    if not onnx_path:
        raise KeyError("ONNX path is not configured in export.onnx_path or paths.onnx_path.")
    class_names = cfg.get("task", {}).get("binary_class_names", cfg["data"]["class_names"])
    classifier = ONNXFaceOcclusionClassifier(
        onnx_path=onnx_path,
        image_size=int(cfg["data"]["image_size"]),
        class_names=class_names,
        occluded_threshold=float(cfg.get("inference", {}).get("occluded_threshold", 0.5)),
    )
    result = classifier.predict(args.image)
    print(f"class: {result['class_name']}")
    print(f"confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
