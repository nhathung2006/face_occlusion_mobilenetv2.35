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
    onnx_path = args.onnx or cfg["export"]["onnx_path"]
    classifier = ONNXFaceOcclusionClassifier(
        onnx_path=onnx_path,
        image_size=int(cfg["data"]["image_size"]),
        class_names=cfg["data"]["class_names"],
    )
    result = classifier.predict(args.image)
    print(f"class: {result['class_name']}")
    print(f"confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
