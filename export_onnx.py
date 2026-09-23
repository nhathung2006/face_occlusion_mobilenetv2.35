from __future__ import annotations

import argparse
from pathlib import Path
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

import torch

from src.utils.config import load_config
from src.utils.model import build_model, count_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint or cfg["checkpoint"]["best_path"])
    onnx_path = Path(cfg["export"]["onnx_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).cpu().eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    image_size = int(cfg["data"]["image_size"])
    dummy = torch.randn(1, 3, image_size, image_size)

    dynamic_axes = None
    if bool(cfg["export"]["dynamic_batch"]):
        dynamic_axes = {
            "images": {0: "batch"},
            "logits": {0: "batch"},
        }

    kwargs = dict(
        input_names=["images"],
        output_names=["logits"],
        opset_version=int(cfg["export"]["opset"]),
        dynamic_axes=dynamic_axes,
    )

    try:
        torch.onnx.export(model, dummy, onnx_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, onnx_path, **kwargs)

    print(f"Exported: {onnx_path}")
    print(f"Parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    main()
