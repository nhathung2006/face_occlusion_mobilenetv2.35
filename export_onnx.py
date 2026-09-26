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
import torch.nn as nn

from src.utils.config import load_config
from src.utils.model import build_model, count_parameters


class ClampedLogitModel(nn.Module):
    """Export raw logits with a bounded range; sigmoid remains outside the graph."""

    def __init__(self, model: nn.Module, max_logit: float):
        super().__init__()
        self.model = model
        self.max_logit = float(max_logit)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(images)
        return torch.clamp(logits, min=-self.max_logit, max=self.max_logit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint or cfg["checkpoint"]["last_path"])
    onnx_path = Path(cfg["export"]["onnx_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg).cpu().eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    export_model = ClampedLogitModel(
        model,
        max_logit=float(cfg["training"].get("logit_penalty", {}).get("max_logit", 3.8918203)),
    ).cpu().eval()

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
        torch.onnx.export(export_model, dummy, onnx_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(export_model, dummy, onnx_path, **kwargs)

    print(f"Exported: {onnx_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Logit clamp: [-{export_model.max_logit}, {export_model.max_logit}]")
    print(f"Parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    main()
