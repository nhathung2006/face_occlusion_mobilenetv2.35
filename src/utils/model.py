from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
import torch.nn as nn

from src.models.mobilenetv2 import MobileNetV2

PRETRAINED_URLS = {
    0.35: "https://github.com/d-li14/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_0.35-b2e15951.pth",
}


def download_pretrained(dest_path: Path, width_mult: float) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = PRETRAINED_URLS.get(width_mult)
    if not url:
        raise ValueError(f"No default pretrained URL available for width_mult={width_mult}")
    print(f"Downloading pretrained weights from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    print(f"Pretrained weights downloaded successfully to {dest_path}")


def build_model(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    width_mult = float(model_cfg["width_mult"])
    num_classes = int(model_cfg["num_classes"])
    dropout = float(model_cfg["dropout"])

    # Build a 1000-class model first so the upstream 0.35 ImageNet checkpoint
    # can be loaded exactly, then replace the classifier for this 2-class task.
    model = MobileNetV2(num_classes=1000, width_mult=width_mult)

    pretrained_path = Path(model_cfg["pretrained_path"])
    if bool(model_cfg["pretrained"]):
        if not pretrained_path.exists():
            try:
                download_pretrained(pretrained_path, width_mult)
            except Exception as e:
                raise FileNotFoundError(
                    f"Pretrained weight not found at {pretrained_path} and failed to auto-download: {e}. "
                    "Please download mobilenetv2_0.35-b2e15951.pth manually into weights/."
                )
        state = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

    last_channel = model.classifier.in_features
    if dropout > 0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )
    else:
        model.classifier = nn.Linear(last_channel, num_classes)

    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_layer_freezing(model: nn.Module, unfreeze_from_block: int = 11) -> None:
    """
    Freezes all layers from block 0 up to block (unfreeze_from_block - 1).
    Unfreezes blocks from unfreeze_from_block to the end of features,
    along with the final 1x1 conv projection and the classifier head.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        total_layers = len(model.features)
        for i in range(unfreeze_from_block, total_layers):
            for p in model.features[i].parameters():
                p.requires_grad = True

    if hasattr(model, "conv"):
        for p in model.conv.parameters():
            p.requires_grad = True

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True


def get_parameter_groups(model: nn.Module):
    """
    Returns separate parameter lists for:
      - backbone_params: unfrozen layers in features and conv
      - classifier_params: parameters in classifier
    """
    backbone_params = []
    classifier_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            classifier_params.append(p)
        else:
            backbone_params.append(p)

    return backbone_params, classifier_params


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all backbone layers, train only classifier."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier")


def unfreeze_last_n_blocks(model: nn.Module, n_blocks: int = 5) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        total_blocks = len(model.features)
        unfreeze_from = max(0, total_blocks - n_blocks)
        for i in range(unfreeze_from, total_blocks):
            for p in model.features[i].parameters():
                p.requires_grad = True

    if hasattr(model, "conv"):
        for p in model.conv.parameters():
            p.requires_grad = True

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True
