from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.datasets.dataset import build_loaders
from src.utils.config import load_config
from src.utils.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint or cfg["checkpoint"]["best_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = build_loaders(cfg)
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(targets.tolist())

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    Path("outputs/confusion_matrix").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, ax=ax, cmap="Blues")
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix/test_confusion_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
