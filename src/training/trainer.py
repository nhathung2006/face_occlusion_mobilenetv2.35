from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.utils.training import save_checkpoint


def run_epoch(model, loader, criterion, device, num_classes, optimizer=None, lr_scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true, y_pred = [], []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)

            if training:
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

        total_loss += loss.item() * images.size(0)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    return total_loss / len(loader.dataset), metrics


def save_history(history, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    epochs = [h["epoch"] for h in history]

    plots = [
        ("loss", "Loss", ["train_loss", "val_loss"]),
        ("accuracy", "Accuracy", ["train_accuracy", "val_accuracy"]),
        ("f1", "Macro F1", ["train_f1", "val_f1"]),
    ]
    for name, ylabel, fields in plots:
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, [h[fields[0]] for h in history], label="train")
        plt.plot(epochs, [h[fields[1]] for h in history], label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.png", dpi=150)
        plt.close()
