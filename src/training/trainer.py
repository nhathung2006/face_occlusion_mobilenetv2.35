from __future__ import annotations

import csv
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics
from src.utils.training import save_checkpoint


class EarlyStopping:
    """
    Early Stopping callback with model weight restoration, patience tracking,
    and overfitting diagnostics.

    Parameters:
        monitor: Metric to observe (e.g. 'val_loss', 'val_f1', 'val_accuracy').
        mode: 'min' (for loss) or 'max' (for accuracy/f1).
        patience: Number of epochs to wait without improvement before stopping.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        restore_best_weights: Whether to restore model weights from the epoch with the best value.
        enabled: Whether early stopping is active.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 8,
        min_delta: float = 0.0005,
        restore_best_weights: bool = True,
        enabled: bool = True,
    ):
        self.monitor = str(monitor).strip()
        self.mode = str(mode).strip().lower()
        if self.mode not in ["min", "max"]:
            self.mode = "min" if "loss" in self.monitor.lower() else "max"
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.restore_best_weights = bool(restore_best_weights)
        self.enabled = bool(enabled)

        self.patience_counter = 0
        self.best_epoch: int | None = None
        self.best_metric = float("inf") if self.mode == "min" else float("-inf")
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_row: dict | None = None
        self.stopped_epoch: int | None = None

    def is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return (self.best_metric - current) > self.min_delta
        return (current - self.best_metric) > self.min_delta

    def step(self, epoch: int, row: dict, model: torch.nn.Module) -> bool:
        """
        Evaluate current metrics at end of epoch.
        Returns True if early stopping triggers, False otherwise.
        """
        if self.monitor not in row:
            raise KeyError(f"Monitored metric '{self.monitor}' not found in epoch summary row: {list(row.keys())}")

        current = float(row[self.monitor])
        improved = self.is_improvement(current)

        if improved:
            self.best_metric = current
            self.best_epoch = epoch
            self.best_row = row
            self.patience_counter = 0
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.patience_counter += 1

        if self.enabled and self.patience_counter >= self.patience:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_state_dict is not None:
                self.restore(model)
                # Reset patience counter to 0 after restoration
                self.patience_counter = 0
            return True

        return False

    def restore(self, model: torch.nn.Module) -> None:
        """Restore the best model weights into model and reset patience."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            self.patience_counter = 0

    def reset_patience(self) -> None:
        """Reset the patience counter (e.g., when transitioning between training stages)."""
        self.patience_counter = 0

    def get_status_str(self, current_row: dict | None = None) -> str:
        """Returns concise diagnostic status including overfitting indicators."""
        best_str = f"{self.best_metric:.4f}" if self.best_epoch is not None else "N/A"
        epoch_str = f"Epoch {self.best_epoch:03d}" if self.best_epoch is not None else "N/A"
        status = f"Patience: {self.patience_counter}/{self.patience} | Best {self.monitor}: {best_str} ({epoch_str})"

        if current_row is not None:
            train_loss = current_row.get("train_loss")
            val_loss = current_row.get("val_loss")
            train_acc = current_row.get("train_accuracy")
            val_acc = current_row.get("val_accuracy")
            if train_loss is not None and val_loss is not None:
                loss_gap = float(val_loss) - float(train_loss)
                status += f" | Loss Gap: {loss_gap:+.4f}"
            if train_acc is not None and val_acc is not None:
                acc_gap = float(train_acc) - float(val_acc)
                status += f", Acc Gap: {acc_gap:+.4f}"
        return status


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

    def save_plot_safely(fig, output_path: Path) -> None:
        """Save plots without failing training when Windows locks an old PNG."""
        output_path = Path(output_path)
        temp_path = output_path.with_name(
            f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}"
        )
        try:
            fig.savefig(temp_path, dpi=150)
            for attempt in range(5):
                try:
                    os.replace(temp_path, output_path)
                    return
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(0.5 * (attempt + 1))
        except OSError as exc:
            fallback_path = output_path.with_name(
                f"{output_path.stem}_latest_{os.getpid()}{output_path.suffix}"
            )
            try:
                fig.savefig(fallback_path, dpi=150)
                print(
                    f"Warning: could not replace {output_path}; "
                    f"saved plot to {fallback_path} ({exc})"
                )
            except OSError as fallback_exc:
                print(
                    f"Warning: could not save plot {output_path}; "
                    f"continuing without plot ({fallback_exc})"
                )
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    for name, ylabel, fields in plots:
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, [h[fields[0]] for h in history], label="train")
        plt.plot(epochs, [h[fields[1]] for h in history], label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        figure = plt.gcf()
        save_plot_safely(figure, out_dir / f"{name}.png")
        plt.close()
