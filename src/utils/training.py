from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def build_optimizer(
    model: nn.Module,
    cfg: dict,
    lr: float | None = None,
    backbone_lr: float | None = None,
    classifier_lr: float | None = None,
):
    t = cfg["training"]
    name = str(t["optimizer"]).lower()
    weight_decay = float(t.get("weight_decay", 0.0001))

    if lr is not None and backbone_lr is None and classifier_lr is None:
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": float(lr),
                "initial_lr": float(lr),
                "name": "all",
            }
        ]
        default_lr = float(lr)
    elif backbone_lr is not None or classifier_lr is not None:
        from src.utils.model import get_parameter_groups

        b_lr = float(backbone_lr if backbone_lr is not None else t.get("backbone_lr", 0.0001))
        c_lr = float(classifier_lr if classifier_lr is not None else t.get("classifier_lr", 0.001))

        backbone_params, classifier_params = get_parameter_groups(model)
        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": b_lr,
                "initial_lr": b_lr,
                "name": "backbone",
            })
        if classifier_params:
            param_groups.append({
                "params": classifier_params,
                "lr": c_lr,
                "initial_lr": c_lr,
                "name": "classifier",
            })
        default_lr = c_lr
    else:
        learning_rate = float(t.get("learning_rate", 0.0001))
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "name": "all",
            }
        ]
        default_lr = learning_rate

    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=default_lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=default_lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=default_lr,
            momentum=float(t.get("momentum", 0.90)),
            nesterov=bool(t.get("nesterov", True)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


class WarmupCosineScheduler:
    """Batch-level cosine scheduler with warmup supporting multiple parameter groups."""

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        warmup_start_factor: float = 0.1,
        eta_min: float = 1e-6,
        base_lr: float | None = None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_epochs * steps_per_epoch))
        self.total_steps = max(1, int(total_epochs * steps_per_epoch))
        self.warmup_start_factor = float(warmup_start_factor)
        self.eta_min = float(eta_min)
        self.step_num = 0

        self.base_lrs = [float(g.get("initial_lr", g["lr"])) for g in optimizer.param_groups]
        if self.warmup_steps > 0:
            for group, b_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = b_lr * self.warmup_start_factor

    def step(self) -> list[float]:
        self.step_num += 1
        if self.warmup_steps > 0 and self.step_num <= self.warmup_steps:
            progress = min(max(self.step_num / self.warmup_steps, 0.0), 1.0)
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            lrs = [b_lr * factor for b_lr in self.base_lrs]
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lrs = [self.eta_min + max(0.0, b_lr - self.eta_min) * cosine for b_lr in self.base_lrs]

        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        return lrs


class WarmupCosinePerGroupScheduler:
    """Epoch-level scheduler that warms up and decays each optimizer parameter group independently."""

    def __init__(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        eta_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.warmup_start_factor = float(warmup_start_factor)
        self.eta_min = float(eta_min)
        self.epoch = 0
        self.base_lrs = [float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups]

        if not 0.0 <= self.warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be between 0 and 1")

        if self.warmup_epochs > 0:
            self._set_lrs(self._warmup_lrs(0))

    def _set_lrs(self, lrs: list[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def _warmup_lrs(self, epoch: int) -> list[float]:
        if self.warmup_epochs <= 0:
            return self.base_lrs
        progress = min(max(epoch / self.warmup_epochs, 0.0), 1.0)
        factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
        return [b_lr * factor for b_lr in self.base_lrs]

    def _cosine_lrs(self, epoch: int) -> list[float]:
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.eta_min + max(0.0, b_lr - self.eta_min) * cosine for b_lr in self.base_lrs]

    def step(self) -> list[float]:
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            lrs = self._warmup_lrs(self.epoch)
        else:
            lrs = self._cosine_lrs(self.epoch)
        self._set_lrs(lrs)
        return lrs


def build_scheduler(
    optimizer,
    cfg: dict,
    steps_per_epoch: int,
    lr: float | None = None,
    total_epochs: int | None = None,
    warmup_epochs: int = 0,
    warmup_start_factor: float = 0.1,
) -> Tuple[Any, str]:
    """
    Builds LR scheduler and returns (scheduler_instance, step_type)
    step_type can be:
      - 'batch': stepped every batch iteration (e.g. WarmupCosineScheduler)
      - 'epoch': stepped once per epoch (e.g. WarmupCosinePerGroupScheduler, StepLR)
      - 'plateau': stepped once per epoch with a metric value (e.g. ReduceLROnPlateau)
    """
    t = cfg["training"]
    name = str(t.get("scheduler", "cosine")).lower()
    epochs = int(total_epochs if total_epochs is not None else t["epochs"])

    if name in ["reduce_on_plateau", "plateau", "reducelronplateau"]:
        p_cfg = t.get("plateau", {})
        mode = str(p_cfg.get("mode", t.get("early_stopping", {}).get("mode", "max"))).lower()
        factor = float(p_cfg.get("factor", 0.5))
        patience = int(p_cfg.get("patience", 3))
        min_lr = float(p_cfg.get("min_lr", 1e-6))
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
        return scheduler, "plateau"

    if name in ["cosine", "cosine_annealing", "cosine_annealing_lr"]:
        c_cfg = t.get("cosine", {})
        eta_min = float(c_cfg.get("eta_min", 1e-6))
        scheduler = WarmupCosinePerGroupScheduler(
            optimizer,
            total_epochs=epochs,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=warmup_start_factor,
            eta_min=eta_min,
        )
        return scheduler, "epoch"

    if name in ["warmup_cosine", "warmup_cosine_scheduler"]:
        w_epochs = warmup_epochs if warmup_epochs > 0 else int(t.get("warmup_epochs", 2))
        c_cfg = t.get("cosine", {})
        eta_min = float(c_cfg.get("eta_min", 1e-6))
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=w_epochs,
            total_epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_start_factor=warmup_start_factor,
            eta_min=eta_min,
        )
        return scheduler, "batch"

    if name in ["step", "steplr"]:
        step_size = int(t.get("step_size", 10))
        gamma = float(t.get("gamma", 0.5))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler, "epoch"

    raise ValueError(f"Unsupported scheduler: {name}")


def save_checkpoint(path, model, optimizer, epoch, best_metric, history, cfg):
    """Save a checkpoint safely on Windows.

    Writing directly to an existing checkpoint can fail with Windows error
    1224 when another process briefly has the destination file mapped/locked.
    Save to a sibling temporary file first, then atomically replace the
    destination with a retry window. If Windows keeps the destination locked,
    retain the successfully written checkpoint under a fallback name instead
    of aborting the training process.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,
        "history": history,
        "config": cfg,
    }

    try:
        torch.save(payload, temp_path)
        for attempt in range(15):
            try:
                os.replace(temp_path, path)
                return path
            except OSError as exc:
                # Windows may report ERROR_ACCESS_DENIED (5),
                # ERROR_LOCK_VIOLATION (33), or ERROR_USER_MAPPED_FILE (1224)
                # while antivirus, indexing, or a reader holds the file.
                if getattr(exc, "winerror", None) not in (5, 32, 33, 1224):
                    raise
                if attempt == 14:
                    fallback_path = path.with_name(
                        f"{path.stem}_fallback_{os.getpid()}_{time.time_ns()}{path.suffix}"
                    )
                    os.replace(temp_path, fallback_path)
                    print(
                        f"Warning: could not replace locked checkpoint {path}; "
                        f"saved the latest checkpoint to {fallback_path}"
                    )
                    return fallback_path
                time.sleep(min(1.0, 0.25 * (attempt + 1)))
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
