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


class PlateauThenCosineScheduler:
    """
    Plateau-to-Cosine Transition Scheduler:
    - Phase 1 (Plateau / Epoch 1 -> switch_epoch, default: 50):
      Acts as ReduceLROnPlateau: If val_loss does not improve for `patience` (8) epochs,
      reduces LR by `factor` (0.5x).
    - Transition: At epoch >= switch_epoch (50), automatically transitions to Phase 2 (Cosine Annealing).
    - Phase 2 (Cosine Annealing + Reactive Decay / Epoch 51 -> total_epochs):
      LR decays smoothly along the Cosine curve from the current LR to eta_min.
      If val_loss continues not to improve for another `patience` (8) epochs, LR is scaled down by `factor`.
    """

    def __init__(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        switch_epoch: int = 50,
        eta_min: float = 1e-6,
        patience: int = 8,
        factor: float = 0.50,
        threshold: float = 0.0005,
        mode: str = "min",
        min_scale: float = 1e-4,
    ):
        self.optimizer = optimizer
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.warmup_start_factor = float(warmup_start_factor)
        self.switch_epoch = int(switch_epoch)
        self.eta_min = float(eta_min)
        self.patience = int(patience)
        self.factor = float(factor)
        self.threshold = float(threshold)
        self.mode = str(mode).lower()
        self.min_scale = float(min_scale)

        self.epoch = 0
        self.phase = "plateau"
        self.base_lrs = [float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups]
        self.best_metric = float("inf") if self.mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.cosine_start_epoch = self.switch_epoch
        self.cosine_start_lrs = list(self.base_lrs)
        self.scale = 1.0

        # Initialize for epoch 0
        self._update_lrs(0)

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.threshold
        return current > best + self.threshold

    def _calc_lrs(self, epoch: int) -> list[float]:
        if self.phase == "plateau":
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                progress = min(max(epoch / self.warmup_epochs, 0.0), 1.0)
                warmup_factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
                return [max(self.eta_min, b_lr * warmup_factor) for b_lr in self.base_lrs]
            return list(self.base_lrs)
        else:
            # Cosine decay scaled by self.scale across remaining epochs
            decay_epochs = max(1, self.total_epochs - self.cosine_start_epoch)
            progress = (epoch - self.cosine_start_epoch) / decay_epochs
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.eta_min + max(0.0, (start_lr * self.scale) - self.eta_min) * cosine
                for start_lr in self.cosine_start_lrs
            ]

    def _update_lrs(self, epoch: int) -> list[float]:
        lrs = self._calc_lrs(epoch)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        return lrs

    def step(self, metrics: float | None = None) -> tuple[list[float], str | None]:
        """
        Steps the scheduler.
        Returns (current_lrs, event_type) where event_type is:
          - 'switched_to_cosine' when transitioning from Plateau to Cosine at switch_epoch
          - 'reduced_by_factor' when LR or LR scale is reduced by factor (0.50x)
          - None otherwise
        """
        self.epoch += 1
        event = None

        # Check if reaching switch_epoch to transition to cosine
        if self.phase == "plateau" and self.epoch >= self.switch_epoch:
            self.phase = "cosine"
            self.cosine_start_epoch = self.epoch
            self.cosine_start_lrs = [float(g["lr"]) for g in self.optimizer.param_groups]
            self.num_bad_epochs = 0
            event = "switched_to_cosine"
        elif metrics is not None:
            if self._is_better(metrics, self.best_metric):
                self.best_metric = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
                if self.num_bad_epochs >= self.patience:
                    if self.phase == "plateau":
                        # Reduce base_lrs by factor during plateau phase
                        self.base_lrs = [max(self.eta_min, lr * self.factor) for lr in self.base_lrs]
                        self.num_bad_epochs = 0
                        event = "reduced_by_factor"
                    elif self.phase == "cosine":
                        # Reduce cosine scale by factor during cosine phase
                        new_scale = max(self.min_scale, self.scale * self.factor)
                        if new_scale < self.scale:
                            self.scale = new_scale
                            self.num_bad_epochs = 0
                            event = "reduced_by_factor"

        lrs = self._update_lrs(self.epoch)
        return lrs, event


class AdaptiveCosinePlateauScheduler:
    """
    Adaptive Cosine-Plateau Hybrid Scheduler:
    - Base trajectory: Smooth Warmup + Cosine Annealing decay across epochs.
    - Reactive control: If a monitored metric (e.g. val_loss) plateaus for `patience` epochs,
      it multiplies the adaptive scale factor by `plateau_factor` (e.g. 0.5), accelerating decay.
    """

    def __init__(
        self,
        optimizer,
        total_epochs: int,
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        eta_min: float = 1e-6,
        plateau_factor: float = 0.5,
        patience: int = 5,
        threshold: float = 0.0005,
        mode: str = "min",
        cooldown: int = 1,
        min_scale: float = 1e-4,
    ):
        self.optimizer = optimizer
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.warmup_start_factor = float(warmup_start_factor)
        self.eta_min = float(eta_min)

        # Plateau parameters
        self.plateau_factor = float(plateau_factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.mode = str(mode).lower()
        self.cooldown = int(cooldown)
        self.min_scale = float(min_scale)

        self.epoch = 0
        self.base_lrs = [float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups]
        self.scale = 1.0
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.best_metric = float("inf") if self.mode == "min" else float("-inf")

        # Set initial learning rates for epoch 0
        self._update_lrs(0)

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.threshold
        return current > best + self.threshold

    def _calc_lrs(self, epoch: int) -> list[float]:
        scaled_base_lrs = [max(self.eta_min, b_lr * self.scale) for b_lr in self.base_lrs]

        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            progress = min(max(epoch / self.warmup_epochs, 0.0), 1.0)
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
            return [max(self.eta_min, b_lr * factor) for b_lr in scaled_base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.eta_min + max(0.0, b_lr - self.eta_min) * cosine for b_lr in scaled_base_lrs]

    def _update_lrs(self, epoch: int) -> list[float]:
        lrs = self._calc_lrs(epoch)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        return lrs

    def step(self, metrics: float | None = None) -> tuple[list[float], bool]:
        """
        Steps the scheduler. If `metrics` is given, updates plateau tracking.
        Returns (current_lrs, was_reduced_by_plateau).
        """
        self.epoch += 1
        reduced_by_plateau = False

        if metrics is not None:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0
            elif self._is_better(metrics, self.best_metric):
                self.best_metric = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                new_scale = max(self.min_scale, self.scale * self.plateau_factor)
                if new_scale < self.scale:
                    self.scale = new_scale
                    reduced_by_plateau = True
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0

        lrs = self._update_lrs(self.epoch)
        return lrs, reduced_by_plateau


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
      - 'cosine_plateau': hybrid stepped once per epoch with a metric value (AdaptiveCosinePlateauScheduler)
    """
    t = cfg["training"]
    name = str(t.get("scheduler", "plateau_to_cosine")).lower()
    epochs = int(total_epochs if total_epochs is not None else t["epochs"])

    if name in ["plateau_to_cosine", "plateau_then_cosine", "plateau_cosine"]:
        c_cfg = t.get("cosine", {})
        p_cfg = t.get("plateau", {})
        ptc_cfg = t.get("plateau_to_cosine", {})
        switch_epoch = int(ptc_cfg.get("switch_epoch", t.get("switch_epoch", 50)))
        eta_min = float(c_cfg.get("eta_min", 1e-6))
        patience = int(p_cfg.get("patience", 8))
        factor = float(p_cfg.get("factor", 0.50))
        threshold = float(p_cfg.get("threshold", 0.0005))
        monitor = str(p_cfg.get("monitor", "val_loss")).lower()
        default_mode = "min" if "loss" in monitor else "max"
        mode = str(p_cfg.get("mode", default_mode)).lower()

        scheduler = PlateauThenCosineScheduler(
            optimizer,
            total_epochs=epochs,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=warmup_start_factor,
            switch_epoch=switch_epoch,
            eta_min=eta_min,
            patience=patience,
            factor=factor,
            threshold=threshold,
            mode=mode,
        )
        return scheduler, "plateau_to_cosine"

    if name in ["cosine_plateau", "adaptive_cosine_plateau", "hybrid", "cosine_adaptive"]:
        c_cfg = t.get("cosine", {})
        p_cfg = t.get("plateau", {})
        eta_min = float(c_cfg.get("eta_min", 1e-6))
        factor = float(p_cfg.get("factor", 0.5))
        patience = int(p_cfg.get("patience", 5))
        threshold = float(p_cfg.get("threshold", 0.0005))
        monitor = str(p_cfg.get("monitor", "val_loss")).lower()
        default_mode = "min" if "loss" in monitor else "max"
        mode = str(p_cfg.get("mode", default_mode)).lower()
        cooldown = int(p_cfg.get("cooldown", 1))

        scheduler = AdaptiveCosinePlateauScheduler(
            optimizer,
            total_epochs=epochs,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=warmup_start_factor,
            eta_min=eta_min,
            plateau_factor=factor,
            patience=patience,
            threshold=threshold,
            mode=mode,
            cooldown=cooldown,
        )
        return scheduler, "cosine_plateau"

    if name in ["reduce_on_plateau", "plateau", "reducelronplateau"]:
        p_cfg = t.get("plateau", {})
        monitor = str(p_cfg.get("monitor", "val_loss")).lower()
        default_mode = "min" if "loss" in monitor else "max"
        mode = str(p_cfg.get("mode", default_mode)).lower()
        factor = float(p_cfg.get("factor", 0.5))
        patience = int(p_cfg.get("patience", 8))
        min_lr = float(p_cfg.get("min_lr", 1e-6))
        threshold = float(p_cfg.get("threshold", 0.0005))
        cooldown = int(p_cfg.get("cooldown", 1))
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            threshold=threshold,
            cooldown=cooldown,
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
