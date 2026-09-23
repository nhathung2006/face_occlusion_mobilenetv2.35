from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_target_smoothing(
    targets: torch.Tensor,
    target_low: float = 0.02,
    target_high: float = 0.98,
) -> torch.Tensor:
    """
    Smooths binary targets from {0.0, 1.0} into [target_low, target_high] (e.g. [0.02, 0.98]).
    Target 0 (Clear) -> target_low (0.02, optimal logit ≈ -3.89)
    Target 1 (Occluded) -> target_high (0.98, optimal logit ≈ +3.89)
    """
    targets_f = targets.float()
    return targets_f * (target_high - target_low) + target_low


class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss with Logits implementation for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters:
        gamma (float): Focusing parameter that modulates the loss for easy/hard examples (default: 1.0/2.0).
        alpha (float | None): Balancing factor between positive and negative classes.
        pos_weight (torch.Tensor | None): Weight tensor for positive class in BCE loss calculation.
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        alpha: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha) if alpha is not None and 0.0 < float(alpha) < 1.0 else None
        self.reduction = str(reduction).lower()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logit values, shape [N] or [N, 1]
            targets: Ground truth targets (can be float / smoothed), same shape as logits
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable BCE loss with Log-Sum-Exp trick
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        # Probabilities
        probs = torch.sigmoid(logits)
        # p_t is the probability of the true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            focal_loss = alpha_t * modulating_factor * bce_loss
        else:
            focal_loss = modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LogitPenaltyLoss(nn.Module):
    """
    Logit Regularization Penalty for Quantization INT8 optimization:
    1. Margin Penalty: Penalizes logits when |logits| > max_logit (e.g. 3.90).
       Prevents logit explosion/saturation while leaving normal range unpenalized.
    2. L2 Penalty: Slight global L2 shrinkage on logits to keep dynamic range compact.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_logit: float = 3.90,
        margin_weight: float = 0.05,
        l2_weight: float = 0.001,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.max_logit = float(max_logit)
        self.margin_weight = float(margin_weight)
        self.l2_weight = float(l2_weight)

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if not self.enabled:
            zero_penalty = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            return zero_penalty, {"margin_penalty": 0.0, "l2_penalty": 0.0}

        logits_flat = logits.view(-1)
        total_penalty = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        margin_loss_val = 0.0
        l2_loss_val = 0.0

        # Margin penalty for exceeding max_logit (e.g. 3.90 ~ sigmoid 0.98)
        if self.margin_weight > 0.0 and self.max_logit > 0.0:
            excess = F.relu(torch.abs(logits_flat) - self.max_logit)
            margin_loss = torch.mean(excess ** 2)
            total_penalty = total_penalty + self.margin_weight * margin_loss
            margin_loss_val = margin_loss.item()

        # Global L2 penalty on logits
        if self.l2_weight > 0.0:
            l2_loss = torch.mean(logits_flat ** 2)
            total_penalty = total_penalty + self.l2_weight * l2_loss
            l2_loss_val = l2_loss.item()

        info = {
            "margin_penalty": margin_loss_val,
            "l2_penalty": l2_loss_val,
            "total_penalty": total_penalty.item(),
        }
        return total_penalty, info


class LossWithLogitPenalty(nn.Module):
    """
    Combined Loss Criterion that wraps a base classification loss (BCE/Focal)
    with Logit Regularization Penalty.
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        penalty_criterion: LogitPenaltyLoss,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.penalty_criterion = penalty_criterion

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_criterion(logits, targets)
        if self.training and self.penalty_criterion.enabled:
            penalty, _ = self.penalty_criterion(logits)
            return base_loss + penalty
        return base_loss


def build_loss_criterion(cfg: dict, device: torch.device) -> nn.Module:
    """
    Builds loss criterion based on configuration:
      - 'focal' / 'binary_focal' / 'bce_focal': BinaryFocalLossWithLogits
      - 'bce' / 'bce_with_logits': nn.BCEWithLogitsLoss
      - 'cross_entropy' / 'ce': nn.CrossEntropyLoss
    Applies LogitPenaltyLoss wrapper if configured for binary classification.
    """
    t_cfg = cfg.get("training", {})
    loss_type = str(t_cfg.get("loss_type", "focal")).lower()
    num_classes = int(cfg.get("model", {}).get("num_classes", 1))

    if num_classes == 1:
        pos_weight_val = float(t_cfg.get("pos_weight", 1.0))
        pos_weight = torch.tensor([pos_weight_val], device=device) if pos_weight_val != 1.0 else None

        if loss_type in ["focal", "bce_focal", "binary_focal", "focal_loss"]:
            focal_cfg = t_cfg.get("focal", {})
            gamma = float(focal_cfg.get("gamma", 1.0))
            alpha = focal_cfg.get("alpha", None)
            if alpha is not None:
                alpha = float(alpha)
            base_criterion = BinaryFocalLossWithLogits(
                gamma=gamma,
                alpha=alpha,
                pos_weight=pos_weight,
                reduction="mean",
            )
        else:
            base_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Logit Penalty Configuration
        lp_cfg = t_cfg.get("logit_penalty", {})
        lp_enabled = bool(lp_cfg.get("enabled", True))
        penalty_loss = LogitPenaltyLoss(
            enabled=lp_enabled,
            max_logit=float(lp_cfg.get("max_logit", 3.90)),
            margin_weight=float(lp_cfg.get("margin_weight", 0.05)),
            l2_weight=float(lp_cfg.get("l2_weight", 0.001)),
        )

        return LossWithLogitPenalty(base_criterion=base_criterion, penalty_criterion=penalty_loss)
    else:
        label_smoothing = float(t_cfg.get("label_smoothing", 0.0))
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
