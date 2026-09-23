from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss with Logits implementation for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters:
        gamma (float): Focusing parameter that modulates the loss for easy/hard examples (default: 2.0).
        alpha (float | None): Balancing factor between positive and negative classes (e.g. 0.25, 0.5, or None).
                              If alpha is in (0, 1), alpha_t = alpha * y + (1 - alpha) * (1 - y).
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
        # If alpha is 0, None, or outside (0, 1), treat as unweighted (alpha_t = 1.0)
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
        # For targets y in [0, 1], p_t = y * p + (1 - y) * (1 - p)
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


def build_loss_criterion(cfg: dict, device: torch.device) -> nn.Module:
    """
    Builds loss criterion based on configuration:
      - 'focal' / 'binary_focal' / 'bce_focal': BinaryFocalLossWithLogits
      - 'bce' / 'bce_with_logits': nn.BCEWithLogitsLoss
      - 'cross_entropy' / 'ce': nn.CrossEntropyLoss
    """
    t_cfg = cfg.get("training", {})
    loss_type = str(t_cfg.get("loss_type", "focal")).lower()
    num_classes = int(cfg.get("model", {}).get("num_classes", 1))

    if num_classes == 1:
        pos_weight_val = float(t_cfg.get("pos_weight", 1.0))
        pos_weight = torch.tensor([pos_weight_val], device=device) if pos_weight_val != 1.0 else None

        if loss_type in ["focal", "bce_focal", "binary_focal", "focal_loss"]:
            focal_cfg = t_cfg.get("focal", {})
            gamma = float(focal_cfg.get("gamma", 2.0))
            alpha = focal_cfg.get("alpha", None)
            if alpha is not None:
                alpha = float(alpha)
            return BinaryFocalLossWithLogits(
                gamma=gamma,
                alpha=alpha,
                pos_weight=pos_weight,
                reduction="mean",
            )
        else:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        label_smoothing = float(t_cfg.get("label_smoothing", 0.0))
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
