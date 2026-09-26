from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
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

from src.datasets.dataset import build_loaders
from src.training.losses import build_loss_criterion
from src.training.trainer import EarlyStopping, plot_history, run_epoch, save_history
from src.utils.config import load_config
from src.utils.model import (
    build_model,
    count_parameters,
    count_trainable_parameters,
    freeze_backbone,
    setup_layer_freezing,
)
from src.utils.seed import seed_everything
from src.utils.training import build_optimizer, build_scheduler, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 for face occlusion classification")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _, class_names = build_loaders(cfg)
    expected_classes = list(cfg["data"]["class_names"])
    print(f"Classes: {class_names}")
    if class_names != expected_classes:
        raise ValueError(f"Dataset classes {class_names} != config classes {expected_classes}")

    num_classes = int(cfg["model"]["num_classes"])
    if num_classes == 1:
        if len(class_names) != 2:
            raise ValueError(
                f"num_classes=1 (binary classification), but dataset has {len(class_names)} classes (expected 2: {expected_classes})"
            )
    elif len(class_names) != num_classes:
        raise ValueError(f"num_classes={num_classes}, but dataset has {len(class_names)} class folders")

    model = build_model(cfg).to(device)
    total_params = count_parameters(model)
    total_epochs = int(cfg["training"]["epochs"])

    two_stage_cfg = cfg["training"].get("two_stage", {})
    two_stage_enabled = bool(two_stage_cfg.get("enabled", False))
    stage1_epochs = int(two_stage_cfg.get("stage1_epochs", 8)) if two_stage_enabled else 0
    stage1_lr = float(two_stage_cfg.get("stage1_lr", 0.001))
    unfreeze_from_block = int(two_stage_cfg.get("unfreeze_from_block", cfg["training"].get("unfreeze_from_block", 11)))
    warmup_epochs = int(two_stage_cfg.get("warmup_epochs", 0))
    warmup_start_factor = float(two_stage_cfg.get("warmup_start_factor", 0.1))
    stage2_bb_lr = float(two_stage_cfg.get("stage2_backbone_lr", cfg["training"].get("backbone_lr", 0.0001)))
    stage2_clf_lr = float(two_stage_cfg.get("stage2_classifier_lr", cfg["training"].get("classifier_lr", 0.001)))

    scheduler_name = str(cfg["training"].get("scheduler", "cosine")).upper()
    if two_stage_enabled and stage1_epochs > 0:
        freeze_backbone(model)
        trainable_s1 = count_trainable_parameters(model)
        print("\n" + "=" * 65)
        print(f"=== STAGE 1: CLASSIFIER WARMUP (Epoch 1 -> {stage1_epochs}) ===")
        print(f"  Total parameters:      {total_params:,}")
        print(f"  Trainable parameters:  {trainable_s1:,} ({trainable_s1/total_params*100:.2f}%) - Only Classifier")
        print(f"  Optimizer:             {str(cfg['training']['optimizer']).upper()} (Momentum: {cfg['training'].get('momentum', 0.9)}, Nesterov: {cfg['training'].get('nesterov', True)}, Weight Decay: {cfg['training'].get('weight_decay', 1e-4)})")
        print(f"  Stage 1 LR:            {stage1_lr}")
        print(f"  Scheduler:             {scheduler_name}")
        print("=" * 65 + "\n")

        optimizer = build_optimizer(model, cfg, lr=stage1_lr)
        scheduler, step_type = build_scheduler(
            optimizer, cfg, len(train_loader), lr=stage1_lr, total_epochs=stage1_epochs
        )
    else:
        setup_layer_freezing(model, unfreeze_from_block=unfreeze_from_block)
        trainable_p = count_trainable_parameters(model)
        print("\n" + "=" * 65)
        print("                 MODEL & FREEZING CONFIGURATION")
        print("=" * 65)
        print(f"  Total parameters:      {total_params:,}")
        print(f"  Trainable parameters:  {trainable_p:,} ({trainable_p/total_params*100:.2f}%)")
        print(f"  Frozen layers:         Block 0 -> Block {unfreeze_from_block - 1}")
        print(f"  Unfrozen layers:       Block {unfreeze_from_block} -> Block 17 + Conv 1x1 + Classifier")
        print(f"  Optimizer:             {str(cfg['training']['optimizer']).upper()} (Momentum: {cfg['training'].get('momentum', 0.9)}, Nesterov: {cfg['training'].get('nesterov', True)}, Weight Decay: {cfg['training'].get('weight_decay', 1e-4)})")
        print(f"  Backbone LR:           {stage2_bb_lr}")
        print(f"  Classifier LR:          {stage2_clf_lr}")
        print(f"  Scheduler:             {scheduler_name}")
        print("=" * 65 + "\n")

        optimizer = build_optimizer(model, cfg, backbone_lr=stage2_bb_lr, classifier_lr=stage2_clf_lr)
        scheduler, step_type = build_scheduler(
            optimizer,
            cfg,
            len(train_loader),
            lr=stage2_clf_lr,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=warmup_start_factor,
        )

    label_smoothing = float(cfg["training"].get("label_smoothing", 0.05))
    ts_cfg = cfg["training"].get("target_smoothing", {})
    ts_enabled = bool(ts_cfg.get("enabled", False))
    ts_low = float(ts_cfg.get("target_low", 0.02))
    ts_high = float(ts_cfg.get("target_high", 0.98))

    lp_cfg = cfg["training"].get("logit_penalty", {})
    lp_enabled = bool(lp_cfg.get("enabled", False))

    criterion = build_loss_criterion(cfg, device)
    loss_type_name = str(cfg["training"].get("loss_type", "bce")).upper()
    if num_classes == 1 and "FOCAL" in loss_type_name:
        f_cfg = cfg["training"].get("focal", {})
        print(f"  Loss Function:         BCE Focal Loss (gamma: {f_cfg.get('gamma', 1.0)}, alpha: {f_cfg.get('alpha', 0.0)})")
    elif num_classes == 1:
        print(f"  Loss Function:         BCEWithLogitsLoss (pos_weight: {cfg['training'].get('pos_weight', 1.0)})")
    else:
        print(f"  Loss Function:         CrossEntropyLoss (label_smoothing: {label_smoothing})")

    if num_classes == 1:
        if ts_enabled:
            print(f"  Target Smoothing:      ENABLED -> Clear={ts_low:.2f}, Occluded={ts_high:.2f} (Target Logit ~ [{-3.89:.2f}, {+3.89:.2f}])")
        if lp_enabled:
            print(f"  Logit Regularizer:     ENABLED -> Max Logit={lp_cfg.get('max_logit', 3.90)}, Margin Weight={lp_cfg.get('margin_weight', 0.05)}, L2 Weight={lp_cfg.get('l2_weight', 0.001)}")

    es_cfg = cfg["training"].get("early_stopping", {})
    early_stopping = EarlyStopping(
        monitor=str(es_cfg.get("monitor", "val_loss")),
        mode=str(es_cfg.get("mode", "min")),
        patience=int(es_cfg.get("patience", 16)),
        min_delta=float(es_cfg.get("min_delta", 0.0005)),
        restore_best_weights=bool(es_cfg.get("restore_best_weights", True)),
        enabled=bool(es_cfg.get("enabled", True)),
    )
    best_epoch_info = None
    history = []

    best_path = Path(cfg["checkpoint"]["best_path"])
    last_path = Path(cfg["checkpoint"]["last_path"])
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, total_epochs + 1):
        # Stage 2 Transition at epoch stage1_epochs + 1
        if two_stage_enabled and epoch == stage1_epochs + 1:
            setup_layer_freezing(model, unfreeze_from_block=unfreeze_from_block)
            trainable_s2 = count_trainable_parameters(model)
            remaining_epochs = max(1, total_epochs - stage1_epochs)

            print("\n" + "=" * 65)
            print(f"=== STAGE 2: DISCRIMINATIVE FINE-TUNING (Epoch {epoch} -> {total_epochs}) ===")
            print(f"  Frozen layers:         Block 0 -> Block {unfreeze_from_block - 1}")
            print(f"  Unfrozen layers:       Block {unfreeze_from_block} -> Block 17 + Conv 1x1 + Classifier")
            print(f"  Trainable parameters:  {trainable_s2:,} / {total_params:,} ({trainable_s2/total_params*100:.2f}%)")
            print(f"  Backbone LR (Block 11+ & Conv): {stage2_bb_lr}")
            print(f"  Classifier LR:          {stage2_clf_lr}")
            print(f"  Warmup:                 {warmup_epochs} epochs (start factor: {warmup_start_factor:.2f})")
            print(f"  Scheduler:             {scheduler_name}")
            print("=" * 65 + "\n")

            optimizer = build_optimizer(model, cfg, backbone_lr=stage2_bb_lr, classifier_lr=stage2_clf_lr)
            scheduler, step_type = build_scheduler(
                optimizer,
                cfg,
                len(train_loader),
                lr=stage2_clf_lr,
                total_epochs=remaining_epochs,
                warmup_epochs=warmup_epochs,
                warmup_start_factor=warmup_start_factor,
            )
            early_stopping.reset_patience()

        batch_scheduler = scheduler if step_type == "batch" else None
        train_loss, train_m, train_details = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes,
            optimizer,
            batch_scheduler,
            label_smoothing=label_smoothing,
            target_smoothing_enabled=ts_enabled,
            target_low=ts_low,
            target_high=ts_high,
            return_details=True,
        )
        with torch.no_grad():
            val_loss, val_m, val_details = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                num_classes,
                label_smoothing=0.0,
                target_smoothing_enabled=ts_enabled,
                target_low=ts_low,
                target_high=ts_high,
                return_details=True,
            )

        # Extract per-group learning rates safely by group name or index
        group_map = {group.get("name", f"group_{i}"): group["lr"] for i, group in enumerate(optimizer.param_groups)}
        if "backbone" in group_map and "classifier" in group_map:
            bb_lr = float(group_map["backbone"])
            clf_lr = float(group_map["classifier"])
        elif len(optimizer.param_groups) > 1:
            bb_lr = float(optimizer.param_groups[0]["lr"])
            clf_lr = float(optimizer.param_groups[1]["lr"])
        else:
            clf_lr = float(optimizer.param_groups[0]["lr"])
            bb_lr = 0.0 if (two_stage_enabled and epoch <= stage1_epochs) else clf_lr

        stage_tag = "S1" if (two_stage_enabled and epoch <= stage1_epochs) else "S2"
        row = {
            "epoch": epoch,
            "stage": stage_tag,
            "lr": clf_lr,
            "backbone_lr": bb_lr,
            "classifier_lr": clf_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_classification_loss": train_details["classification_loss"],
            "train_logit_penalty": train_details["logit_penalty"],
            "val_classification_loss": val_details["classification_loss"],
            "val_logit_penalty": val_details["logit_penalty"],
            "val_max_abs_logit": val_details["max_abs_logit"],
            "val_out_of_range_ratio": val_details["out_of_range_ratio"],
            "train_accuracy": train_m.accuracy,
            "val_accuracy": val_m.accuracy,
            "train_precision": train_m.precision,
            "val_precision": val_m.precision,
            "train_recall": train_m.recall,
            "val_recall": val_m.recall,
            "train_f1": train_m.f1,
            "val_f1": val_m.f1,
        }
        if val_m.logit_min is not None:
            row["val_logit_min"] = val_m.logit_min
            row["val_logit_max"] = val_m.logit_max
            row["val_prob_min"] = val_m.prob_min
            row["val_prob_max"] = val_m.prob_max
        history.append(row)

        if step_type in ["plateau_to_cosine", "plateau_then_cosine"]:
            plateau_cfg = cfg["training"].get("plateau", {})
            plateau_monitor = str(plateau_cfg.get("monitor", "val_loss"))
            plateau_val = float(row.get(plateau_monitor, val_loss))
            next_lrs, event = scheduler.step(plateau_val)
            if event == "switched_to_cosine":
                print(
                    f"  >>> [PlateauToCosine] Reached Epoch {epoch} (switch_epoch: {getattr(scheduler, 'switch_epoch', 50)}). "
                    f"Activated Phase 2 Cosine Annealing (Epoch {epoch + 1} -> {total_epochs}) for deep convergence!"
                )
            elif event == "reduced_by_factor":
                phase_name = "Phase 1 Plateau" if getattr(scheduler, 'phase', '') == 'plateau' else "Phase 2 Cosine"
                print(
                    f"  >>> [PlateauToCosine] {plateau_monitor} did not improve in {scheduler.patience} epochs ({phase_name}). "
                    f"Reduced Learning Rate by factor {scheduler.factor:.2f}!"
                )
        elif step_type in ["cosine_plateau", "hybrid"]:
            plateau_cfg = cfg["training"].get("plateau", {})
            plateau_monitor = str(plateau_cfg.get("monitor", "val_loss"))
            plateau_val = float(row.get(plateau_monitor, val_loss))
            prev_scale = scheduler.scale
            next_lrs, was_reduced = scheduler.step(plateau_val)
            if was_reduced:
                print(
                    f"  >>> [AdaptiveCosinePlateau] {plateau_monitor} plateaued after {scheduler.patience} epochs. "
                    f"Reduced LR Scale: {prev_scale:.3f} -> {scheduler.scale:.3f}"
                )
        elif step_type == "plateau":
            plateau_cfg = cfg["training"].get("plateau", {})
            plateau_monitor = str(plateau_cfg.get("monitor", "val_loss"))
            plateau_val = float(row.get(plateau_monitor, val_loss))
            prev_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler.step(plateau_val)
            new_lrs = [group["lr"] for group in optimizer.param_groups]
            if any(n_lr < p_lr for n_lr, p_lr in zip(new_lrs, prev_lrs)):
                print(f"  >>> [ReduceLROnPlateau] {plateau_monitor} plateaued after {plateau_cfg.get('patience', 8)} epochs. Reducing Learning Rate:")
                for i, (n_lr, p_lr) in enumerate(zip(new_lrs, prev_lrs)):
                    g_name = optimizer.param_groups[i].get("name", f"group_{i}")
                    print(f"      Group '{g_name}': {p_lr:.6e} -> {n_lr:.6e}")
        elif step_type == "epoch":
            scheduler.step()

        lr_str = f"LR(BB: {bb_lr:.6f}, Clf: {clf_lr:.6f})" if len(optimizer.param_groups) > 1 else f"LR: {clf_lr:.6f}"
        print(
            f"Epoch {epoch:03d}/{total_epochs:03d} [{stage_tag}] | "
            f"Train [Loss: {train_loss:.4f}, BCE: {train_details['classification_loss']:.4f}, Penalty: {train_details['logit_penalty']:.4f}, "
            f"Acc: {train_m.accuracy:.4f}, Prec: {train_m.precision:.4f}, Rec: {train_m.recall:.4f}, F1: {train_m.f1:.4f}] | "
            f"Val [Loss: {val_loss:.4f}, Penalty: {val_details['logit_penalty']:.4f}, "
            f"Acc: {val_m.accuracy:.4f}, Prec: {val_m.precision:.4f}, Rec: {val_m.recall:.4f}, F1: {val_m.f1:.4f}] | "
            f"{lr_str}"
        )

        save_checkpoint(last_path, model, optimizer, epoch, early_stopping.best_metric, history, cfg)

        should_stop = early_stopping.step(epoch, row, model)
        if early_stopping.best_epoch == epoch:
            best_epoch_info = row
            saved_best_path = save_checkpoint(
                best_path, model, optimizer, epoch, early_stopping.best_metric, history, cfg
            )
            best_path = Path(saved_best_path)
            print(
                f"  >>> Saved new best checkpoint at Epoch {epoch:03d} "
                f"({early_stopping.monitor}: {early_stopping.best_metric:.4f}) to {best_path}"
            )

        status_str = early_stopping.get_status_str(row)
        print(f"      [{status_str}]")

        if (
            should_stop
            and (not two_stage_enabled or epoch > stage1_epochs)
        ):
            print(
                f"\nEarly stopping triggered at Epoch {epoch:03d}: "
                f"'{early_stopping.monitor}' did not improve in {early_stopping.patience} epochs."
            )
            if early_stopping.restore_best_weights:
                print(
                    f"  >>> Restored best weights from Epoch {early_stopping.best_epoch:03d} "
                    f"({early_stopping.monitor}: {early_stopping.best_metric:.4f}). Patience reset to 0."
                )
            break

    if early_stopping.restore_best_weights and early_stopping.best_state_dict is not None:
        early_stopping.restore(model)
        if early_stopping.best_row is not None:
            best_epoch_info = early_stopping.best_row

    save_history(history, Path("outputs/history.csv"))
    plot_history(history, Path("outputs/plots"))

    evaluation_cfg = cfg.get("evaluation", {})
    if bool(evaluation_cfg.get("enabled", True)) and best_epoch_info is not None:
        project_root = Path(__file__).resolve().parent
        eval_split = str(evaluation_cfg.get("split", "val"))
        eval_output = Path(
            evaluation_cfg.get("misclassified_dir", "outputs/misclassified_val")
        )
        if not eval_output.is_absolute():
            eval_output = project_root / eval_output

        eval_command = [
            sys.executable,
            str(project_root / "test.py"),
            "--config",
            str(Path(args.config).resolve()),
            "--checkpoint",
            str(best_path.resolve()),
            "--split",
            eval_split,
            "--misclassified-dir",
            str(eval_output.resolve()),
        ]
        print("\nRunning automatic validation evaluation...")
        try:
            subprocess.run(eval_command, cwd=project_root, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                "Warning: training completed, but automatic misclassification "
                f"evaluation failed with exit code {exc.returncode}."
            )

    print("\n" + "=" * 65)
    print(f"               TRAINING SUMMARY (BEST {str(early_stopping.monitor).upper()})")
    print("=" * 65)
    if best_epoch_info is not None:
        print(f"  Best Epoch:       {best_epoch_info['epoch']:03d} / {total_epochs:03d} [Stage {best_epoch_info.get('stage', 'N/A')}]")
        print(f"  Train Loss:       {best_epoch_info['train_loss']:.4f}")
        print(f"  Val Loss:         {best_epoch_info['val_loss']:.4f}")
        print(f"  Train Accuracy:   {best_epoch_info['train_accuracy']:.4f} ({best_epoch_info['train_accuracy']*100:.2f}%)")
        print(f"  Val Accuracy:     {best_epoch_info['val_accuracy']:.4f} ({best_epoch_info['val_accuracy']*100:.2f}%)")
        print(f"  Val Macro F1:     {best_epoch_info['val_f1']:.4f}")
        print(f"  Val Precision:    {best_epoch_info['val_precision']:.4f}")
        print(f"  Val Recall:       {best_epoch_info['val_recall']:.4f}")
        print("-" * 65)
        print(f"  Train Macro F1:   {best_epoch_info['train_f1']:.4f}")
        print(f"  Train Precision:  {best_epoch_info['train_precision']:.4f}")
        print(f"  Train Recall:     {best_epoch_info['train_recall']:.4f}")
        print(f"  Best Checkpoint:  {best_path.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
