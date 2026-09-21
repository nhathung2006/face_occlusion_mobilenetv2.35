from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn as nn

from src.datasets.dataset import build_loaders
from src.training.trainer import plot_history, run_epoch, save_history
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
    if len(class_names) != num_classes:
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

    if two_stage_enabled and stage1_epochs > 0:
        freeze_backbone(model)
        trainable_s1 = count_trainable_parameters(model)
        print("\n" + "=" * 65)
        print(f"=== GIAI ĐOẠN 1: CLASSIFIER WARMUP (Epoch 1 -> {stage1_epochs}) ===")
        print(f"  Total parameters:      {total_params:,}")
        print(f"  Trainable parameters:  {trainable_s1:,} ({trainable_s1/total_params*100:.2f}%) - Only Classifier")
        print(f"  Optimizer:             {str(cfg['training']['optimizer']).upper()} (Momentum: {cfg['training'].get('momentum', 0.9)}, Nesterov: {cfg['training'].get('nesterov', True)}, Weight Decay: {cfg['training'].get('weight_decay', 1e-4)})")
        print(f"  Stage 1 LR:            {stage1_lr}")
        print(f"  Scheduler:             CosineAnnealingLR (T_max: {stage1_epochs}, eta_min: {cfg['training'].get('cosine', {}).get('eta_min', 1e-6)})")
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
        print(f"  Scheduler:             CosineAnnealingLR (T_max: {total_epochs}, eta_min: {cfg['training'].get('cosine', {}).get('eta_min', 1e-6)})")
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

    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"]["label_smoothing"]))

    best_metric = float("-inf")
    best_epoch_info = None
    patience_counter = 0
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
            print(f"=== GIAI ĐOẠN 2: DISCRIMINATIVE FINE-TUNING (Epoch {epoch} -> {total_epochs}) ===")
            print(f"  Frozen layers:         Block 0 -> Block {unfreeze_from_block - 1}")
            print(f"  Unfrozen layers:       Block {unfreeze_from_block} -> Block 17 + Conv 1x1 + Classifier")
            print(f"  Trainable parameters:  {trainable_s2:,} / {total_params:,} ({trainable_s2/total_params*100:.2f}%)")
            print(f"  Backbone LR (Block 11+ & Conv): {stage2_bb_lr}")
            print(f"  Classifier LR:          {stage2_clf_lr}")
            print(f"  Warmup:                 {warmup_epochs} epochs (start factor: {warmup_start_factor:.2f})")
            print(f"  Scheduler:             CosineAnnealingLR (T_max: {remaining_epochs}, eta_min: {cfg['training'].get('cosine', {}).get('eta_min', 1e-6)})")
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
            patience_counter = 0

        batch_scheduler = scheduler if step_type == "batch" else None
        train_loss, train_m = run_epoch(
            model, train_loader, criterion, device, num_classes, optimizer, batch_scheduler
        )
        with torch.no_grad():
            val_loss, val_m = run_epoch(
                model, val_loader, criterion, device, num_classes
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
            "train_accuracy": train_m.accuracy,
            "val_accuracy": val_m.accuracy,
            "train_precision": train_m.precision,
            "val_precision": val_m.precision,
            "train_recall": train_m.recall,
            "val_recall": val_m.recall,
            "train_f1": train_m.f1,
            "val_f1": val_m.f1,
        }
        history.append(row)

        if step_type == "plateau":
            plateau_monitor = str(cfg["training"].get("plateau", {}).get("monitor", "val_f1"))
            plateau_val = row.get(plateau_monitor, val_loss)
            scheduler.step(plateau_val)
        elif step_type == "epoch":
            scheduler.step()

        lr_str = f"LR(BB: {bb_lr:.6f}, Clf: {clf_lr:.6f})" if len(optimizer.param_groups) > 1 else f"LR: {clf_lr:.6f}"
        print(
            f"Epoch {epoch:03d}/{total_epochs:03d} [{stage_tag}] | "
            f"Train [Loss: {train_loss:.4f}, Acc: {train_m.accuracy:.4f}, Prec: {train_m.precision:.4f}, Rec: {train_m.recall:.4f}, F1: {train_m.f1:.4f}] | "
            f"Val [Loss: {val_loss:.4f}, Acc: {val_m.accuracy:.4f}, Prec: {val_m.precision:.4f}, Rec: {val_m.recall:.4f}, F1: {val_m.f1:.4f}] | "
            f"{lr_str}"
        )

        save_checkpoint(last_path, model, optimizer, epoch, best_metric, history, cfg)

        monitor = str(cfg["training"]["early_stopping"]["monitor"])
        current = float(row[monitor])
        mode = str(cfg["training"]["early_stopping"]["mode"]).lower()
        improved = current > best_metric if mode == "max" else current < best_metric
        if improved:
            best_metric = current
            best_epoch_info = row
            patience_counter = 0
            saved_best_path = save_checkpoint(
                best_path, model, optimizer, epoch, best_metric, history, cfg
            )
            best_path = Path(saved_best_path)
            print(
                f"  >>> Saved new best checkpoint at Epoch {epoch:03d} "
                f"({monitor}: {current:.4f}) to {best_path}"
            )
        else:
            patience_counter += 1

        if (
            bool(cfg["training"]["early_stopping"]["enabled"])
            and (not two_stage_enabled or epoch > stage1_epochs)
            and patience_counter >= int(cfg["training"]["early_stopping"]["patience"])
        ):
            print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement in Stage 2.")
            break

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
    print("                     TRAINING SUMMARY (BEST F1)")
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
