from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "035": PROJECT_ROOT / "config" / "benchmark_035.yaml",
    "05": PROJECT_ROOT / "config" / "benchmark_05.yaml",
}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_checked(command: list[str]) -> None:
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def validate_shared_benchmark_settings(configs: dict[str, dict]) -> None:
    reference = configs["035"]
    shared_sections = ("data", "training", "augmentation", "evaluation", "inference")
    for section in shared_sections:
        ref_value = reference.get(section, {})
        for name, config in configs.items():
            value = config.get(section, {})
            if value != ref_value:
                raise ValueError(
                    f"Benchmark variants must share the '{section}' section; "
                    f"035={ref_value!r}, {name}={value!r}. "
                    "Edit config/benchmark_base.yaml so the comparison uses the same setup."
                )

    model_shared_keys = ("name", "num_classes", "dropout", "pretrained", "freeze_backbone_epochs")
    for key in model_shared_keys:
        ref_value = reference["model"].get(key)
        for name, config in configs.items():
            value = config["model"].get(key)
            if value != ref_value:
                raise ValueError(
                    f"Benchmark variants must share model.{key}; "
                    f"035={ref_value!r}, {name}={value!r}. "
                    "Edit config/benchmark_base.yaml so the comparison uses the same setup."
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate MobileNetV2 0.35 and 0.5 using separate configs and artifacts."
    )
    parser.add_argument(
        "--train",
        choices=("none", "035", "05", "both"),
        default="none",
        help=(
            "Optionally train a variant before evaluating both. "
            "Use '05' to keep the existing 0.35 benchmark checkpoint unchanged."
        ),
    )
    args = parser.parse_args()

    configs = {name: load_config(path) for name, path in VARIANTS.items()}
    validate_shared_benchmark_settings(configs)

    if args.train != "none":
        train_variants = ("035", "05") if args.train == "both" else (args.train,)
        for name in train_variants:
            print(f"\n===== Training MobileNetV2 {name} (sequentially) =====", flush=True)
            run_checked(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "train.py"),
                    "--config",
                    str(VARIANTS[name]),
                ]
            )

    rows = []
    for name, config_path in VARIANTS.items():
        cfg = configs[name]
        checkpoint = project_path(cfg["checkpoint"]["best_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing MobileNetV2 {name} checkpoint: {checkpoint}\n"
                f"Train it with: python benchmark.py --train {name}"
            )

        output_root = project_path(cfg.get("outputs", {}).get("root", "outputs"))
        eval_dir = output_root / f"eval_{cfg['evaluation'].get('split', 'val')}"
        print(f"\n===== Evaluating MobileNetV2 {name} =====", flush=True)
        run_checked(
            [
                sys.executable,
                str(PROJECT_ROOT / "test.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--split",
                str(cfg["evaluation"].get("split", "val")),
                "--output-dir",
                str(eval_dir),
            ]
        )

        metrics_path = eval_dir / "metrics.json"
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
        rows.append(
            {
                "model": f"MobileNetV2 {name}",
                "width_mult": cfg["model"]["width_mult"],
                "split": metrics["split"],
                "num_samples": metrics["num_samples"],
                "parameters": metrics["model_parameters"],
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "evaluation_seconds": metrics["evaluation_seconds"],
                "images_per_second": metrics["images_per_second"],
                "checkpoint": str(checkpoint),
                "metrics_file": str(metrics_path),
            }
        )

    summary_path = PROJECT_ROOT / "outputs" / "benchmark" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Benchmark summary (metrics are fractions 0-1) =====")
    for row in rows:
        print(
            f"{row['model']}: accuracy={row['accuracy']:.4f}, "
            f"macro-F1={row['macro_f1']:.4f}, loss={row['loss']:.4f}, "
            f"params={row['parameters']:,}, val-throughput={row['images_per_second']:.1f} img/s, "
            f"samples={row['num_samples']}"
        )
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
