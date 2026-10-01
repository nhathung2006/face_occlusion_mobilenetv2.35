# Face Occlusion Classification - MobileNetV2 0.35

Purpose: classify a detected face crop into two classes:

- `clear`: khuôn mặt phù hợp/không bị che khuất đáng kể
- `occluded`: khuôn mặt bị che khuất

The project is intentionally separated from the future face detector. The expected production pipeline is:

`RTSP -> Face Detector ONNX -> face crop -> MobileNetV2 ONNX -> clear/occluded`

## Project structure

```text
face_occlusion_mobilenetv2/
├── config/
│   ├── config.yaml                 # Existing/default 0.35 experiment
│   ├── benchmark_base.yaml         # Shared benchmark settings
│   ├── benchmark_035.yaml          # 0.35-specific paths and width
│   └── benchmark_05.yaml           # 0.5-specific paths and width
├── src/
│   ├── models/
│   │   └── mobilenetv2.py
│   ├── datasets/
│   │   └── dataset.py
│   ├── training/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── inference/
│   │   └── __init__.py
│   └── utils/
│       ├── config.py
│       ├── model.py
│       ├── seed.py
│       └── training.py
├── dataset/
│   ├── train/clear
│   ├── train/occluded
│   ├── val/clear
│   ├── val/occluded
│   ├── test/clear
│   └── test/occluded
├── weights/
├── checkpoints/
├── outputs/
├── train.py
├── benchmark.py
├── test.py
├── export_onnx.py
├── inference.py
├── requirements.txt
└── README.md
```

## Dataset

Keep all data inside the single `dataset/` directory. Each split must have the two class folders shown above.

## Initial configuration

The default experiment is:

- MobileNetV2
- width multiplier `0.35`
- 2 classes
- dropout `0.20`
- image size `160x160`
- SGD + Momentum + Nesterov
- cosine learning rate
- 3 warmup epochs
- early stopping on validation macro F1

With the upstream architecture and 2-class classifier, the model has about **398,690 trainable parameters**.

## Setup

Create/activate a virtual environment, then:

```bash
pip install -r requirements.txt
```

Copy the pretrained checkpoint described in `weights/README.md`.

## Train

```bash
python train.py --config config/config.yaml
```

Outputs:

- `checkpoints/best.pth`
- `checkpoints/last.pth`
- `outputs/history.csv`
- `outputs/plots/*.png`

## Test

```bash
python test.py --config config/config.yaml
```

## Export ONNX

```bash
python export_onnx.py --config config/config.yaml
```

Output:

`outputs/onnx/mobilenetv2_035_face_occlusion.onnx`

The export keeps a dynamic batch dimension so the later detector pipeline can classify multiple face crops in one inference call.

By default, export uses `checkpoints/last.pth`; pass `--checkpoint` to select another checkpoint. The ONNX input is `images` in NCHW format after RGB resize and ImageNet normalization. The output is the raw logit before sigmoid, without clamping. Apply sigmoid outside ONNX only when a probability is needed.

## Single-image ONNX inference

```bash
python inference.py path/to/face.jpg
```

## What to change for experiments

Use `config/config.yaml` for:

- `model.width_mult`: `0.35`, later test `0.25`
- `model.dropout`
- `data.image_size`: `160`, later test `128`
- optimizer and learning rate
- warmup
- early stopping patience
- augmentation strength

Do not modify `src/models/mobilenetv2.py` just to change normal training hyperparameters.

## Four-label training and automatic evaluation

Run the four-label experiment with:

```powershell
.\.venv\Scripts\python.exe train_4class_single_logit.py --config config/config_4class_single_logit.yaml --mode train
```

When training finishes, the best checkpoint is automatically evaluated by default. A new timestamped
folder is written under `outputs/model_evaluation/` with the 2-class and 4-class metrics, confusion
matrices, training curves, and misclassified validation images. `latest_evaluation.json` points to
the newest result folder. Set `evaluation.auto_evaluate_after_training: false` in the config to
disable this behavior; `--mode evaluate` remains available for a manual evaluation.

When exporting this four-label checkpoint later with `--mode export`, the ONNX graph returns only
the binary clear/occluded logit (shape `[batch, 1]`, output name `logit`). It clamps the logit to
`±training.logit_penalty.max_logit` (currently `3.8918203`) and contains no sigmoid. Training and
validation still use sigmoid for their loss and metrics; the training penalty is soft, so raw
PyTorch logits may exceed this bound before export.

### Train a single-logit clear/occluded model

The binary dataset is stored at
`C:/Thực tập LUMI/model_BaiToan/Deepleaning/data/dataset_hungtn/dataset_hungtn_binary_train_val_v1`.
Its only class folders are `clear` and `occluded`; the original subclasses are
retained below those folders only for provenance. The classifier outputs one
raw logit. Focal loss applies sigmoid internally in a numerically stable form;
training targets use smoothing and a soft logit penalty. Validation uses the
original hard labels. The exported ONNX model returns the unclamped raw logit;
apply sigmoid outside the model to get the occluded probability.

```powershell
.\.venv\Scripts\python.exe train.py
```

`train.py` uses the existing `config/config.yaml` by default. Checkpoints and
training/evaluation outputs are kept under `checkpoints/dataset_hungtn_single_logit/`
and `outputs/dataset_hungtn_single_logit/`, preserving the saved two-logit run.

## Matched auxiliary-loss experiments

Experiment settings are in `experiments` in `config/config_4class_single_logit.yaml`.
The runner uses the same pretrained initialization, seed, grouped 80/20 split, scheduler,
augmentation and early stopping for each loss weight. It selects the best **binary validation
macro-F1**, with accuracy as the tie-breaker. No ONNX is exported.

First audit, then prepare and train (use a new run directory for a new dataset revision):

```powershell
.\.venv\Scripts\python.exe run_aux_weight_experiments.py --mode prepare --run-dir outputs/aux_weight_experiments/my_run
.\.venv\Scripts\python.exe run_aux_weight_experiments.py --mode prepare --run-dir outputs/aux_weight_experiments/my_run --apply-dedup
.\.venv\Scripts\python.exe run_aux_weight_experiments.py --mode run --run-dir outputs/aux_weight_experiments/my_run
```

Preparation compares decoded RGB pixels, reports byte-identical and pixel-identical copies,
and refuses conflicting labels. `--apply-dedup` moves only exact copies from the four class
folders to `data.root/_dedup_quarantine/<run-name>/`; nothing is permanently deleted.
`dedup_audit.json` records the retained file, original path, checksum and quarantine path for
every copy. To restore a copy, move that specific quarantine file back to its recorded original
path only if that path is free. Do not restore duplicates during an experiment.

pHash-near images are **not deleted**: their connected groups remain wholly in train or val.
This reduces one form of leakage but does not guarantee that every same-person/video sequence
is detected. `split_manifest.json` freezes file hashes, labels and group assignments. Trial
configs reference it through `data.split_manifest`; the original config retains its existing
random split unless a manifest is explicitly configured. Dataset additions, removals or relabels
invalidate the fixed trial split: prepare a new run after finishing dataset review.

Each trial contains its own `config.yaml`, `train.log`, `checkpoints/`, `evaluation/` (plots,
confusion matrices and error images), and `result.json`. `comparison.json` names the winner.
The normal checkpoints and evaluation reports are not overwritten. Use the chosen trial's
`config.yaml` when evaluating or reviewing that checkpoint; do not pair it with the old random
split. A zero auxiliary weight is a binary-only ablation: its untrained four-class head is not
suitable for four-label review. Accuracy gaps use train and val evaluated without augmentation.
One seed/one validation split is an initial comparison, not an independent test-set guarantee.

## Web review for previous-run errors

The local review page uses the configured best checkpoint and reproduces the deterministic train/val
split. It shows four-class mistakes at any confidence, plus correctly classified images below the
configured confidence cutoff. The split, checkpoint prediction, confidence, and review progress are
saved in a manifest keyed by checkpoint and filter settings, so each checkpoint/filter session keeps
its own history after images move. It uses the Python standard library and existing project
dependencies; Flask/Streamlit are not required.

Start it from the project root:

```powershell
.\.venv\Scripts\python.exe relabel_web.py
```

Then open `http://127.0.0.1:8765`. Choosing one of the four labels immediately moves that image into
the matching class folder under `data.root`. The move is recorded in the configured manifest. A
name collision is rejected instead of overwriting a file. “Đưa ảnh khỏi dataset” moves an unwanted
image into the configured `_relabel_deleted` quarantine folder (not permanently deleted); it can be
restored from the review page. The page shows a completion banner and a one-time browser alert when
the final candidate is reviewed. The manifest records label changes and quarantine/restore actions.
The page confirms quarantine/restore inline and reports the new relative file path after a successful
move. If it reports a permission error, the dataset has not changed: stop the old web process and
restart it from a VS Code terminal that can write to the configured `data.root` (outside this project).

To review strong pose candidates currently labeled `clear_side_face`, start a separate web mode:

```powershell
.\.venv\Scripts\python.exe relabel_web.py --mode side-pose --port 8766
```

Open `http://127.0.0.1:8766/`. The default filter prioritizes photos whose auxiliary head gives
`occluded_pose` a probability of at least `relabel_web.side_pose_review.priority_probability_min`
(currently `0.20`). This is a review score, not a verified face-angle measurement. Switch the
selector to **Toàn bộ clear_side_face** to inspect every image, and filter train or val separately.
Assigning a new label moves that original file to the corresponding class folder. The side-pose
mode has its own manifest and does not replace the ordinary error-review history.

## Fixed four-class train/val snapshot

The original four-class config divides images deterministically, but its split is recalculated
when files are added or relabeled. To lock a new 80/20 split after relabeling is finished, run:

```powershell
.\.venv\Scripts\python.exe tools\freeze_4class_split.py --output dataset\fixed_4class_20260929
```

This copies, rather than moves, source images into `train/<class>` and `val/<class>` under the
new folder. `split_manifest.json` records image hashes and similarity groups. Do not overwrite
an existing frozen folder; choose a new output name for a later dataset version. The generated
`train_config.yaml` trains and evaluates from these physical folders and writes checkpoints to
a separate location:

```powershell
.\.venv\Scripts\python.exe train_4class_single_logit.py --config dataset\fixed_4class_20260929\train_config.yaml --check-data
.\.venv\Scripts\python.exe train_4class_single_logit.py --config dataset\fixed_4class_20260929\train_config.yaml --mode train
```

Keep using the original config for `relabel_web.py`; it changes the source dataset, not this
frozen snapshot. A later relabel does not silently modify the snapshot or its train/val split.

## Benchmark MobileNetV2 0.35 vs 0.5

The two variants share the same source code and dataset. Shared training/data settings live in
`config/benchmark_base.yaml`; edit `config/benchmark_035.yaml` and
`config/benchmark_05.yaml` for model-specific settings. Checkpoints and run outputs are kept
separate so one variant does not overwrite the other.

To train only the new 0.5 model from ImageNet pretrained weights, then evaluate both saved
checkpoints on the same validation split:

```bash
python benchmark.py --train 05
```

To evaluate both existing checkpoints without training:

```bash
python benchmark.py
```

To intentionally retrain both variants sequentially from their configured pretrained weights:

```bash
python benchmark.py --train both
```

The comparison table is written to `outputs/benchmark/summary.csv`; each model's detailed
report, confusion matrix, and misclassified images are saved under its own `outputs/benchmark/`
subfolder. The 0.5 pretrained weights are downloaded automatically if internet access is available.
`benchmark.py` runs the variants sequentially to avoid competing for the same GPU and memory.

## Upstream GitHub files

See `COPY_FROM_GITHUB.md`. The project can use upstream pretrained weights for both the 0.35 and
0.5 width variants.
