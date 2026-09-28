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

By default, export uses `checkpoints/last.pth`; pass `--checkpoint` to select another checkpoint. The ONNX input is `images` in NCHW format after RGB resize and ImageNet normalization. The output is raw `logits` (no sigmoid), clamped to `[-3.8918203, 3.8918203]`. Apply sigmoid outside ONNX only when a probability is needed.

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
