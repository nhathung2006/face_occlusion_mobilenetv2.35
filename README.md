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
│   └── config.yaml
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

## Upstream GitHub files

See `COPY_FROM_GITHUB.md`. Only the MobileNetV2 architecture source and, optionally, the pretrained 0.35 checkpoint are needed from the upstream repository.
