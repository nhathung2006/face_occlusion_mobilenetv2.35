# Files to copy from the upstream GitHub repo

Repository:
https://github.com/d-li14/mobilenetv2.pytorch

## Required source file

Copy only this Python source file:

`models/imagenet/mobilenetv2.py`

Into this project as:

`src/models/mobilenetv2.py`

The included file is already a source-compatible copy of the upstream architecture, with attribution. You can replace it with the upstream file if you prefer to keep the repository version unchanged.

## Required pretrained checkpoint (for transfer learning)

Download/copy:

`pretrained/mobilenetv2_0.35-b2e15951.pth`

Into:

`weights/mobilenetv2_0.35-b2e15951.pth`

No other upstream files are required for this project. In particular, do not copy the upstream `imagenet.py` training script or its ImageNet-specific utilities; this project has its own config-driven dataset/training/evaluation/export pipeline.

The upstream repository is Apache License 2.0. See its LICENSE for the applicable terms.
