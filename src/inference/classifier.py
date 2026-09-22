from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


class ONNXFaceOcclusionClassifier:
    """Small wrapper intended for later RTSP -> detector -> classifier pipeline."""

    def __init__(self, onnx_path: str, image_size: int = 112, class_names=None):
        self.class_names = list(class_names or ["clear", "occluded"])
        self.image_size = int(image_size)
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess_one(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError("Expected HxWxC image")
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        return self.transform(image).numpy()

    def predict_batch(self, images):
        batch = np.stack([self._preprocess_one(img) for img in images]).astype(np.float32)
        logits = self.session.run(["logits"], {self.input_name: batch})[0]
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            logits_1d = logits.reshape(-1)
            probs_occ = 1.0 / (1.0 + np.exp(-logits_1d))
            indices = (probs_occ >= 0.5).astype(int)
            confidences = np.where(indices == 1, probs_occ, 1.0 - probs_occ)
            return [
                {
                    "class_id": int(i),
                    "class_name": self.class_names[int(i)],
                    "confidence": float(conf),
                }
                for i, conf in zip(indices, confidences)
            ]
        else:
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            indices = probs.argmax(axis=1)
            return [
                {
                    "class_id": int(i),
                    "class_name": self.class_names[int(i)],
                    "confidence": float(probs[row, int(i)]),
                }
                for row, i in enumerate(indices)
            ]

    def predict(self, image):
        return self.predict_batch([image])[0]
