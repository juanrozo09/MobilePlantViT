from __future__ import annotations
from typing import Dict, Any

from PIL import Image

from .model import DiseaseClassifier
from ..utils.image_utils import preprocess_image


def predict(image: Image.Image) -> Dict[str, Any]:
    """Run inference on a PIL image and return placeholder results."""
    _ = preprocess_image(image)
    model = DiseaseClassifier(num_classes=2)
    # Placeholder: return fake probabilities
    return {"predicted_class": "healthy", "confidence": 0.5, "probs": {"healthy": 0.5, "diseased": 0.5}}
