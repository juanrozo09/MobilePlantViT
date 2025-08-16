from __future__ import annotations
from typing import Tuple

from PIL import Image


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize/crop/normalize the image for model input (placeholder)."""
    if not isinstance(image, Image.Image):
        raise TypeError("preprocess_image expects a PIL.Image.Image")
    return image.resize(target_size)
