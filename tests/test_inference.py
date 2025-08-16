from __future__ import annotations
from PIL import Image

from src.model.inference import predict


def test_predict_returns_dict() -> None:
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    result = predict(img)
    assert isinstance(result, dict)
    assert "predicted_class" in result
    assert "confidence" in result
