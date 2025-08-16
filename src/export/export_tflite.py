from __future__ import annotations


def export_to_tflite(saved_model_dir: str, output_path: str) -> None:
    """Convert a TensorFlow model to TFLite (placeholder)."""
    _ = (saved_model_dir, output_path)


if __name__ == "__main__":
    # Example usage placeholder
    export_to_tflite("path/to/saved_model", "mobile_integration/model.tflite")
