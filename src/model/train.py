from __future__ import annotations
import argparse

from .model import DiseaseClassifier
from ..data.dataloader import get_training_dataset, get_validation_dataset


def train(epochs: int = 1) -> None:
    model = DiseaseClassifier(num_classes=2)
    train_ds = get_training_dataset()
    val_ds = get_validation_dataset()
    # Placeholder training loop
    for _ in range(epochs):
        _ = (model, train_ds, val_ds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train disease classifier")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    train(epochs=args.epochs)


if __name__ == "__main__":
    main()
