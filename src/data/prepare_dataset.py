from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image


def resize_images(input_dir: Path, output_dir: Path, size: Tuple[int, int] = (224, 224)) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in input_dir.rglob("*.jpg"):
        img = Image.open(image_path).convert("RGB").resize(size)
        target = output_dir / image_path.name
        img.save(target)


def augment_dataset(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Placeholder: implement flips, rotations, color jitter, etc.
    for image_path in input_dir.rglob("*.jpg"):
        target = output_dir / image_path.name
        Image.open(image_path).convert("RGB").save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset (resize, augment)")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw images")
    parser.add_argument("--output", type=Path, required=True, help="Path to processed output")
    parser.add_argument("--size", type=int, nargs=2, default=(224, 224), help="Resize (h w)")
    parser.add_argument("--augment", action="store_true", help="Apply simple augmentations")
    args = parser.parse_args()

    resize_images(args.input, args.output, (args.size[0], args.size[1]))
    if args.augment:
        augment_dataset(args.output, args.output)


if __name__ == "__main__":
    main()
