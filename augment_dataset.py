#!/usr/bin/env python3
"""Create a separate dataset with a fixed number of mild image augmentations per class."""
import argparse
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

LABELS = ("fist", "palm", "peace", "thumbs_up", "none")
SIZE = 96
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare(path):
    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        resampling = getattr(Image, "Resampling", Image)
        return image.crop((left, top, left + side, top + side)).resize(
            (SIZE, SIZE), resampling.BOX
        )


def augment(image, rng):
    # Include substantial scale/position variation to mimic hands closer to the lens.
    angle = rng.uniform(-18.0, 18.0)
    scale = rng.uniform(0.70, 1.40)
    dx, dy = rng.uniform(-10, 10), rng.uniform(-10, 10)
    radians = np.deg2rad(angle)
    cosine, sine = np.cos(radians) / scale, np.sin(radians) / scale
    cx = cy = (SIZE - 1) / 2.0
    # Pillow affine coefficients map output pixels back to the source image.
    matrix = (
        cosine, sine, cx - cosine * (cx + dx) - sine * (cy + dy),
        -sine, cosine, cy + sine * (cx + dx) - cosine * (cy + dy),
    )
    transform = getattr(Image, "Transform", Image).AFFINE
    resampling = getattr(Image, "Resampling", Image)
    image = image.transform((SIZE, SIZE), transform, matrix, resample=resampling.BICUBIC,
                            fillcolor=128)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.70, 1.30))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.70, 1.30))
    if rng.random() < 0.12:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.6)))
    if rng.random() < 0.15:
        pixels = np.asarray(image, dtype=np.int16)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 2.0, pixels.shape)
        image = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8), mode="L")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset", help="Bộ ảnh nguồn")
    parser.add_argument("--out", default="dataset_augmented", help="Bộ ảnh đầu ra riêng")
    parser.add_argument("--per-class", type=int, default=100, help="Số ảnh biến thể thêm mỗi lớp")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--force", action="store_true", help="Xóa và tạo lại thư mục đầu ra")
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class phải từ 1 trở lên")
    source_root, output_root = Path(args.data), Path(args.out)
    if output_root.exists():
        if not args.force:
            parser.error(f"Thư mục {output_root} đã tồn tại; dùng --force để tạo lại")
        shutil.rmtree(output_root)

    rng = random.Random(args.seed)
    for label in LABELS:
        source_dirs = [source_root / label]
        if label == "none":
            source_dirs.append(source_root / "other")
        sources = sorted({p for folder in source_dirs if folder.is_dir()
                          for p in folder.iterdir()
                          if p.is_file() and p.suffix.lower() in EXTENSIONS})
        if not sources:
            parser.error(f"Không có ảnh nguồn cho lớp {label}: {source_dirs}")

        destination = output_root / label
        destination.mkdir(parents=True, exist_ok=True)
        for path in sources:
            shutil.copy2(path, destination / (path.stem + ".jpg"))
        prepared = {path: prepare(path) for path in sources}
        for number in range(args.per_class):
            source = rng.choice(sources)
            result = augment(prepared[source], rng)
            result.save(destination / f"aug__{source.stem}__{number:04d}.jpg", quality=94)
        print(f"{label}: {len(sources)} ảnh gốc + {args.per_class} ảnh biến thể")

    print(f"Đã tạo bộ dữ liệu: {output_root.resolve()}")
    print("Ảnh aug__ được gắn với ảnh gốc tương ứng để giữ cùng một nhóm khi chia validation.")


if __name__ == "__main__":
    main()
