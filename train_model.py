#!/usr/bin/env python3
"""Train CNN 96x96 grayscale, quantize int8, export model_data.h for Arduino."""
import argparse
from datetime import datetime
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
from gesture_preprocess import apply_outline

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

LABELS = ("fist", "palm", "peace", "thumbs_up", "none")
SIZE = 96


def source_group_id(path):
    stem = path.stem
    if stem.startswith("aug__"):
        parts = stem[5:].split("__")
        if len(parts) >= 3 and parts[-2] in ("v1", "v2"):
            stem = "__".join(parts[:-2])
        elif len(parts) >= 2:
            stem = "__".join(parts[:-1])
    # Collector filenames contain label_YYYYMMDD_HHMMSS_microseconds. Group
    # nearby burst frames into 5-second capture windows to reduce validation leakage.
    parts = stem.split("_")
    if len(parts) >= 4 and len(parts[-3]) == 8 and len(parts[-2]) == 6:
        try:
            captured = datetime.strptime(parts[-3] + parts[-2], "%Y%m%d%H%M%S")
            window = int(captured.timestamp()) // 5
            return f"capture-{parts[-3]}-{window}"
        except ValueError:
            pass
    return stem


def load_dataset(root, val_fraction, seed, use_outline=True):
    rng = random.Random(seed)
    xs, ys, group_ids = [], [], []
    for index, label in enumerate(LABELS):
        folders = [root / label]
        # Accept the common folder name "other" for the negative/unknown class.
        if label == "none":
            folders.append(root / "other")
        files = []
        for folder in folders:
            files.extend(folder.glob("*.jpg"))
            files.extend(folder.glob("*.png"))
        files = sorted(set(files))
        if len(files) < 10:
            raise ValueError(f"Lớp {label} chỉ có {len(files)} ảnh; cần ít nhất 10 ảnh để chạy thử (nên thu 200-500+).")
        rng.shuffle(files)
        for path in files:
            with Image.open(path) as image:
                image = image.convert("L")
                # Cùng quy tắc với firmware: cắt hình vuông ở giữa rồi resize 96x96.
                w, h = image.size
                side = min(w, h)
                left, top = (w - side) // 2, (h - side) // 2
                resampling = getattr(Image, "Resampling", Image)
                image = image.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), resampling.BOX)
                pixels = np.asarray(image, dtype=np.uint8)
                if use_outline:
                    pixels = apply_outline(pixels)
                xs.append(pixels.astype(np.float32)[..., None] / 255.0)
                ys.append(index)
                group_ids.append((index, source_group_id(path)))
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int32)
    # Keep each original image and all its aug__ variants in a single split.
    # This prevents near-duplicate augmentations leaking into validation.
    train_groups, val_groups = set(), set()
    for class_index in range(len(LABELS)):
        class_groups = sorted({group for idx, group in group_ids if idx == class_index})
        rng.shuffle(class_groups)
        val_count = max(1, int(len(class_groups) * val_fraction))
        if len(class_groups) < 2:
            raise ValueError(f"Lớp {LABELS[class_index]} cần ít nhất 2 ảnh gốc khác nhau để chia validation.")
        val_groups.update((class_index, group) for group in class_groups[:val_count])
        train_groups.update((class_index, group) for group in class_groups[val_count:])
    train = np.asarray([i for i, group in enumerate(group_ids) if group in train_groups])
    val = np.asarray([i for i, group in enumerate(group_ids) if group in val_groups])
    return x[train], y[train], x[val], y[val]


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SIZE, SIZE, 1)),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(48, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def write_header(model_bytes, output):
    values = ",\n".join("  " + ", ".join(f"0x{b:02x}" for b in model_bytes[i:i + 12])
                        for i in range(0, len(model_bytes), 12))
    labels = ", ".join('"' + name + '"' for name in LABELS)
    header = ("#pragma once\n#include <cstdint>\n\n"
              f"alignas(16) const unsigned char g_model[] = {{\n{values}\n}};\n"
              f"const unsigned int g_model_len = {len(model_bytes)};\n"
              f"constexpr int kNumClasses = {len(LABELS)};\n"
              f"const char* const kLabels[kNumClasses] = {{{labels}}};\n")
    output.write_text(header, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset", help="Thư mục chứa fist/palm/peace/thumbs_up/none")
    parser.add_argument("--out", default="model_outline.h", help="Header C++ đầu ra đặt cạnh file .ino")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-outline", action="store_true",
                        help="Dùng ảnh grayscale gốc; không áp dụng outline/Sobel")
    parser.add_argument("--class-multiplier", action="append", default=[], metavar="LABEL=VALUE",
                        help="Nhân trọng số lớp sau khi cân bằng, ví dụ thumbs_up=1.5; có thể lặp lại")
    args = parser.parse_args()
    if not 0.05 <= args.val <= 0.4:
        parser.error("--val nên nằm trong khoảng 0.05 đến 0.4")
    tf.random.set_seed(args.seed)
    x_train, y_train, x_val, y_val = load_dataset(
        Path(args.data), args.val, args.seed, use_outline=not args.no_outline
    )
    # Give the model both natural hand orientations; do not mirror camera output at inference.
    x_train = np.concatenate((x_train, np.flip(x_train, axis=2)), axis=0)
    y_train = np.concatenate((y_train, y_train), axis=0)
    model = build_model()
    class_counts = np.bincount(y_train, minlength=len(LABELS))
    class_weight = {
        i: len(y_train) / (len(LABELS) * max(1, int(count)))
        for i, count in enumerate(class_counts)
    }
    multipliers = {}
    for item in args.class_multiplier:
        label, separator, value = item.partition("=")
        if not separator or label not in LABELS:
            parser.error("--class-multiplier cần LABEL=VALUE với LABEL thuộc {}".format(LABELS))
        try:
            multiplier = float(value)
        except ValueError:
            parser.error("Trọng số không phải số: {}".format(item))
        if not np.isfinite(multiplier) or multiplier <= 0:
            parser.error("Trọng số phải dương và hữu hạn: {}".format(item))
        multipliers[label] = multiplier
    for index, label in enumerate(LABELS):
        class_weight[index] *= multipliers.get(label, 1.0)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                          restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-5),
    ]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
              batch_size=32, callbacks=callbacks, class_weight=class_weight, verbose=2)
    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation: accuracy={accuracy:.3f}, loss={loss:.3f} (chỉ tham khảo; cần test ảnh mới)")
    validation_scores = model.predict(x_val, batch_size=32, verbose=0)
    validation_predicted = np.argmax(validation_scores, axis=1)
    for index, label in enumerate(LABELS):
        mask = y_val == index
        recall = float(np.mean(validation_predicted[mask] == index)) if np.any(mask) else 0.0
        print(f"Validation recall {label}: {int(np.sum(validation_predicted[mask] == index))}/{int(np.sum(mask))} = {recall:.1%}")

    # Representative data và int8 I/O để chạy bằng TensorFlow Lite Micro.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Calibrate quantization from every class, not just the first sorted class.
    representative_rng = np.random.RandomState(args.seed)
    representative_indices = []
    for class_index in range(len(LABELS)):
        candidates = np.flatnonzero(y_train == class_index)
        take = min(40, len(candidates))
        if take:
            representative_indices.extend(
                representative_rng.choice(candidates, size=take, replace=False).tolist()
            )
    representative = x_train[representative_indices]
    converter.representative_dataset = lambda: ([sample[None, ...]] for sample in representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    tflite_path = Path(args.out).with_suffix(".tflite")
    tflite_path.write_bytes(model_bytes)
    write_header(model_bytes, Path(args.out))
    print(f"Đã xuất {args.out} ({len(model_bytes)} bytes) và {tflite_path}")
    print(f"Tiền xử lý: {'grayscale không outline' if args.no_outline else 'grayscale + outline'}")
    print(f"Copy {args.out} cạnh .ino, compile/upload bằng Arduino IDE.")


if __name__ == "__main__":
    main()
