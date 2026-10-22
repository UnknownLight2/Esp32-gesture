#!/usr/bin/env python3
"""Train CNN 96x96 grayscale, quantize int8, export model_data.h for Arduino."""
import argparse
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import tensorflow as tf

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

LABELS = ("fist", "palm", "peace", "thumbs_up", "none")
SIZE = 96


def load_dataset(root, val_fraction, seed):
    rng = random.Random(seed)
    xs, ys = [], []
    for index, label in enumerate(LABELS):
        files = sorted((root / label).glob("*.jpg")) + sorted((root / label).glob("*.png"))
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
                xs.append(np.asarray(image, dtype=np.float32)[..., None] / 255.0)
                ys.append(index)
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int32)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - val_fraction)))
    train, val = indices[:split], indices[split:]
    return x[train], y[train], x[val], y[val]


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SIZE, SIZE, 1)),
        tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(24, activation="relu"),
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
    parser.add_argument("--out", default="model_data.h", help="Header C++ đầu ra đặt cạnh file .ino")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 0.05 <= args.val <= 0.4:
        parser.error("--val nên nằm trong khoảng 0.05 đến 0.4")
    tf.random.set_seed(args.seed)
    x_train, y_train, x_val, y_val = load_dataset(Path(args.data), args.val, args.seed)
    model = build_model()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                                  restore_best_weights=True)]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs,
              batch_size=32, callbacks=callbacks, verbose=2)
    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation: accuracy={accuracy:.3f}, loss={loss:.3f} (chỉ tham khảo; cần test ảnh mới)")

    # Representative data và int8 I/O để chạy bằng TensorFlow Lite Micro.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    representative = x_train[:min(200, len(x_train))]
    converter.representative_dataset = lambda: ([sample[None, ...]] for sample in representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    tflite_path = Path(args.out).with_suffix(".tflite")
    tflite_path.write_bytes(model_bytes)
    write_header(model_bytes, Path(args.out))
    print(f"Đã xuất {args.out} ({len(model_bytes)} bytes) và {tflite_path}")
    print("Copy model_data.h cạnh .ino, compile/upload bằng Arduino IDE.")


if __name__ == "__main__":
    main()
