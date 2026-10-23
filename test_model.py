#!/usr/bin/env python3
"""Run an exported TFLite model on one image or a folder of images."""
import argparse
from pathlib import Path
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
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess(path):
    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        resampling = getattr(Image, "Resampling", Image)
        image = image.crop((left, top, left + side, top + side)).resize(
            (SIZE, SIZE), resampling.BOX
        )
        outlined = apply_outline(np.asarray(image, dtype=np.uint8))
        return outlined.astype(np.float32)[None, ..., None] / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Ảnh cần đoán hoặc thư mục ảnh")
    parser.add_argument("--model", default="model_outline.tflite", help="File .tflite")
    args = parser.parse_args()

    target = Path(args.image)
    if target.is_dir():
        images = sorted(p for p in target.rglob("*") if p.suffix.lower() in EXTENSIONS)
    elif target.is_file():
        images = [target]
    else:
        parser.error(f"Không tìm thấy: {target}")
    if not images:
        parser.error(f"Không tìm thấy ảnh trong: {target}")

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    scale, zero_point = input_info["quantization"]

    correct = total = 0
    has_labels = target.is_dir() and all(p.parent.name in LABELS for p in images)
    for path in images:
        input_float = preprocess(path)
        if input_info["dtype"] == np.int8:
            input_data = np.clip(np.rint(input_float / scale + zero_point), -128, 127).astype(np.int8)
        elif input_info["dtype"] == np.uint8:
            input_data = np.clip(np.rint(input_float / scale + zero_point), 0, 255).astype(np.uint8)
        else:
            input_data = input_float.astype(input_info["dtype"])

        interpreter.set_tensor(input_info["index"], input_data)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_info["index"])[0].astype(np.float32)
        out_scale, out_zero = output_info["quantization"]
        if output_info["dtype"] in (np.int8, np.uint8) and out_scale:
            scores = (scores - out_zero) * out_scale
        predicted = int(np.argmax(scores))
        confidence = float(scores[predicted])
        expected = path.parent.name if has_labels else None
        print(f"{path}: {LABELS[predicted]} ({confidence:.1%})" +
              (f" | nhãn đúng: {expected}" if expected else ""))
        if expected:
            correct += LABELS[predicted] == expected
            total += 1

    if total:
        print(f"Accuracy trên tập ảnh này: {correct}/{total} = {correct / total:.1%}")


if __name__ == "__main__":
    main()
