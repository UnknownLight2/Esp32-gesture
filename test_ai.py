#!/usr/bin/env python3
"""Test the image-based TFLite gesture classifier on images, folders, or a live camera."""
import argparse
from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np
import tensorflow as tf
from gesture_preprocess import apply_outline

try:
    import requests
except ImportError:
    requests = None

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

LABELS = ("fist", "palm", "peace", "thumbs_up", "none")
LABELS_VI = ("nắm đấm", "xòe bàn tay", "hai ngón / peace", "like", "không nhận dạng")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE = 96
WINDOW = "Gesture AI - CNN"
STABLE_WINDOW = 7
REQUIRED_MATCHES = 5
MIN_CONFIDENCE = 0.50


class GestureModel:
    def __init__(self, model_path, outline=False):
        self.path = Path(model_path)
        if not self.path.is_file():
            raise FileNotFoundError("Không tìm thấy model: {}".format(self.path))
        self.outline = outline
        self.interpreter = tf.lite.Interpreter(model_path=str(self.path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        if tuple(self.input["shape"]) != (1, SIZE, SIZE, 1):
            raise ValueError("Model cần input (1, 96, 96, 1), nhận được {}".format(self.input["shape"]))

    def preprocess(self, frame):
        height, width = frame.shape[:2]
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        crop = frame[top:top + side, left:left + side]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        pixels = apply_outline(gray) if self.outline else gray
        return pixels.astype(np.float32)[None, ..., None] / 255.0

    def predict(self, frame):
        values = self.preprocess(frame)
        scale, zero = self.input["quantization"]
        dtype = self.input["dtype"]
        if dtype in (np.int8, np.uint8):
            limits = np.iinfo(dtype)
            values = np.clip(np.rint(values / scale + zero), limits.min, limits.max).astype(dtype)
        else:
            values = values.astype(dtype)
        self.interpreter.set_tensor(self.input["index"], values)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        out_scale, out_zero = self.output["quantization"]
        if self.output["dtype"] in (np.int8, np.uint8) and out_scale:
            scores = (scores - out_zero) * out_scale
        index = int(np.argmax(scores))
        return index, float(scores[index]), scores


def frames_from_mjpeg(url):
    if requests is None:
        raise RuntimeError("Thiếu requests; cài bằng: py -m pip install requests")
    buffer = bytearray()
    with requests.get(url, stream=True, timeout=(5, 30)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start < 0:
                    if len(buffer) > 1:
                        del buffer[:-1]
                    break
                if end < 0:
                    if start:
                        del buffer[:start]
                    if len(buffer) > 2_000_000:
                        del buffer[:-1]
                    break
                jpg = bytes(buffer[start:end + 2])
                del buffer[:end + 2]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame


def choose_image():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Chọn ảnh để kiểm tra model",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")],
        )
        root.destroy()
        return path
    except Exception as exc:
        print("Không mở được hộp thoại chọn ảnh: {}".format(exc), file=sys.stderr)
        return ""


def print_prediction(model, frame, name, expected=None):
    index, confidence, scores = model.predict(frame)
    line = "{}: {} / {} ({:.1%})".format(name, LABELS[index], LABELS_VI[index], confidence)
    if expected is not None:
        line += " | nhãn đúng: {} | {}".format(expected, "ĐÚNG" if LABELS[index] == expected else "SAI")
    print(line)
    print("  " + " | ".join("{}={:.1%}".format(label, float(score)) for label, score in zip(LABELS, scores)))
    return index


def show_model_input(display, model, frame):
    values = model.preprocess(frame)[0, :, :, 0]
    tile = cv2.cvtColor(np.clip(values * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    h, w = display.shape[:2]
    side = min(144, w // 3, h // 3)
    if side < 48:
        return
    tile = cv2.resize(tile, (side, side), interpolation=cv2.INTER_NEAREST)
    x, y = w - side - 8, 8
    display[y:y + side, x:x + side] = tile
    cv2.rectangle(display, (x, y), (x + side, y + side), (0, 255, 255), 1)
    label = "INPUT: OUTLINE" if model.outline else "INPUT: GRAY"
    cv2.putText(display, label, (x, min(h - 4, y + side + 16)), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 255, 255), 1)


def evaluate_folder(model, folder):
    root = Path(folder)
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError("Không tìm thấy ảnh trong {}".format(root))
    if not all(p.parent.name in LABELS for p in images):
        raise ValueError("Tập test cần thư mục con tên theo nhãn: {}".format(", ".join(LABELS)))
    totals = {label: 0 for label in LABELS}
    correct = {label: 0 for label in LABELS}
    matrix = {label: {pred: 0 for pred in LABELS} for label in LABELS}
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print("Bỏ qua ảnh không đọc được:", path, file=sys.stderr)
            continue
        expected = path.parent.name
        predicted = print_prediction(model, frame, str(path), expected)
        totals[expected] += 1
        correct[expected] += int(LABELS[predicted] == expected)
        matrix[expected][LABELS[predicted]] += 1
    total = sum(totals.values())
    good = sum(correct.values())
    print("\nAccuracy: {}/{} = {:.1%}".format(good, total, good / total if total else 0.0))
    for label in LABELS:
        print("{}: {}/{}".format(label, correct[label], totals[label]))
    print("\nMa trận nhầm lẫn (hàng=nhãn đúng, cột=dự đoán):")
    print("expected\\pred | " + " | ".join(LABELS))
    for label in LABELS:
        print("{} | {}".format(label, " | ".join(str(matrix[label][pred]) for pred in LABELS)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Phân loại một ảnh tĩnh")
    source.add_argument("--test-dir", help="Đánh giá folder test có các thư mục theo nhãn")
    source.add_argument("--camera", type=int, help="Webcam laptop, thường là 0")
    source.add_argument("--url", help="MJPEG stream ESP32, ví dụ http://192.168.4.1:81/stream")
    parser.add_argument("--model", default="model_raw_focus4.tflite", help="Model ảnh TFLite")
    parser.add_argument("--outline", action="store_true", help="Chỉ bật cho model đã train bằng outline")
    parser.add_argument("--no-preview", action="store_true", help="Không mở cửa sổ khi dùng --image")
    args = parser.parse_args()
    try:
        model = GestureModel(args.model, outline=args.outline)
        if args.test_dir:
            evaluate_folder(model, args.test_dir)
            return 0
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                raise ValueError("Không đọc được ảnh: {}".format(args.image))
            index = print_prediction(model, frame, args.image)
            if not args.no_preview:
                preview = frame.copy()
                cv2.putText(preview, "CNN: {} ({:.0%})".format(LABELS_VI[index], model.predict(frame)[1]),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                show_model_input(preview, model, frame)
                cv2.imshow("Gesture AI - image", preview)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return 0

        cap = None
        frames = None
        if args.camera is not None:
            cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError("Không mở được webcam {}; thử --camera 1".format(args.camera))
        else:
            frames = frames_from_mjpeg(args.url)
        print("Model: {} | input {} | nhấn Q/ESC thoát, I chọn ảnh".format(
            model.path, "grayscale + outline" if args.outline else "grayscale gốc"))
        recent = deque(maxlen=STABLE_WINDOW)
        try:
            while True:
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Không đọc được frame webcam")
                else:
                    frame = next(frames)
                index, confidence, scores = model.predict(frame)
                recent.append((index, confidence))
                counts = np.bincount([item[0] for item in recent], minlength=len(LABELS))
                candidate = int(np.argmax(counts))
                candidate_scores = [score for label, score in recent if label == candidate]
                confirmed = (len(recent) == STABLE_WINDOW and counts[candidate] >= REQUIRED_MATCHES
                             and float(np.mean(candidate_scores)) >= MIN_CONFIDENCE)
                if confirmed:
                    message = "CNN: {} ({:.0%}) [stable]".format(
                        LABELS_VI[candidate], float(np.mean(candidate_scores)))
                    color = (0, 255, 0)
                else:
                    message = "CNN: {} ({:.0%}) [{}/{}]".format(
                        LABELS_VI[index], confidence, counts[index], STABLE_WINDOW)
                    color = (0, 200, 255)
                display = frame.copy()
                cv2.putText(display, message, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
                cv2.putText(display, "I: chon anh | Q: thoat", (10, display.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                show_model_input(display, model, frame)
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("i"):
                    cv2.destroyWindow(WINDOW)
                    path = choose_image()
                    if path:
                        photo = cv2.imread(path)
                        if photo is not None:
                            selected = print_prediction(model, photo, path)
                            photo_display = photo.copy()
                            cv2.putText(photo_display, "CNN: {} ({:.0%})".format(
                                LABELS_VI[selected], model.predict(photo)[1]), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            show_model_input(photo_display, model, photo)
                            cv2.imshow("Gesture AI - selected image", photo_display)
                            cv2.waitKey(0)
                            cv2.destroyWindow("Gesture AI - selected image")
                    recent.clear()
        finally:
            if cap is not None:
                cap.release()
            if frames is not None:
                frames.close()
            cv2.destroyAllWindows()
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        print("Lỗi: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
