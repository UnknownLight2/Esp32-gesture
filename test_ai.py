#!/usr/bin/env python3
"""Test the TFLite gesture model on a photo, laptop camera, or ESP32 MJPEG stream."""
import argparse
from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from gesture_preprocess import apply_outline

try:
    import requests
except ImportError:
    requests = None

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

LABELS = ("fist", "palm", "peace", "thumbs_up", "none")
LABELS_VI = ("nắm đấm", "xòe bàn tay", "hai ngón / peace", "like", "không nhận dạng")
SIZE = 96
LIVE_WINDOW = "Gesture model live"
STABLE_WINDOW = 7
REQUIRED_MATCHES = 5
MIN_LIVE_CONFIDENCE = 0.50


class GestureModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        if tuple(self.input["shape"]) != (1, SIZE, SIZE, 1):
            raise ValueError(f"Model có input lạ: {self.input['shape']}; cần (1, 96, 96, 1).")

    @staticmethod
    def preprocess(frame):
        height, width = frame.shape[:2]
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        crop = frame[top:top + side, left:left + side]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        outlined = apply_outline(gray)
        return outlined.astype(np.float32)[None, ..., None] / 255.0

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
        output = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        out_scale, out_zero = self.output["quantization"]
        if self.output["dtype"] in (np.int8, np.uint8) and out_scale:
            output = (output - out_zero) * out_scale
        index = int(np.argmax(output))
        return index, float(output[index]), output


def draw_outline_preview(display, model, source_frame):
    """Show the exact edge-enhanced 96x96 image fed to the classifier."""
    values = model.preprocess(source_frame)[0, :, :, 0]
    outline = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    outline = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
    height, width = display.shape[:2]
    side = min(144, width // 3, height // 3)
    if side < 48:
        return display
    outline = cv2.resize(outline, (side, side), interpolation=cv2.INTER_NEAREST)
    x, y = width - side - 8, 8
    display[y:y + side, x:x + side] = outline
    cv2.rectangle(display, (x, y), (x + side, y + side), (0, 255, 255), 1)
    cv2.putText(display, "OUTLINE", (x, min(height - 4, y + side + 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return display


def frames_from_mjpeg(url):
    if requests is None:
        raise RuntimeError("Thiếu requests: chạy py -m pip install requests")
    buf = bytearray()
    with requests.get(url, stream=True, timeout=(5, 30)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start < 0:
                    if len(buf) > 2:
                        del buf[:-1]
                    break
                if end < 0:
                    if start:
                        del buf[:start]
                    if len(buf) > 2_000_000:
                        del buf[:-1]
                    break
                jpg = bytes(buf[start:end + 2])
                del buf[:end + 2]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
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
        print(f"Không mở được hộp thoại chọn ảnh: {exc}", file=sys.stderr)
        return ""


def print_prediction(model, frame, name):
    index, confidence, scores = model.predict(frame)
    print(f"{name}: {LABELS[index]} / {LABELS_VI[index]} ({confidence:.1%})")
    print("  " + " | ".join(f"{label}={float(score):.1%}" for label, score in zip(LABELS, scores)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Ảnh đơn cần phân loại")
    source.add_argument("--camera", type=int, help="Webcam máy tính, thường là 0")
    source.add_argument("--url", help="ESP32 MJPEG URL, ví dụ http://192.168.4.1:81/stream")
    parser.add_argument("--model", default="model_outline.tflite", help="Đường dẫn model .tflite")
    parser.add_argument("--no-preview", action="store_true", help="Chỉ in kết quả ảnh, không mở cửa sổ xem trước")
    args = parser.parse_args()

    try:
        model = GestureModel(args.model)
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                parser.error(f"Không đọc được ảnh: {args.image}")
            print_prediction(model, frame, args.image)
            if not args.no_preview:
                index, confidence, _ = model.predict(frame)
                preview = frame.copy()
                cv2.putText(preview, f"{LABELS_VI[index]} ({confidence:.0%})", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                draw_outline_preview(preview, model, frame)
                cv2.imshow("Gesture model - image", preview)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return 0

        if args.camera is not None:
            cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError(f"Không mở được webcam {args.camera}; thử --camera 1.")
            frames = None
        else:
            cap = None
            frames = frames_from_mjpeg(args.url)

        print("Nhấn Q/ESC để thoát; nhấn I để chọn ảnh. Live chỉ xác nhận khi cùng lớp ổn định ở ít nhất 5/7 khung hình.")
        recent = deque(maxlen=STABLE_WINDOW)
        try:
            while True:
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Không đọc được khung hình từ webcam.")
                else:
                    frame = next(frames)
                index, confidence, scores = model.predict(frame)
                recent.append((index, confidence, scores))
                counts = np.bincount([item[0] for item in recent], minlength=len(LABELS))
                candidate = int(np.argmax(counts))
                candidate_scores = [item[1] for item in recent if item[0] == candidate]
                confirmed = (len(recent) == STABLE_WINDOW and
                             counts[candidate] >= REQUIRED_MATCHES and
                             float(np.mean(candidate_scores)) >= MIN_LIVE_CONFIDENCE)
                display = frame.copy()
                if confirmed:
                    live_text = f"Da xac nhan: {LABELS_VI[candidate]} ({np.mean(candidate_scores):.0%})"
                    color = (0, 255, 0)
                else:
                    live_text = f"Dang xac nhan... {LABELS_VI[index]} ({confidence:.0%}) [{counts[index]}/{STABLE_WINDOW}]"
                    color = (0, 200, 255)
                cv2.putText(display, live_text, (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.putText(display, "I: chon anh | Q: thoat", (10, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                draw_outline_preview(display, model, frame)
                cv2.imshow(LIVE_WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("i"):
                    # Close the live preview while the still-image picker/preview is active.
                    cv2.destroyWindow(LIVE_WINDOW)
                    path = choose_image()
                    if path:
                        photo = cv2.imread(path)
                        if photo is not None:
                            print_prediction(model, photo, path)
                            photo_source = photo.copy()
                            photo_index, photo_conf, _ = model.predict(photo)
                            cv2.putText(photo, f"{LABELS_VI[photo_index]} ({photo_conf:.0%})", (10, 32),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                            draw_outline_preview(photo, model, photo_source)
                            cv2.imshow("Gesture model - selected image", photo)
                            cv2.waitKey(0)
                            cv2.destroyWindow("Gesture model - selected image")
                    # The next loop iteration recreates the live preview and continues the stream.
                    recent.clear()
        finally:
            if cap is not None:
                cap.release()
            if frames is not None:
                frames.close()
            cv2.destroyAllWindows()
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
