#!/usr/bin/env python3
"""Train and test a hand-landmark gesture classifier or the legacy TFLite image model."""
import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import random
import sys
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image
from gesture_preprocess import apply_outline

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

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
LANDMARK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
LANDMARK_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
LANDMARK_LABELS_VI = LABELS_VI


class GestureModel:
    def __init__(self, model_path):
        if tf is None:
            raise RuntimeError("Thiếu TensorFlow. Cài requirements-train.txt hoặc chạy --mode landmark.")
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


def ensure_landmark_model(path):
    path = Path(path)
    if path.is_file() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Đang tải MediaPipe gesture model tới", path)
    try:
        urllib.request.urlretrieve(LANDMARK_MODEL_URL, str(path))
    except Exception as exc:
        raise RuntimeError("Không tải được landmark model; kiểm tra Internet hoặc tự tải model theo README.") from exc
    if path.stat().st_size < 100_000:
        path.unlink(missing_ok=True)
        raise RuntimeError("File MediaPipe model tải về không hợp lệ.")
    return path


class HandLandmarkDetector:
    def __init__(self, model_path, is_image=False):
        if mp is None:
            raise RuntimeError("Thiếu MediaPipe. Cài bằng: py -m pip install -r requirements-landmark.txt")
        model_path = ensure_landmark_model(model_path)
        vision = mp.tasks.vision
        self.image_mode = is_image
        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
                running_mode=vision.RunningMode.IMAGE if is_image else vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.last_timestamp = 0

    def predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.image_mode:
            return self.detector.detect(image)
        self.last_timestamp = max(int(time.monotonic() * 1000), self.last_timestamp + 1)
        return self.detector.detect_for_video(image, self.last_timestamp)

    def close(self):
        self.detector.close()


def landmark_features(landmarks):
    """Normalize 21 xyz points around wrist and palm axis; no image mirroring."""
    points = np.asarray([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    origin = points[0].copy()
    points -= origin
    palm = points[9, :2]
    scale = float(np.linalg.norm(palm))
    if scale < 1e-5:
        return None
    angle = -np.pi / 2.0 - float(np.arctan2(palm[1], palm[0]))
    cosine, sine = np.cos(angle), np.sin(angle)
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    points[:, 0] = cosine * x - sine * y
    points[:, 1] = sine * x + cosine * y
    points /= scale
    return points.reshape(-1)


def mirror_features(features):
    points = np.asarray(features, dtype=np.float32).reshape(21, 3).copy()
    points[:, 0] *= -1.0
    return points.reshape(-1)


def image_group_id(path):
    stem = path.stem
    if stem.startswith("aug__"):
        stem = stem[5:].rsplit("__", 1)[0]
    parts = stem.split("_")
    if len(parts) >= 4 and len(parts[-3]) == 8 and len(parts[-2]) == 6:
        try:
            captured = datetime.strptime(parts[-3] + parts[-2], "%Y%m%d%H%M%S")
            return "{}-{}".format(parts[-3], int(captured.timestamp()) // 5)
        except ValueError:
            pass
    return stem


def predicted_hands(classifier, result, threshold=0.55):
    hands = []
    for i, landmarks in enumerate(result.hand_landmarks):
        features = landmark_features(landmarks)
        if features is None:
            continue
        probabilities = classifier.predict(features[None, :], verbose=0)[0]
        index = int(np.argmax(probabilities))
        label = LABELS[index]
        if float(probabilities[index]) < threshold:
            label = "none"
        handed = "Hand {}".format(i + 1)
        if i < len(result.handedness) and result.handedness[i]:
            category = result.handedness[i][0]
            handed = category.display_name or category.category_name
        hands.append((label, float(probabilities[index]), handed))
    return hands


def train_landmark_model(data_dir, model_path, landmark_model_path, epochs=40):
    if tf is None:
        raise RuntimeError("Thiếu TensorFlow. Cài bằng: py -m pip install -r requirements-train.txt")
    detector = HandLandmarkDetector(landmark_model_path, is_image=True)
    data_dir = Path(data_dir)
    samples, targets, groups = [], [], []
    per_class = {}
    try:
        for class_index, label in enumerate(LABELS):
            folders = [data_dir / label]
            if label == "none":
                folders.append(data_dir / "other")
            files = sorted({p for folder in folders if folder.exists()
                            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
                            for p in folder.glob(pattern)})
            accepted = 0
            for path in files:
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                result = detector.predict(frame)
                if not result.hand_landmarks:
                    continue
                features = landmark_features(result.hand_landmarks[0])
                if features is None:
                    continue
                samples.append(features)
                targets.append(class_index)
                groups.append(image_group_id(path))
                accepted += 1
            per_class[label] = accepted
    finally:
        detector.close()

    print("Landmark samples extracted:", per_class)
    if any(per_class[label] < 2 for label in LABELS):
        raise RuntimeError("Mỗi lớp cần ít nhất 2 ảnh có bàn tay nhìn rõ; ảnh không có tay bị bỏ qua.")

    rng = random.Random(42)
    train_indices, val_indices = [], []
    targets_array = np.asarray(targets, dtype=np.int32)
    for class_index in range(len(LABELS)):
        class_groups = sorted({groups[i] for i, target in enumerate(targets) if target == class_index})
        if len(class_groups) < 2:
            raise RuntimeError("Lớp {} cần ảnh chụp ở ít nhất 2 nhóm thời điểm khác nhau để đánh giá.".format(
                LABELS[class_index]))
        rng.shuffle(class_groups)
        val_count = max(1, int(round(len(class_groups) * 0.2)))
        validation_groups = set(class_groups[:val_count])
        for index, target in enumerate(targets):
            if target != class_index:
                continue
            if groups[index] in validation_groups:
                val_indices.append(index)
            else:
                train_indices.append(index)

    features = np.asarray(samples, dtype=np.float32)
    x_train = features[train_indices]
    y_train = targets_array[train_indices]
    x_val = features[val_indices]
    y_val = targets_array[val_indices]
    x_train = np.concatenate([x_train, np.asarray([mirror_features(x) for x in x_train])], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)

    inputs = tf.keras.Input(shape=(63,), name="hand_landmarks")
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(len(LABELS), activation="softmax")(x)
    classifier = tf.keras.Model(inputs, outputs)
    classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    classifier.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                   batch_size=min(32, max(4, len(x_train) // 8)), verbose=2,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                                                  restore_best_weights=True)])
    loss, accuracy = classifier.evaluate(x_val, y_val, verbose=0)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(model_path)
    print("Saved landmark classifier:", model_path)
    print("Validation accuracy={:.1%}, loss={:.4f} (ảnh test độc lập cần thiết để đánh giá thật)".format(
        accuracy, loss))


class LandmarkAI:
    """User-trained classifier operating on MediaPipe's 21 detected hand joints."""
    def __init__(self, classifier_path, detector_path, is_image=False):
        if tf is None:
            raise RuntimeError("Thiếu TensorFlow. Cài requirements-train.txt để dùng classifier landmark.")
        classifier_path = Path(classifier_path)
        if not classifier_path.is_file():
            raise RuntimeError("Chưa có model landmark {}. Hãy train trước với --train --data dataset.".format(
                classifier_path))
        self.classifier = tf.keras.models.load_model(classifier_path)
        self.detector = HandLandmarkDetector(detector_path, is_image=is_image)

    def predict(self, frame, threshold):
        result = self.detector.predict(frame)
        return result, predicted_hands(self.classifier, result, threshold)

    def close(self):
        self.detector.close()


def draw_landmark_result(frame, result, predictions):
    height, width = frame.shape[:2]
    for index, hand in enumerate(result.hand_landmarks):
        points = [(int(p.x * width), int(p.y * height)) for p in hand]
        for start, end in LANDMARK_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (70, 220, 80), 2, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(frame, (x, y), 4, (30, 40, 245), -1, cv2.LINE_AA)
        if index < len(predictions):
            label, score, handed = predictions[index]
            x = max(8, min(width - 220, points[0][0]))
            y = max(28, points[0][1] - 12)
            text = "{}: {} ({:.0f}%)".format(handed, label, score * 100)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0, 255, 255), 2, cv2.LINE_AA)


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
    parser.add_argument("--train", action="store_true", help="Trích 21 landmark từ ảnh dataset và train classifier riêng")
    parser.add_argument("--data", default="dataset", help="Dataset có thư mục fist/palm/peace/thumbs_up/none")
    parser.add_argument("--epochs", type=int, default=40)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--image", help="Ảnh đơn cần phân loại")
    source.add_argument("--camera", type=int, help="Webcam máy tính, thường là 0")
    source.add_argument("--url", help="ESP32 MJPEG URL, ví dụ http://192.168.4.1:81/stream")
    parser.add_argument("--mode", choices=("both", "cnn", "landmark"), default="both",
                        help="both: CNN ảnh + classifier landmark tự train; cnn: chỉ CNN; landmark: chỉ landmark classifier")
    parser.add_argument("--model", default="model_outline.tflite", help="Đường dẫn model CNN ảnh .tflite")
    parser.add_argument("--landmark-classifier", default="model_landmark.keras",
                        help="Model classifier landmark do --train tạo")
    parser.add_argument("--landmark-model", default="models/hand_landmarker.task",
                        help="Model MediaPipe phát hiện khớp (tự tải nếu chưa có)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Ngưỡng tin cậy MediaPipe (mặc định 0.55)")
    parser.add_argument("--no-preview", action="store_true", help="Chỉ in kết quả ảnh, không mở cửa sổ xem trước")
    args = parser.parse_args()

    if args.train:
        try:
            train_landmark_model(args.data, args.landmark_classifier, args.landmark_model, args.epochs)
            return 0
        except (OSError, ValueError, RuntimeError) as exc:
            print("Lỗi train landmark:", exc, file=sys.stderr)
            return 2
    if not (args.image or args.camera is not None or args.url):
        parser.error("Cần --image, --camera, --url, hoặc dùng --train để train model landmark.")

    use_cnn = args.mode in ("both", "cnn")
    use_landmarks = args.mode in ("both", "landmark")
    model = None
    landmark = None
    cap = None
    frames = None
    try:
        if use_cnn:
            model = GestureModel(args.model)
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                parser.error(f"Không đọc được ảnh: {args.image}")
            result = None
            predictions = []
            if model is not None:
                print_prediction(model, frame, args.image)
            if use_landmarks:
                landmark = LandmarkAI(args.landmark_classifier, args.landmark_model, is_image=True)
                result, predictions = landmark.predict(frame, args.threshold)
                if not predictions:
                    print("{}: MediaPipe không phát hiện bàn tay; kết quả NONE".format(args.image))
                for label, score, handed in predictions:
                    label_index = LABELS.index(label)
                    print("  Landmark AI {}: {} / {} ({:.1%})".format(
                        handed, label, LABELS_VI[label_index], score))
            if not args.no_preview:
                preview = frame.copy()
                if model is not None:
                    index, confidence, _ = model.predict(frame)
                    cv2.putText(preview, "CNN: {} ({:.0%})".format(LABELS_VI[index], confidence), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    draw_outline_preview(preview, model, frame)
                if result is not None:
                    draw_landmark_result(preview, result, predictions)
                cv2.imshow("Gesture AI - image", preview)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if landmark is not None:
                landmark.close()
                landmark = None
            return 0

        if use_landmarks:
            landmark = LandmarkAI(args.landmark_classifier, args.landmark_model, is_image=False)
        if args.camera is not None:
            cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError(f"Không mở được webcam {args.camera}; thử --camera 1.")
            frames = None
        else:
            cap = None
            frames = frames_from_mjpeg(args.url)

        print("Nhấn Q/ESC để thoát; nhấn I để chọn ảnh. CNN và landmark classifier dùng 5/7 frame ổn định.")
        recent = deque(maxlen=STABLE_WINDOW)
        landmark_histories = [deque(maxlen=STABLE_WINDOW) for _ in range(2)]
        try:
            while True:
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Không đọc được khung hình từ webcam.")
                else:
                    frame = next(frames)
                display = frame.copy()
                if model is not None:
                    index, confidence, scores = model.predict(frame)
                    recent.append((index, confidence, scores))
                    counts = np.bincount([item[0] for item in recent], minlength=len(LABELS))
                    candidate = int(np.argmax(counts))
                    candidate_scores = [item[1] for item in recent if item[0] == candidate]
                    confirmed = (len(recent) == STABLE_WINDOW and counts[candidate] >= REQUIRED_MATCHES and
                                 float(np.mean(candidate_scores)) >= MIN_LIVE_CONFIDENCE)
                    if confirmed:
                        cnn_text = "CNN: {} ({:.0%})".format(LABELS_VI[candidate], np.mean(candidate_scores))
                        color = (0, 255, 0)
                    else:
                        cnn_text = "CNN: dang nhan dang {} ({:.0%}) [{}/{}]".format(
                            LABELS_VI[index], confidence, counts[index], STABLE_WINDOW)
                        color = (0, 200, 255)
                    cv2.putText(display, cnn_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
                    draw_outline_preview(display, model, frame)
                if landmark is not None:
                    landmark_result, landmark_predictions = landmark.predict(frame, args.threshold)
                    stable_predictions = []
                    for i, (label, score, handed) in enumerate(landmark_predictions):
                        history = landmark_histories[min(i, 1)]
                        history.append(label)
                        labels = list(history)
                        stable = max(set(labels), key=labels.count) if labels else label
                        if labels.count(stable) < 4:
                            stable = "..."
                        stable_predictions.append((stable, score, handed))
                    for index in range(len(landmark_predictions), len(landmark_histories)):
                        landmark_histories[index].clear()
                    if landmark_result.hand_landmarks:
                        draw_landmark_result(display, landmark_result, stable_predictions)
                    else:
                        cv2.putText(display, "Landmark: khong thay ban tay", (10, 30 if model is None else 58),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 80, 255), 2)
                cv2.putText(display, "I: chon anh | Q: thoat", (10, display.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
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
                            if model is not None:
                                print_prediction(model, photo, path)
                            still_landmark = None
                            still_result = None
                            still_predictions = []
                            if landmark is not None:
                                still_landmark = LandmarkAI(args.landmark_classifier, args.landmark_model,
                                                            is_image=True)
                                still_result, still_predictions = still_landmark.predict(photo, args.threshold)
                                for label, score, handed in still_predictions:
                                    label_index = LABELS.index(label)
                                    print("  Landmark AI {}: {} / {} ({:.1%})".format(
                                        handed, label, LABELS_VI[label_index], score))
                                if not still_predictions:
                                    print("  Landmark AI: không phát hiện bàn tay, kết quả NONE")
                            photo_display = photo.copy()
                            if model is not None:
                                photo_index, photo_conf, _ = model.predict(photo)
                                cv2.putText(photo_display, "CNN: {} ({:.0%})".format(
                                    LABELS_VI[photo_index], photo_conf), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                                draw_outline_preview(photo_display, model, photo)
                            if still_result is not None:
                                draw_landmark_result(photo_display, still_result, still_predictions)
                            cv2.imshow("Gesture AI - selected image", photo_display)
                            cv2.waitKey(0)
                            cv2.destroyWindow("Gesture AI - selected image")
                            if still_landmark is not None:
                                still_landmark.close()
                    # The next loop iteration recreates the live preview and continues the stream.
                    recent.clear()
        finally:
            if cap is not None:
                cap.release()
            if frames is not None:
                frames.close()
            if landmark is not None:
                landmark.close()
                landmark = None
            cv2.destroyAllWindows()
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        if landmark is not None:
            landmark.close()
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
