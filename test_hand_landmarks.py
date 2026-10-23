#!/usr/bin/env python3
"""Live hand skeleton + gesture recognition using MediaPipe Tasks."""
import argparse
import collections
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
GESTURE_NAMES = {
    "Closed_Fist": "FIST",
    "Open_Palm": "PALM",
    "Victory": "PEACE",
    "Thumb_Up": "THUMBS UP",
}


def ensure_model(path):
    path = Path(path)
    if path.is_file() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading MediaPipe gesture model to", path)
    try:
        urllib.request.urlretrieve(MODEL_URL, str(path))
    except Exception as exc:
        raise RuntimeError(
            "Không tải được model. Kiểm tra Internet hoặc tải model từ URL trong "
            "README.md rồi đặt vào " + str(path)
        ) from exc
    if path.stat().st_size < 100_000:
        path.unlink(missing_ok=True)
        raise RuntimeError("File model tải về không hợp lệ.")
    return path


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    # Palm/finger bones first, then joints, so all 21 points remain visible.
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (70, 220, 80), 2, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (30, 40, 245), -1, cv2.LINE_AA)


def top_gesture(result, hand_index, threshold):
    categories = result.gestures[hand_index] if hand_index < len(result.gestures) else []
    if not categories:
        return "NONE", 0.0
    category = categories[0]
    if category.score < threshold or category.category_name == "None":
        return "NONE", float(category.score)
    return GESTURE_NAMES.get(category.category_name, "NONE"), float(category.score)


def main():
    parser = argparse.ArgumentParser(
        description="Vẽ 21 khớp bàn tay và nhận dạng cử chỉ bằng webcam hoặc ESP32 MJPEG."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam laptop (mặc định: 0)")
    parser.add_argument("--url", help="MJPEG URL, ví dụ http://192.168.4.1:81/stream")
    parser.add_argument("--image", help="Chỉ nhận dạng một ảnh tĩnh")
    parser.add_argument("--model", default="models/gesture_recognizer.task")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--window", default="Hand landmarks | Q/Esc: thoat")
    args = parser.parse_args()

    try:
        model_path = ensure_model(args.model)
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 2

    vision = mp.tasks.vision
    options = vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=vision.RunningMode.IMAGE if args.image else vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    recognizer = vision.GestureRecognizer.create_from_options(options)
    capture = None
    try:
        if args.image:
            frame = cv2.imread(args.image)
            if frame is None:
                raise RuntimeError("Không mở được ảnh: " + args.image)
            frames = [(frame, None)]
        else:
            source = args.url if args.url else args.camera
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError("Không mở được camera/stream: " + str(source))
            frames = None

        histories = [collections.deque(maxlen=7) for _ in range(2)]
        frame_index = 0
        while True:
            if frames is not None:
                if not frames:
                    break
                frame, _ = frames.pop(0)
            else:
                ok, frame = capture.read()
                if not ok:
                    print("Không đọc được frame; kiểm tra camera hoặc URL stream.")
                    break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if args.image:
                result = recognizer.recognize(mp_image)
            else:
                timestamp_ms = int(time.monotonic() * 1000)
                result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            if not result.hand_landmarks:
                cv2.putText(frame, "NO HAND DETECTED", (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 80, 255), 2, cv2.LINE_AA)
            for i, landmarks in enumerate(result.hand_landmarks):
                draw_hand(frame, landmarks)
                label, score = top_gesture(result, i, args.threshold)
                histories[min(i, 1)].append(label)
                # Short temporal vote prevents single-frame flicker.
                recent = list(histories[min(i, 1)])
                stable = max(set(recent), key=recent.count) if recent else label
                if recent.count(stable) < 4:
                    stable = "..."
                handed = "Hand"
                if i < len(result.handedness) and result.handedness[i]:
                    handed = result.handedness[i][0].display_name or result.handedness[i][0].category_name
                x = max(8, min(frame.shape[1] - 220, int(landmarks[0].x * frame.shape[1])))
                y = max(28, int(landmarks[0].y * frame.shape[0]) - 12)
                text = "{} {} {:.0f}%".format(handed, stable, score * 100)
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, "LEFT/RIGHT USE SAME MODEL | Q: quit", (12, frame.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(args.window, frame)
            frame_index += 1
            key = cv2.waitKey(1 if not args.image else 0) & 0xFF
            if key in (ord("q"), 27):
                break
            if args.image:
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 2
    finally:
        if capture is not None:
            capture.release()
        recognizer.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
