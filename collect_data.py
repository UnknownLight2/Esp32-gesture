#!/usr/bin/env python3
"""Lưu ảnh gán nhãn từ webcam laptop hoặc MJPEG stream của ESP32-S3.

Ví dụ webcam: python collect_data.py --camera 0 --label fist
Ví dụ ESP32:  python collect_data.py --url http://192.168.4.1:81/stream --label fist
Nhấn SPACE để lưu frame, Q hoặc ESC để thoát.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import requests

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

LABELS = ("fist", "palm", "peace", "thumbs_up", "none")


def frames_from_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Không mở được webcam index {index}. Hãy thử --camera 1 hoặc 2.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Webcam không trả được frame.")
            yield frame
    finally:
        cap.release()


def frames_from_mjpeg(url):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="URL MJPEG /stream của ESP32, ví dụ http://192.168.4.1:81/stream")
    source.add_argument("--camera", type=int, help="Index webcam laptop; thường dùng 0, thử 1 nếu không mở được")
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--out", default="dataset", help="Thư mục gốc dataset (mặc định: dataset)")
    parser.add_argument("--interval", type=float, default=0.35, help="Khoảng tối thiểu giữa hai ảnh đã lưu, giây")
    parser.add_argument("--mirror-preview", action="store_true", help="Lật hình xem trước; ảnh lưu vẫn giữ nguyên chiều webcam")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval phải lớn hơn hoặc bằng 0")

    folder = Path(args.out) / args.label
    folder.mkdir(parents=True, exist_ok=True)
    source_name = f"webcam {args.camera}" if args.camera is not None else args.url
    print("Nguồn:", source_name)
    print("Nhãn:", args.label, "| Lưu vào:", folder.resolve())
    print("SPACE lưu ảnh; Q/ESC thoát. Khung xanh là vùng giữa sẽ được dùng khi train.")

    count = len(list(folder.glob("*.jpg")))
    last_save = 0.0
    frames = None
    try:
        frames = frames_from_camera(args.camera) if args.camera is not None else frames_from_mjpeg(args.url)
        for frame in frames:
            shown = cv2.flip(frame, 1) if args.mirror_preview else frame.copy()
            height, width = shown.shape[:2]
            side = min(width, height)
            x0, y0 = (width - side) // 2, (height - side) // 2
            cv2.rectangle(shown, (x0, y0), (x0 + side - 1, y0 + side - 1), (0, 255, 0), 2)
            cv2.putText(shown, f"{args.label} | saved {count} | SPACE save / Q quit",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("Gesture sample collector", shown)
            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()
            if key == ord("q") or key == 27:
                break
            if key == 32 and now - last_save >= args.interval:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = folder / f"{args.label}_{stamp}.jpg"
                if cv2.imwrite(str(path), frame):
                    count += 1
                    last_save = now
                    print(f"Đã lưu {path} ({count})")
    except requests.RequestException as exc:
        print("Không đọc được MJPEG stream:", exc, file=sys.stderr)
        print("Kiểm tra URL và kết nối Wi-Fi với ESP32.", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    finally:
        if frames is not None and hasattr(frames, "close"):
            frames.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
