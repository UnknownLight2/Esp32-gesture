"""Shared lightweight input preprocessing for the desktop train/test tools."""
import numpy as np


def apply_outline(gray_u8):
    """Blend grayscale with Sobel edges so contours are emphasized without mirroring."""
    gray = np.asarray(gray_u8, dtype=np.uint8)
    p = np.pad(gray.astype(np.int16), 1, mode="edge")
    gx = (-p[:-2, :-2] + p[:-2, 2:] - 2 * p[1:-1, :-2]
          + 2 * p[1:-1, 2:] - p[2:, :-2] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    edge = np.minimum(255, (np.abs(gx) + np.abs(gy) + 4) // 8)
    edge[[0, -1], :] = 0
    edge[:, [0, -1]] = 0
    return np.maximum((gray.astype(np.uint16) * 90) // 255, edge).astype(np.uint8)
