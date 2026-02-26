from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile


def load_image(path: str | Path, channel: int | None = None) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    if p.suffix.lower() in {".tif", ".tiff"}:
        img = tifffile.imread(str(p))
    else:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to load image: {p}")

    if img.ndim == 3:
        idx = 0 if channel is None else channel
        if idx < 0 or idx >= img.shape[-1]:
            raise ValueError(f"Invalid channel {idx}, image has {img.shape[-1]} channels")
        img = img[..., idx]

    return np.asarray(img)


def normalize_to_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError("Expected grayscale image")

    arr = gray.astype(np.float32)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)
