from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile


def _to_analysis_gray(img: np.ndarray, channel: int | None = None) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError("Expected color image")

    channels = img.shape[-1]
    if channel is not None:
        if channel < 0 or channel >= channels:
            raise ValueError(f"Invalid channel {channel}, image has {channels} channels")
        return img[..., channel]

    rgb = img[..., :3].astype(np.float32)
    if rgb.shape[-1] < 3:
        return np.max(rgb, axis=2)

    # Pseudo-color robust conversion:
    # value channel preserves hot spots, luminance keeps structural contrast.
    value = np.max(rgb, axis=2)
    luminance = rgb[..., 0] * 0.114 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.299
    merged = luminance * 0.85 + value * 0.15
    return merged.astype(img.dtype if np.issubdtype(img.dtype, np.integer) else np.float32)


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
        img = _to_analysis_gray(np.asarray(img), channel=channel)

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
