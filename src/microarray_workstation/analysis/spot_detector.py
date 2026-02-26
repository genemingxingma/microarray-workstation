from __future__ import annotations

import math

import cv2
import numpy as np

from microarray_workstation.domain.models import Spot


def preprocess(gray: np.ndarray) -> np.ndarray:
    img = gray.astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    bg = cv2.GaussianBlur(img, (0, 0), 8.0)
    corrected = cv2.subtract(img, bg)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def detect_spots(preprocessed: np.ndarray, min_radius: float = 2.0, max_radius: float = 12.0) -> list[Spot]:
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = math.pi * min_radius * min_radius
    params.maxArea = math.pi * max_radius * max_radius
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(preprocessed)

    spots: list[Spot] = []
    for kp in keypoints:
        spots.append(Spot(x=float(kp.pt[0]), y=float(kp.pt[1]), radius=float(max(kp.size / 2, min_radius)), score=float(kp.response)))

    if spots:
        return spots

    # Fallback using local maxima for faint images
    dilated = cv2.dilate(preprocessed, np.ones((3, 3), dtype=np.uint8))
    maxima = (preprocessed == dilated) & (preprocessed > np.percentile(preprocessed, 95))
    ys, xs = np.where(maxima)
    return [Spot(x=float(x), y=float(y), radius=float(min_radius), score=float(preprocessed[y, x])) for y, x in zip(ys, xs)]


def infer_regular_grid(
    spots: list[Spot],
    rows: int,
    cols: int,
    image_shape: tuple[int, int],
) -> list[Spot]:
    h, w = image_shape
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    if not spots:
        dx = w / max(cols, 1)
        dy = h / max(rows, 1)
        return [
            Spot(x=(c + 0.5) * dx, y=(r + 0.5) * dy, radius=min(dx, dy) * 0.25)
            for r in range(rows)
            for c in range(cols)
        ]

    xs = np.array([s.x for s in spots], dtype=np.float32)
    ys = np.array([s.y for s in spots], dtype=np.float32)
    rs = np.array([s.radius for s in spots], dtype=np.float32)

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    grid_x = np.linspace(min_x, max_x, num=cols, dtype=np.float32)
    grid_y = np.linspace(min_y, max_y, num=rows, dtype=np.float32)
    radius = float(np.median(rs)) if rs.size else min(w / cols, h / rows) * 0.2

    inferred: list[Spot] = []
    for r in range(rows):
        for c in range(cols):
            x, y = float(grid_x[c]), float(grid_y[r])
            inferred.append(Spot(x=x, y=y, radius=radius, score=1.0))
    return inferred
