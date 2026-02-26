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
        spots.append(
            Spot(
                x=float(kp.pt[0]),
                y=float(kp.pt[1]),
                radius=float(max(kp.size / 2, min_radius)),
                score=float(kp.response),
            )
        )

    if spots:
        return spots

    dilated = cv2.dilate(preprocessed, np.ones((3, 3), dtype=np.uint8))
    maxima = (preprocessed == dilated) & (preprocessed > np.percentile(preprocessed, 95))
    ys, xs = np.where(maxima)
    return [Spot(x=float(x), y=float(y), radius=float(min_radius), score=float(preprocessed[y, x])) for y, x in zip(ys, xs)]


def _filter_spots_by_spacing(
    spots: list[Spot],
    spacing_min_px: float = 0.0,
    spacing_max_px: float = 0.0,
) -> list[Spot]:
    if len(spots) < 3 or (spacing_min_px <= 0 and spacing_max_px <= 0):
        return spots

    xs = np.array([s.x for s in spots], dtype=np.float32)
    ys = np.array([s.y for s in spots], dtype=np.float32)
    pts = np.stack([xs, ys], axis=1)
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nearest = np.sqrt(np.min(d2, axis=1))

    kept: list[Spot] = []
    for spot, nn in zip(spots, nearest):
        if spacing_min_px > 0 and nn < spacing_min_px:
            continue
        if spacing_max_px > 0 and nn > spacing_max_px:
            continue
        kept.append(spot)
    return kept if len(kept) >= 4 else spots


def infer_regular_grid(
    spots: list[Spot],
    rows: int,
    cols: int,
    image_shape: tuple[int, int],
    spacing_min_px: float = 0.0,
    spacing_max_px: float = 0.0,
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

    filtered = _filter_spots_by_spacing(spots, spacing_min_px=spacing_min_px, spacing_max_px=spacing_max_px)
    expected = rows * cols
    sorted_spots = sorted(filtered, key=lambda s: s.score, reverse=True)
    selected = sorted_spots[: max(expected * 2, expected)]

    xs = np.array([s.x for s in selected], dtype=np.float32)
    ys = np.array([s.y for s in selected], dtype=np.float32)
    rs = np.array([s.radius for s in selected], dtype=np.float32)

    min_x, max_x = float(np.percentile(xs, 2.0)), float(np.percentile(xs, 98.0))
    min_y, max_y = float(np.percentile(ys, 2.0)), float(np.percentile(ys, 98.0))
    center_x = float(np.median(xs))
    center_y = float(np.median(ys))

    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    pitch_x = span_x / max(cols - 1, 1)
    pitch_y = span_y / max(rows - 1, 1)

    if spacing_min_px > 0:
        pitch_x = max(pitch_x, spacing_min_px)
        pitch_y = max(pitch_y, spacing_min_px)
    if spacing_max_px > 0:
        pitch_x = min(pitch_x, spacing_max_px)
        pitch_y = min(pitch_y, spacing_max_px)

    span_x = pitch_x * max(cols - 1, 1)
    span_y = pitch_y * max(rows - 1, 1)
    min_x = max(0.0, center_x - span_x / 2.0)
    max_x = min(float(w - 1), center_x + span_x / 2.0)
    min_y = max(0.0, center_y - span_y / 2.0)
    max_y = min(float(h - 1), center_y + span_y / 2.0)

    grid_x = np.linspace(min_x, max_x, num=cols, dtype=np.float32)
    grid_y = np.linspace(min_y, max_y, num=rows, dtype=np.float32)
    radius = float(np.median(rs)) if rs.size else min(w / cols, h / rows) * 0.2

    inferred: list[Spot] = []
    for r in range(rows):
        for c in range(cols):
            x, y = float(grid_x[c]), float(grid_y[r])
            inferred.append(Spot(x=x, y=y, radius=radius, score=1.0))
    return inferred


def refine_grid_by_local_peaks(gray: np.ndarray, grid: list[Spot], search_radius_px: int = 5) -> list[Spot]:
    if gray.ndim != 2:
        raise ValueError("Expected grayscale image")

    h, w = gray.shape
    arr = gray.astype(np.float32)
    refined: list[Spot] = []
    for spot in grid:
        x0 = int(round(spot.x))
        y0 = int(round(spot.y))
        x1 = max(0, x0 - search_radius_px)
        x2 = min(w, x0 + search_radius_px + 1)
        y1 = max(0, y0 - search_radius_px)
        y2 = min(h, y0 + search_radius_px + 1)

        window = arr[y1:y2, x1:x2]
        if window.size == 0:
            refined.append(spot)
            continue

        max_idx = int(np.argmax(window))
        wy, wx = np.unravel_index(max_idx, window.shape)
        refined.append(
            Spot(
                x=float(x1 + wx),
                y=float(y1 + wy),
                radius=spot.radius,
                score=float(window[wy, wx]),
            )
        )
    return refined


def shift_grid(grid: list[Spot], dx: float = 0.0, dy: float = 0.0) -> list[Spot]:
    return [Spot(x=s.x + dx, y=s.y + dy, radius=s.radius, score=s.score) for s in grid]
