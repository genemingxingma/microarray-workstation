from __future__ import annotations

import math

import numpy as np

from microarray_workstation.domain.models import Spot, SpotMeasurement


def _circle_mask(h: int, w: int, cx: float, cy: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def _annulus_mask(h: int, w: int, cx: float, cy: float, r_in: float, r_out: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 >= r_in**2) & (d2 <= r_out**2)


def quantify_spots(
    gray: np.ndarray,
    spots: list[Spot],
    rows: int,
    cols: int,
    sample_spots: list[Spot] | None = None,
) -> list[SpotMeasurement]:
    h, w = gray.shape
    arr = gray.astype(np.float32)
    max_val = 65535.0 if gray.dtype in (np.uint16,) else float(np.max(arr) if np.max(arr) > 0 else 255.0)

    results: list[SpotMeasurement] = []
    for idx, spot in enumerate(spots):
        sample_spot = sample_spots[idx] if sample_spots and idx < len(sample_spots) else spot
        r = max(2.0, float(sample_spot.radius))

        fg_mask = _circle_mask(h, w, sample_spot.x, sample_spot.y, r)
        bg_mask = _annulus_mask(h, w, sample_spot.x, sample_spot.y, r * 1.5, r * 2.5)

        fg_vals = arr[fg_mask]
        bg_vals = arr[bg_mask]

        if fg_vals.size == 0:
            fg_vals = np.array([0.0], dtype=np.float32)
        if bg_vals.size == 0:
            bg_vals = np.array([0.0], dtype=np.float32)

        fg_mean = float(np.mean(fg_vals))
        fg_median = float(np.median(fg_vals))
        bg_mean = float(np.mean(bg_vals))
        bg_median = float(np.median(bg_vals))
        net_mean = fg_mean - bg_mean
        net_median = fg_median - bg_median
        snr = float(net_mean / (np.std(bg_vals) + 1e-6))
        saturated_pct = float(np.mean(fg_vals >= max_val * 0.99) * 100.0)

        row = idx // cols + 1
        col = idx % cols + 1
        flag = "SATURATED" if saturated_pct > 5.0 else "LOW_SNR" if snr < 1.5 else "OK"

        results.append(
            SpotMeasurement(
                row=row,
                col=col,
                x=float(spot.x),
                y=float(spot.y),
                signal_x=float(sample_spot.x),
                signal_y=float(sample_spot.y),
                radius=r,
                foreground_mean=fg_mean,
                foreground_median=fg_median,
                background_mean=bg_mean,
                background_median=bg_median,
                net_mean=net_mean,
                net_median=net_median,
                snr=snr,
                saturated_pct=saturated_pct,
                flag=flag,
            )
        )
    return results
