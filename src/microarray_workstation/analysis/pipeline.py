from __future__ import annotations

from pathlib import Path

import pandas as pd

from microarray_workstation.analysis.image_loader import load_image
from microarray_workstation.analysis.qc import compute_qc_metrics
from microarray_workstation.analysis.quantification import quantify_spots
from microarray_workstation.analysis.spot_detector import (
    detect_spots,
    infer_regular_grid,
    preprocess,
    refine_grid_by_local_peaks,
    shift_grid,
)
from microarray_workstation.domain.models import AnalysisResult


def run_analysis(
    image_path: str | Path,
    rows: int,
    cols: int,
    channel: int | None = None,
    grid_shift: tuple[float, float] = (0.0, 0.0),
    spot_diameter_min_px: float = 4.0,
    spot_diameter_max_px: float = 24.0,
    spacing_min_px: float = 0.0,
    spacing_max_px: float = 0.0,
) -> AnalysisResult:
    gray = load_image(image_path, channel=channel)
    prep = preprocess(gray)
    min_radius = max(1.0, float(spot_diameter_min_px) / 2.0)
    max_radius = max(min_radius + 0.5, float(spot_diameter_max_px) / 2.0)
    spots = detect_spots(prep, min_radius=min_radius, max_radius=max_radius)
    grid = infer_regular_grid(
        spots,
        rows=rows,
        cols=cols,
        image_shape=gray.shape,
        spacing_min_px=float(spacing_min_px),
        spacing_max_px=float(spacing_max_px),
    )
    if spacing_max_px > 0:
        step_guess = int(max(3, min(spacing_max_px * 0.5, 32)))
    elif spacing_min_px > 0:
        step_guess = int(max(3, min(spacing_min_px * 0.7, 32)))
    else:
        step_guess = int(max(5, min(gray.shape[1] / max(cols, 1), gray.shape[0] / max(rows, 1)) * 0.35))
    refined = refine_grid_by_local_peaks(gray, grid, search_radius_px=step_guess)
    if grid_shift != (0.0, 0.0):
        grid = shift_grid(grid, dx=float(grid_shift[0]), dy=float(grid_shift[1]))
        refined = shift_grid(refined, dx=float(grid_shift[0]), dy=float(grid_shift[1]))
    measurements = quantify_spots(gray, grid, rows=rows, cols=cols, sample_spots=refined)
    df = _measurements_to_df(measurements)
    qc = compute_qc_metrics(df)

    return AnalysisResult(
        image_path=str(image_path),
        rows=rows,
        cols=cols,
        measurements=measurements,
        metadata={
            "detected_candidates": len(spots),
            "grid_size": rows * cols,
            "channel": channel,
            "grid_shift": {"dx": float(grid_shift[0]), "dy": float(grid_shift[1])},
            "detection_params": {
                "spot_diameter_min_px": float(spot_diameter_min_px),
                "spot_diameter_max_px": float(spot_diameter_max_px),
                "spacing_min_px": float(spacing_min_px),
                "spacing_max_px": float(spacing_max_px),
            },
            "grid_bbox": {
                "x_min": float(min(s.x for s in grid)) if grid else 0.0,
                "x_max": float(max(s.x for s in grid)) if grid else 0.0,
                "y_min": float(min(s.y for s in grid)) if grid else 0.0,
                "y_max": float(max(s.y for s in grid)) if grid else 0.0,
            },
            "qc": qc,
        },
    )


def _measurements_to_df(measurements) -> pd.DataFrame:
    rows = []
    for m in measurements:
        rows.append(
            {
                "row": m.row,
                "col": m.col,
                "x": m.x,
                "y": m.y,
                "signal_x": m.signal_x,
                "signal_y": m.signal_y,
                "radius": m.radius,
                "foreground_mean": m.foreground_mean,
                "foreground_median": m.foreground_median,
                "background_mean": m.background_mean,
                "background_median": m.background_median,
                "net_mean": m.net_mean,
                "net_median": m.net_median,
                "snr": m.snr,
                "saturated_pct": m.saturated_pct,
                "flag": m.flag,
            }
        )
    return pd.DataFrame(rows)


def to_dataframe(result: AnalysisResult) -> pd.DataFrame:
    return _measurements_to_df(result.measurements)
