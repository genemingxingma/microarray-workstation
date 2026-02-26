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
) -> AnalysisResult:
    gray = load_image(image_path, channel=channel)
    prep = preprocess(gray)
    spots = detect_spots(prep)
    grid = infer_regular_grid(spots, rows=rows, cols=cols, image_shape=gray.shape)
    step_guess = int(max(5, min(gray.shape[1] / max(cols, 1), gray.shape[0] / max(rows, 1)) * 0.35))
    grid = refine_grid_by_local_peaks(gray, grid, search_radius_px=step_guess)
    if grid_shift != (0.0, 0.0):
        grid = shift_grid(grid, dx=float(grid_shift[0]), dy=float(grid_shift[1]))
    measurements = quantify_spots(gray, grid, rows=rows, cols=cols)
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
