from __future__ import annotations

from pathlib import Path

import pandas as pd

from microarray_workstation.analysis.image_loader import load_image
from microarray_workstation.analysis.spot_detector import detect_spots, infer_regular_grid, preprocess
from microarray_workstation.analysis.quantification import quantify_spots
from microarray_workstation.domain.models import AnalysisResult


def run_analysis(
    image_path: str | Path,
    rows: int,
    cols: int,
    channel: int | None = None,
) -> AnalysisResult:
    gray = load_image(image_path, channel=channel)
    prep = preprocess(gray)
    spots = detect_spots(prep)
    grid = infer_regular_grid(spots, rows=rows, cols=cols, image_shape=gray.shape)
    measurements = quantify_spots(gray, grid, rows=rows, cols=cols)

    return AnalysisResult(
        image_path=str(image_path),
        rows=rows,
        cols=cols,
        measurements=measurements,
        metadata={
            "detected_candidates": len(spots),
            "grid_size": rows * cols,
            "channel": channel,
        },
    )


def to_dataframe(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for m in result.measurements:
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
