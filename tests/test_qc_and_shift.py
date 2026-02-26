from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from microarray_workstation.analysis.pipeline import run_analysis, to_dataframe
from microarray_workstation.analysis.qc import compute_qc_metrics


def _synthetic(path: Path, rows: int = 8, cols: int = 12) -> None:
    h, w = 480, 640
    img = np.random.normal(loc=120, scale=10, size=(h, w)).astype(np.float32)
    ys = np.linspace(40, h - 40, rows)
    xs = np.linspace(40, w - 40, cols)
    for y in ys:
        for x in xs:
            rr, cc = np.ogrid[:h, :w]
            mask = (cc - x) ** 2 + (rr - y) ** 2 <= 7**2
            img[mask] += 280
    tifffile.imwrite(path, np.clip(img, 0, 65535).astype(np.uint16))


def test_qc_metrics_has_status(tmp_path: Path) -> None:
    p = tmp_path / "chip.tif"
    _synthetic(p)
    result = run_analysis(p, rows=8, cols=12)
    df = to_dataframe(result)

    qc = compute_qc_metrics(df)
    assert qc["qc_status"] in {"PASS", "WARN", "FAIL"}
    assert "mean_snr" in qc


def test_grid_shift_changes_coordinates(tmp_path: Path) -> None:
    p = tmp_path / "chip_shift.tif"
    _synthetic(p)

    r1 = run_analysis(p, rows=8, cols=12, grid_shift=(0.0, 0.0))
    r2 = run_analysis(p, rows=8, cols=12, grid_shift=(3.0, -2.0))
    d1 = to_dataframe(r1)
    d2 = to_dataframe(r2)

    assert abs(float(d2["x"].mean() - d1["x"].mean())) > 1.0
    assert abs(float(d2["y"].mean() - d1["y"].mean())) > 1.0


def test_detection_params_are_accepted(tmp_path: Path) -> None:
    p = tmp_path / "chip_params.tif"
    _synthetic(p)

    r = run_analysis(
        p,
        rows=8,
        cols=12,
        spot_diameter_min_px=6.0,
        spot_diameter_max_px=20.0,
        spacing_min_px=8.0,
        spacing_max_px=80.0,
    )
    df = to_dataframe(r)
    assert len(df) == 96
    det = r.metadata.get("detection_params", {})
    assert det.get("spot_diameter_min_px") == 6.0
