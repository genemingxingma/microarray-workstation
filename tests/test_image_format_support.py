from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from microarray_workstation.analysis.pipeline import run_analysis, to_dataframe


def _synthetic_array(rows: int = 8, cols: int = 12) -> np.ndarray:
    h, w = 360, 520
    img = np.random.normal(loc=100, scale=10, size=(h, w)).astype(np.float32)
    ys = np.linspace(30, h - 30, rows)
    xs = np.linspace(30, w - 30, cols)
    for y in ys:
        for x in xs:
            rr, cc = np.ogrid[:h, :w]
            mask = (cc - x) ** 2 + (rr - y) ** 2 <= 6**2
            img[mask] += 260
    return np.clip(img, 0, 255).astype(np.uint8)


def test_pipeline_supports_png_jpg_jpeg(tmp_path: Path) -> None:
    arr = _synthetic_array()
    paths = [
        tmp_path / "chip_png.png",
        tmp_path / "chip_jpg.jpg",
        tmp_path / "chip_jpeg.jpeg",
    ]

    for p in paths:
        ok = cv2.imwrite(str(p), arr)
        assert ok, f"Failed to write test image: {p}"

        result = run_analysis(p, rows=8, cols=12)
        df = to_dataframe(result)
        assert len(df) == 96
        assert "net_median" in df.columns


def test_pipeline_supports_pseudocolor_png(tmp_path: Path) -> None:
    arr = _synthetic_array()
    pseudo = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    p = tmp_path / "chip_pseudocolor.png"
    ok = cv2.imwrite(str(p), pseudo)
    assert ok

    result = run_analysis(p, rows=8, cols=12)
    df = to_dataframe(result)
    assert len(df) == 96
    assert float((df["net_median"] > 0).mean()) > 0.35
    assert float(df["snr"].mean()) > 0.0
