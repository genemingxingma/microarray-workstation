from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from microarray_workstation.analysis.pipeline import run_analysis, to_dataframe


def _synthetic_microarray(path: Path, rows: int = 8, cols: int = 12) -> None:
    h, w = 512, 768
    img = np.random.normal(loc=200, scale=20, size=(h, w)).astype(np.float32)

    ys = np.linspace(40, h - 40, rows)
    xs = np.linspace(40, w - 40, cols)
    for y in ys:
        for x in xs:
            rr, cc = np.ogrid[:h, :w]
            mask = (cc - x) ** 2 + (rr - y) ** 2 <= 8**2
            img[mask] += 500

    img = np.clip(img, 0, 65535).astype(np.uint16)
    tifffile.imwrite(path, img)


def test_run_analysis(tmp_path: Path) -> None:
    image_path = tmp_path / "chip.tif"
    _synthetic_microarray(image_path)

    result = run_analysis(image_path, rows=8, cols=12)
    df = to_dataframe(result)

    assert len(df) == 96
    assert "net_median" in df.columns
    assert (df["net_median"] > 0).mean() > 0.8
