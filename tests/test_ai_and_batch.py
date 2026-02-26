from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from microarray_workstation.analysis.ai_classifier import classify_spot_quality
from microarray_workstation.analysis.pipeline import run_analysis, to_dataframe
from microarray_workstation.cli import _cmd_analyze_batch, build_parser


def _synthetic(path: Path, rows: int = 8, cols: int = 12) -> None:
    h, w = 300, 420
    img = np.random.normal(loc=100, scale=8, size=(h, w)).astype(np.float32)
    ys = np.linspace(30, h - 30, rows)
    xs = np.linspace(30, w - 30, cols)
    for y in ys:
        for x in xs:
            rr, cc = np.ogrid[:h, :w]
            mask = (cc - x) ** 2 + (rr - y) ** 2 <= 5**2
            img[mask] += 240
    tifffile.imwrite(path, np.clip(img, 0, 65535).astype(np.uint16))


def test_ai_classifier_heuristic_mode(tmp_path: Path) -> None:
    p = tmp_path / "one.tif"
    _synthetic(p)

    result = run_analysis(p, rows=8, cols=12)
    df = to_dataframe(result)
    gray = tifffile.imread(p)

    out, summary = classify_spot_quality(gray, df, model_path=None)
    assert "ai_score" in out.columns
    assert "ai_label" in out.columns
    assert summary["mode"] == "heuristic"


def test_cli_analyze_batch(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    _synthetic(in_dir / "a.tif")
    _synthetic(in_dir / "b.tif")

    parser = build_parser()
    args = parser.parse_args(
        [
            "analyze-batch",
            "--input-dir",
            str(in_dir),
            "--rows",
            "8",
            "--cols",
            "12",
            "--output-dir",
            str(out_dir),
        ]
    )
    rc = _cmd_analyze_batch(args)
    assert rc == 0
    assert (out_dir / "batch_summary.csv").exists()
    assert (out_dir / "batch_summary.json").exists()
