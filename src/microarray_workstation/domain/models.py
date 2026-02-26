from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Spot:
    x: float
    y: float
    radius: float
    score: float = 0.0


@dataclass
class SpotMeasurement:
    row: int
    col: int
    x: float
    y: float
    radius: float
    foreground_mean: float
    foreground_median: float
    background_mean: float
    background_median: float
    net_mean: float
    net_median: float
    snr: float
    saturated_pct: float
    flag: str = "OK"


@dataclass
class AnalysisResult:
    image_path: str
    rows: int
    cols: int
    measurements: list[SpotMeasurement]
    metadata: dict[str, Any]
