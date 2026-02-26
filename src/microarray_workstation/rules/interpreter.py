from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_template(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Template must be a YAML mapping")
    return data


def interpret(df: pd.DataFrame, template: dict[str, Any]) -> pd.DataFrame:
    cfg = template.get("interpretation", {})
    snr_threshold = float(cfg.get("snr_threshold", 2.0))
    net_threshold = float(cfg.get("net_median_threshold", 200.0))

    result = df.copy()
    result["call"] = "NEGATIVE"
    mask = (result["snr"] >= snr_threshold) & (result["net_median"] >= net_threshold)
    result.loc[mask, "call"] = "POSITIVE"
    result.loc[result["flag"] == "SATURATED", "call"] = "REVIEW"

    annotation = template.get("layout", {}).get("annotation", {})
    if annotation:
        result["target"] = result.apply(
            lambda row: annotation.get(f"R{int(row['row'])}C{int(row['col'])}", "UNASSIGNED"),
            axis=1,
        )
    else:
        result["target"] = "UNASSIGNED"

    return result


def summarize_calls(interpreted: pd.DataFrame) -> dict[str, Any]:
    return {
        "total": int(len(interpreted)),
        "positive": int((interpreted["call"] == "POSITIVE").sum()),
        "negative": int((interpreted["call"] == "NEGATIVE").sum()),
        "review": int((interpreted["call"] == "REVIEW").sum()),
    }
