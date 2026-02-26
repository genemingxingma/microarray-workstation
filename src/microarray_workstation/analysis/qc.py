from __future__ import annotations

from typing import Any

import pandas as pd


def compute_qc_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "mean_snr": 0.0,
            "median_net_median": 0.0,
            "saturated_rate_pct": 0.0,
            "low_snr_rate_pct": 0.0,
            "pass_rate_pct": 0.0,
            "qc_status": "FAIL",
        }

    mean_snr = float(df["snr"].mean())
    median_net_median = float(df["net_median"].median())
    saturated_rate_pct = float((df["flag"] == "SATURATED").mean() * 100.0)
    low_snr_rate_pct = float((df["flag"] == "LOW_SNR").mean() * 100.0)
    pass_rate_pct = float((df["flag"] == "OK").mean() * 100.0)

    qc_status = "PASS"
    if mean_snr < 1.8 or pass_rate_pct < 80.0 or saturated_rate_pct > 10.0:
        qc_status = "WARN"
    if mean_snr < 1.2 or pass_rate_pct < 60.0:
        qc_status = "FAIL"

    return {
        "mean_snr": round(mean_snr, 4),
        "median_net_median": round(median_net_median, 4),
        "saturated_rate_pct": round(saturated_rate_pct, 4),
        "low_snr_rate_pct": round(low_snr_rate_pct, 4),
        "pass_rate_pct": round(pass_rate_pct, 4),
        "qc_status": qc_status,
    }
