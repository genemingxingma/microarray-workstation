from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _extract_patch(gray: np.ndarray, x: float, y: float, radius: float) -> np.ndarray:
    h, w = gray.shape
    r = int(max(3, round(radius * 2)))
    cx, cy = int(round(x)), int(round(y))

    x1 = max(0, cx - r)
    x2 = min(w, cx + r + 1)
    y1 = max(0, cy - r)
    y2 = min(h, cy + r + 1)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((8, 8), dtype=np.float32)
    return patch.astype(np.float32)


def _heuristic_score(row: pd.Series) -> float:
    snr = float(row.get("snr", 0.0))
    net = float(row.get("net_median", 0.0))
    sat = float(row.get("saturated_pct", 0.0))

    score = 0.5
    score += min(snr / 10.0, 0.35)
    score += min(max(net, 0.0) / 5000.0, 0.25)
    score -= min(sat / 100.0, 0.3)
    return float(max(0.0, min(score, 1.0)))


def _label_from_score(score: float) -> str:
    if score >= 0.75:
        return "HIGH_CONF"
    if score >= 0.45:
        return "MEDIUM_CONF"
    return "LOW_CONF"


def _onnx_predict(gray: np.ndarray, df: pd.DataFrame, model_path: str) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name

    scores = []
    labels = []
    for _, row in df.iterrows():
        patch = _extract_patch(gray, float(row["x"]), float(row["y"]), float(row["radius"]))
        patch = patch.astype(np.float32)
        patch = patch - patch.min()
        denom = patch.max() if patch.max() > 0 else 1.0
        patch = patch / denom
        tensor = patch[np.newaxis, np.newaxis, :, :]
        pred = sess.run(None, {input_name: tensor})
        score = float(np.ravel(pred[0])[0])
        score = max(0.0, min(score, 1.0))
        scores.append(score)
        labels.append(_label_from_score(score))

    return np.array(scores, dtype=np.float32), np.array(labels, dtype=object)


def classify_spot_quality(
    gray: np.ndarray,
    quantified_df: pd.DataFrame,
    model_path: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = quantified_df.copy()
    mode = "heuristic"

    if model_path:
        try:
            scores, labels = _onnx_predict(gray, out, model_path)
            mode = "onnx"
        except Exception:
            scores = np.array([_heuristic_score(row) for _, row in out.iterrows()], dtype=np.float32)
            labels = np.array([_label_from_score(float(v)) for v in scores], dtype=object)
            mode = "heuristic_fallback"
    else:
        scores = np.array([_heuristic_score(row) for _, row in out.iterrows()], dtype=np.float32)
        labels = np.array([_label_from_score(float(v)) for v in scores], dtype=object)

    out["ai_score"] = scores
    out["ai_label"] = labels

    summary = {
        "mode": mode,
        "mean_ai_score": round(float(out["ai_score"].mean()) if not out.empty else 0.0, 4),
        "low_conf_count": int((out["ai_label"] == "LOW_CONF").sum()) if not out.empty else 0,
    }
    return out, summary
