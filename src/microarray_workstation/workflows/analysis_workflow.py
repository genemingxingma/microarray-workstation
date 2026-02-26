from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from microarray_workstation.analysis.ai_classifier import classify_spot_quality
from microarray_workstation.analysis.image_loader import load_image
from microarray_workstation.analysis.pipeline import run_analysis, to_dataframe
from microarray_workstation.io.exporters import export_dataframe_csv, export_json
from microarray_workstation.rules.interpreter import interpret, load_template, summarize_calls

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def load_template_or_default(path: str | None) -> dict[str, Any]:
    if path:
        return load_template(path)
    return {"interpretation": {"snr_threshold": 2.0, "net_median_threshold": 200.0}}


def analyze_one_image(
    image_path: str,
    rows: int,
    cols: int,
    template_path: str | None,
    output_dir: str,
    channel: int | None,
    ai_model: str | None,
    spot_diameter_min_px: float = 4.0,
    spot_diameter_max_px: float = 24.0,
    spacing_min_px: float = 0.0,
    spacing_max_px: float = 0.0,
    background_mode: str = "local",
    global_background_percentile: float = 20.0,
    low_snr_threshold: float = 1.5,
    saturation_threshold_pct: float = 5.0,
    low_net_threshold: float = 0.0,
) -> dict[str, Any]:
    result = run_analysis(
        image_path,
        rows=rows,
        cols=cols,
        channel=channel,
        spot_diameter_min_px=spot_diameter_min_px,
        spot_diameter_max_px=spot_diameter_max_px,
        spacing_min_px=spacing_min_px,
        spacing_max_px=spacing_max_px,
        background_mode=background_mode,
        global_background_percentile=global_background_percentile,
        low_snr_threshold=low_snr_threshold,
        saturation_threshold_pct=saturation_threshold_pct,
        low_net_threshold=low_net_threshold,
    )
    df = to_dataframe(result)

    gray = load_image(image_path, channel=channel)
    ai_df, ai_summary = classify_spot_quality(gray, df, model_path=ai_model)

    tpl = load_template_or_default(template_path)
    interpreted = interpret(ai_df, tpl)
    summary = summarize_calls(interpreted)
    accession = Path(image_path).stem
    interface_results = []
    for _, row in interpreted.iterrows():
        service_code = str(row.get("target") or "").strip()
        if not service_code or service_code == "UNASSIGNED":
            service_code = f"R{int(row['row'])}C{int(row['col'])}"
        result_text = str(row.get("call") or "")
        note = (
            f"snr={float(row.get('snr', 0.0)):.3f};"
            f"net_median={float(row.get('net_median', 0.0)):.3f};"
            f"ai={float(row.get('ai_score', 0.0)):.3f};"
            f"flag={row.get('flag', '')}"
        )
        interface_results.append({"service_code": service_code, "result": result_text, "note": note})

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    raw_csv = out_dir / f"{stem}_raw.csv"
    interpreted_csv = out_dir / f"{stem}_interpreted.csv"
    summary_json = out_dir / f"{stem}_summary.json"

    export_dataframe_csv(ai_df, raw_csv)
    export_dataframe_csv(interpreted, interpreted_csv)

    payload = {
        "image": result.image_path,
        "accession": accession,
        "rows": result.rows,
        "cols": result.cols,
        "metadata": result.metadata,
        "summary": summary,
        "ai_summary": ai_summary,
        "results": interface_results,
    }
    export_json(payload, summary_json)

    return {
        "image": image_path,
        "raw_csv": str(raw_csv),
        "interpreted_csv": str(interpreted_csv),
        "summary_json": str(summary_json),
        "summary": summary,
        "ai_summary": ai_summary,
        "qc_status": result.metadata.get("qc", {}).get("qc_status", "NA"),
    }


def analyze_batch_images(
    input_dir: str,
    rows: int,
    cols: int,
    template_path: str | None,
    output_dir: str,
    channel: int | None,
    ai_model: str | None,
    spot_diameter_min_px: float = 4.0,
    spot_diameter_max_px: float = 24.0,
    spacing_min_px: float = 0.0,
    spacing_max_px: float = 0.0,
    background_mode: str = "local",
    global_background_percentile: float = 20.0,
    low_snr_threshold: float = 1.5,
    saturation_threshold_pct: float = 5.0,
    low_net_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], str, str]:
    source = Path(input_dir)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Input directory not found: {source}")

    images = [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise ValueError(f"No image files found in {source}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict[str, Any]] = []
    for image in images:
        out = analyze_one_image(
            image_path=str(image),
            rows=rows,
            cols=cols,
            template_path=template_path,
            output_dir=str(out_dir),
            channel=channel,
            ai_model=ai_model,
            spot_diameter_min_px=spot_diameter_min_px,
            spot_diameter_max_px=spot_diameter_max_px,
            spacing_min_px=spacing_min_px,
            spacing_max_px=spacing_max_px,
            background_mode=background_mode,
            global_background_percentile=global_background_percentile,
            low_snr_threshold=low_snr_threshold,
            saturation_threshold_pct=saturation_threshold_pct,
            low_net_threshold=low_net_threshold,
        )
        rows_out.append(
            {
                "image": out["image"],
                "qc_status": out["qc_status"],
                "positive": out["summary"]["positive"],
                "negative": out["summary"]["negative"],
                "review": out["summary"]["review"],
                "ai_mode": out["ai_summary"]["mode"],
                "ai_mean_score": out["ai_summary"]["mean_ai_score"],
                "summary_json": out["summary_json"],
            }
        )

    batch_df = pd.DataFrame(rows_out)
    summary_csv = out_dir / "batch_summary.csv"
    summary_json = out_dir / "batch_summary.json"
    export_dataframe_csv(batch_df, summary_csv)
    export_json({"total_images": len(rows_out), "records": rows_out}, summary_json)
    return rows_out, str(summary_csv), str(summary_json)


def list_summary_payloads(input_dir: str) -> list[tuple[str, dict[str, Any]]]:
    source = Path(input_dir)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Input directory not found: {source}")

    files = sorted(source.glob("*_summary.json"))
    if not files:
        raise ValueError(f"No *_summary.json files found in {source}")

    out: list[tuple[str, dict[str, Any]]] = []
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        out.append((str(f), payload))
    return out


def build_lab_interface_jobs_from_summaries(input_dir: str) -> list[tuple[str, str, dict[str, Any], str | None]]:
    summaries = list_summary_payloads(input_dir)
    jobs: list[tuple[str, str, dict[str, Any], str | None]] = []
    for source, data in summaries:
        accession = str(data.get("accession") or Path(source).stem.replace("_summary", ""))
        payload = {
            "accession": accession,
            "results": data.get("results") or [],
        }
        external_uid = f"MW-{accession}"
        jobs.append((source, "result", payload, external_uid))
    return jobs
