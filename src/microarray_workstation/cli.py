from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from microarray_workstation.analysis.ai_classifier import classify_spot_quality
from microarray_workstation.analysis.image_loader import load_image
from microarray_workstation.analysis.pipeline import run_analysis, to_dataframe
from microarray_workstation.integration.lims_client import LIMSClient
from microarray_workstation.io.exporters import export_dataframe_csv, export_json
from microarray_workstation.rules.interpreter import interpret, load_template, summarize_calls


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microarray-cli")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run microarray analysis")
    analyze.add_argument("--image", required=True, help="Path to image file")
    analyze.add_argument("--rows", required=True, type=int)
    analyze.add_argument("--cols", required=True, type=int)
    analyze.add_argument("--template", required=False)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--channel", required=False, type=int)
    analyze.add_argument("--ai-model", required=False, help="Optional ONNX model path")

    batch = sub.add_parser("analyze-batch", help="Batch analyze all chip images in a directory")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--rows", required=True, type=int)
    batch.add_argument("--cols", required=True, type=int)
    batch.add_argument("--template", required=False)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--channel", required=False, type=int)
    batch.add_argument("--ai-model", required=False)

    submit = sub.add_parser("submit-lims", help="Submit interpreted result to LIMS")
    submit.add_argument("--base-url", required=True)
    submit.add_argument("--endpoint", required=True)
    submit.add_argument("--json", required=True, help="Summary JSON path")
    submit.add_argument("--token", required=False)

    return parser


def _load_template_or_default(path: str | None) -> dict[str, Any]:
    if path:
        return load_template(path)
    return {"interpretation": {"snr_threshold": 2.0, "net_median_threshold": 200.0}}


def _analyze_one(
    image_path: str,
    rows: int,
    cols: int,
    template_path: str | None,
    output_dir: str,
    channel: int | None,
    ai_model: str | None,
) -> dict[str, Any]:
    result = run_analysis(image_path, rows=rows, cols=cols, channel=channel)
    df = to_dataframe(result)

    gray = load_image(image_path, channel=channel)
    ai_df, ai_summary = classify_spot_quality(gray, df, model_path=ai_model)

    tpl = _load_template_or_default(template_path)
    interpreted = interpret(ai_df, tpl)
    summary = summarize_calls(interpreted)

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
        "rows": result.rows,
        "cols": result.cols,
        "metadata": result.metadata,
        "summary": summary,
        "ai_summary": ai_summary,
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


def _cmd_analyze(args: argparse.Namespace) -> int:
    out = _analyze_one(
        image_path=args.image,
        rows=args.rows,
        cols=args.cols,
        template_path=args.template,
        output_dir=args.output_dir,
        channel=args.channel,
        ai_model=args.ai_model,
    )

    print(f"raw_csv={out['raw_csv']}")
    print(f"interpreted_csv={out['interpreted_csv']}")
    print(f"summary_json={out['summary_json']}")
    print(f"qc_status={out['qc_status']} ai_mode={out['ai_summary']['mode']}")
    return 0


def _cmd_analyze_batch(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise ValueError(f"No image files found in {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_rows = []
    for image in images:
        out = _analyze_one(
            image_path=str(image),
            rows=args.rows,
            cols=args.cols,
            template_path=args.template,
            output_dir=str(output_dir),
            channel=args.channel,
            ai_model=args.ai_model,
        )
        batch_rows.append(
            {
                "image": out["image"],
                "qc_status": out["qc_status"],
                "positive": out["summary"]["positive"],
                "negative": out["summary"]["negative"],
                "review": out["summary"]["review"],
                "ai_mode": out["ai_summary"]["mode"],
                "ai_mean_score": out["ai_summary"]["mean_ai_score"],
            }
        )
        print(f"processed={image}")

    batch_df = pd.DataFrame(batch_rows)
    summary_csv = output_dir / "batch_summary.csv"
    batch_json = output_dir / "batch_summary.json"
    export_dataframe_csv(batch_df, summary_csv)
    export_json({"total_images": len(batch_rows), "records": batch_rows}, batch_json)

    print(f"batch_summary_csv={summary_csv}")
    print(f"batch_summary_json={batch_json}")
    return 0


def _cmd_submit_lims(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    client = LIMSClient(base_url=args.base_url, token=args.token)
    resp = client.submit_result(args.endpoint, payload)
    print(resp)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "analyze-batch":
        return _cmd_analyze_batch(args)
    if args.command == "submit-lims":
        return _cmd_submit_lims(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
