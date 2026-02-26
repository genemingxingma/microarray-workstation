from __future__ import annotations

import argparse
from pathlib import Path

from microarray_workstation.analysis.pipeline import run_analysis, to_dataframe
from microarray_workstation.integration.lims_client import LIMSClient
from microarray_workstation.io.exporters import export_dataframe_csv, export_json
from microarray_workstation.rules.interpreter import interpret, load_template, summarize_calls


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

    submit = sub.add_parser("submit-lims", help="Submit interpreted result to LIMS")
    submit.add_argument("--base-url", required=True)
    submit.add_argument("--endpoint", required=True)
    submit.add_argument("--json", required=True, help="Summary JSON path")
    submit.add_argument("--token", required=False)

    return parser


def _cmd_analyze(args: argparse.Namespace) -> int:
    result = run_analysis(args.image, rows=args.rows, cols=args.cols, channel=args.channel)
    df = to_dataframe(result)

    if args.template:
        tpl = load_template(args.template)
    else:
        tpl = {"interpretation": {"snr_threshold": 2.0, "net_median_threshold": 200.0}}

    interpreted = interpret(df, tpl)
    summary = summarize_calls(interpreted)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    raw_csv = out_dir / f"{stem}_raw.csv"
    interpreted_csv = out_dir / f"{stem}_interpreted.csv"
    summary_json = out_dir / f"{stem}_summary.json"

    export_dataframe_csv(df, raw_csv)
    export_dataframe_csv(interpreted, interpreted_csv)
    export_json(
        {
            "image": result.image_path,
            "rows": result.rows,
            "cols": result.cols,
            "metadata": result.metadata,
            "summary": summary,
        },
        summary_json,
    )

    print(f"raw_csv={raw_csv}")
    print(f"interpreted_csv={interpreted_csv}")
    print(f"summary_json={summary_json}")
    return 0


def _cmd_submit_lims(args: argparse.Namespace) -> int:
    import json

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
    if args.command == "submit-lims":
        return _cmd_submit_lims(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
