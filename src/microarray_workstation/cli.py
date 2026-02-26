from __future__ import annotations

import argparse
import json
from pathlib import Path

from microarray_workstation.integration.lims_client import LIMSClient
from microarray_workstation.io.exporters import export_json
from microarray_workstation.workflows.analysis_workflow import (
    analyze_batch_images,
    analyze_one_image,
    list_summary_payloads,
)


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

    submit_batch = sub.add_parser("submit-lims-batch", help="Submit all *_summary.json in directory to LIMS")
    submit_batch.add_argument("--base-url", required=True)
    submit_batch.add_argument("--endpoint", required=True)
    submit_batch.add_argument("--input-dir", required=True, help="Directory containing *_summary.json")
    submit_batch.add_argument("--token", required=False)
    submit_batch.add_argument("--output", required=False, help="Output summary json path")

    return parser


def _cmd_analyze(args: argparse.Namespace) -> int:
    out = analyze_one_image(
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
    rows, summary_csv, summary_json = analyze_batch_images(
        input_dir=args.input_dir,
        rows=args.rows,
        cols=args.cols,
        template_path=args.template,
        output_dir=args.output_dir,
        channel=args.channel,
        ai_model=args.ai_model,
    )

    for row in rows:
        print(f"processed={row['image']}")

    print(f"batch_summary_csv={summary_csv}")
    print(f"batch_summary_json={summary_json}")
    return 0


def _cmd_submit_lims(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.json).read_text(encoding="utf-8"))
    client = LIMSClient(base_url=args.base_url, token=args.token)
    resp = client.submit_result(args.endpoint, payload)
    print(resp)
    return 0


def _cmd_submit_lims_batch(args: argparse.Namespace) -> int:
    payloads = list_summary_payloads(args.input_dir)
    client = LIMSClient(base_url=args.base_url, token=args.token)
    result = client.submit_batch_results(endpoint=args.endpoint, payloads=payloads)

    out_path = Path(args.output) if args.output else Path(args.input_dir) / "lims_submit_summary.json"
    export_json(result, out_path)
    print(f"lims_submit_summary={out_path}")
    print(f"total={result['total']} success={result['success']} failed={result['failed']}")
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
    if args.command == "submit-lims-batch":
        return _cmd_submit_lims_batch(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
