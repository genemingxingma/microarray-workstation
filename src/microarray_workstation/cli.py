from __future__ import annotations

import argparse
import json
from pathlib import Path

from microarray_workstation.integration.lab_interface_client import LaboratoryManagementInterfaceClient
from microarray_workstation.integration.lims_client import LIMSClient
from microarray_workstation.io.exporters import export_json
from microarray_workstation.workflows.analysis_workflow import (
    analyze_batch_images,
    analyze_one_image,
    build_lab_interface_jobs_from_summaries,
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
    analyze.add_argument("--um-per-pixel", required=False, type=float, default=10.0)
    analyze.add_argument("--spot-diameter-min-um", required=False, type=float, default=40.0)
    analyze.add_argument("--spot-diameter-max-um", required=False, type=float, default=240.0)
    analyze.add_argument("--spot-spacing-min", required=False, type=float, default=0.0)
    analyze.add_argument("--spot-spacing-max", required=False, type=float, default=0.0)
    analyze.add_argument("--background-mode", required=False, choices=["local", "global"], default="local")
    analyze.add_argument("--global-bg-percentile", required=False, type=float, default=20.0)
    analyze.add_argument("--low-snr-threshold", required=False, type=float, default=1.5)
    analyze.add_argument("--saturation-threshold-pct", required=False, type=float, default=5.0)
    analyze.add_argument("--low-net-threshold", required=False, type=float, default=0.0)

    batch = sub.add_parser("analyze-batch", help="Batch analyze all chip images in a directory")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--rows", required=True, type=int)
    batch.add_argument("--cols", required=True, type=int)
    batch.add_argument("--template", required=False)
    batch.add_argument("--output-dir", required=True)
    batch.add_argument("--channel", required=False, type=int)
    batch.add_argument("--ai-model", required=False)
    batch.add_argument("--um-per-pixel", required=False, type=float, default=10.0)
    batch.add_argument("--spot-diameter-min-um", required=False, type=float, default=40.0)
    batch.add_argument("--spot-diameter-max-um", required=False, type=float, default=240.0)
    batch.add_argument("--spot-spacing-min", required=False, type=float, default=0.0)
    batch.add_argument("--spot-spacing-max", required=False, type=float, default=0.0)
    batch.add_argument("--background-mode", required=False, choices=["local", "global"], default="local")
    batch.add_argument("--global-bg-percentile", required=False, type=float, default=20.0)
    batch.add_argument("--low-snr-threshold", required=False, type=float, default=1.5)
    batch.add_argument("--saturation-threshold-pct", required=False, type=float, default=5.0)
    batch.add_argument("--low-net-threshold", required=False, type=float, default=0.0)

    submit = sub.add_parser("submit-lims", help="Submit interpreted result to generic LIMS REST endpoint")
    submit.add_argument("--base-url", required=True)
    submit.add_argument("--endpoint", required=True)
    submit.add_argument("--json", required=True, help="Summary JSON path")
    submit.add_argument("--token", required=False)

    submit_batch = sub.add_parser("submit-lims-batch", help="Submit all *_summary.json in directory to generic LIMS")
    submit_batch.add_argument("--base-url", required=True)
    submit_batch.add_argument("--endpoint", required=True)
    submit_batch.add_argument("--input-dir", required=True, help="Directory containing *_summary.json")
    submit_batch.add_argument("--token", required=False)
    submit_batch.add_argument("--output", required=False, help="Output summary json path")

    submit_lab = sub.add_parser(
        "submit-lab-interface",
        help="Submit one summary to laboratory_management interface (/lab/interface/inbound/<endpoint_code>)",
    )
    submit_lab.add_argument("--base-url", required=True)
    submit_lab.add_argument("--endpoint-code", required=True)
    submit_lab.add_argument("--json", required=True, help="Summary JSON path")
    submit_lab.add_argument("--auth-type", required=False, default="none", choices=["none", "bearer", "api_key", "basic"])
    submit_lab.add_argument("--token", required=False)
    submit_lab.add_argument("--api-key", required=False)
    submit_lab.add_argument("--username", required=False)
    submit_lab.add_argument("--password", required=False)

    submit_lab_batch = sub.add_parser(
        "submit-lab-interface-batch",
        help="Submit all *_summary.json in directory to laboratory_management inbound interface",
    )
    submit_lab_batch.add_argument("--base-url", required=True)
    submit_lab_batch.add_argument("--endpoint-code", required=True)
    submit_lab_batch.add_argument("--input-dir", required=True)
    submit_lab_batch.add_argument("--auth-type", required=False, default="none", choices=["none", "bearer", "api_key", "basic"])
    submit_lab_batch.add_argument("--token", required=False)
    submit_lab_batch.add_argument("--api-key", required=False)
    submit_lab_batch.add_argument("--username", required=False)
    submit_lab_batch.add_argument("--password", required=False)
    submit_lab_batch.add_argument("--output", required=False, help="Output summary json path")

    return parser


def _cmd_analyze(args: argparse.Namespace) -> int:
    um_per_px = max(0.1, float(args.um_per_pixel))
    dia_min_px = max(1.0, float(args.spot_diameter_min_um) / um_per_px)
    dia_max_px = max(dia_min_px + 0.5, float(args.spot_diameter_max_um) / um_per_px)
    out = analyze_one_image(
        image_path=args.image,
        rows=args.rows,
        cols=args.cols,
        template_path=args.template,
        output_dir=args.output_dir,
        channel=args.channel,
        ai_model=args.ai_model,
        spot_diameter_min_px=dia_min_px,
        spot_diameter_max_px=dia_max_px,
        spacing_min_px=args.spot_spacing_min,
        spacing_max_px=args.spot_spacing_max,
        background_mode=args.background_mode,
        global_background_percentile=args.global_bg_percentile,
        low_snr_threshold=args.low_snr_threshold,
        saturation_threshold_pct=args.saturation_threshold_pct,
        low_net_threshold=args.low_net_threshold,
    )

    print(f"raw_csv={out['raw_csv']}")
    print(f"interpreted_csv={out['interpreted_csv']}")
    print(f"summary_json={out['summary_json']}")
    print(f"qc_status={out['qc_status']} ai_mode={out['ai_summary']['mode']}")
    return 0


def _cmd_analyze_batch(args: argparse.Namespace) -> int:
    um_per_px = max(0.1, float(args.um_per_pixel))
    dia_min_px = max(1.0, float(args.spot_diameter_min_um) / um_per_px)
    dia_max_px = max(dia_min_px + 0.5, float(args.spot_diameter_max_um) / um_per_px)
    rows, summary_csv, summary_json = analyze_batch_images(
        input_dir=args.input_dir,
        rows=args.rows,
        cols=args.cols,
        template_path=args.template,
        output_dir=args.output_dir,
        channel=args.channel,
        ai_model=args.ai_model,
        spot_diameter_min_px=dia_min_px,
        spot_diameter_max_px=dia_max_px,
        spacing_min_px=args.spot_spacing_min,
        spacing_max_px=args.spot_spacing_max,
        background_mode=args.background_mode,
        global_background_percentile=args.global_bg_percentile,
        low_snr_threshold=args.low_snr_threshold,
        saturation_threshold_pct=args.saturation_threshold_pct,
        low_net_threshold=args.low_net_threshold,
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


def _lab_client_from_args(args: argparse.Namespace) -> LaboratoryManagementInterfaceClient:
    return LaboratoryManagementInterfaceClient(
        base_url=args.base_url,
        auth_type=args.auth_type,
        token=args.token,
        api_key=args.api_key,
        username=args.username,
        password=args.password,
    )


def _cmd_submit_lab_interface(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    payload = {
        "accession": data.get("accession") or Path(args.json).stem.replace("_summary", ""),
        "results": data.get("results") or [],
    }
    client = _lab_client_from_args(args)
    resp = client.submit_result_auto(endpoint_code=args.endpoint_code, payload=payload, external_uid=f"MW-{payload['accession']}")
    print(resp)
    return 0


def _cmd_submit_lab_interface_batch(args: argparse.Namespace) -> int:
    jobs = build_lab_interface_jobs_from_summaries(args.input_dir)
    client = _lab_client_from_args(args)
    result = client.submit_batch_result_auto(endpoint_code=args.endpoint_code, jobs=jobs)

    out_path = Path(args.output) if args.output else Path(args.input_dir) / "lab_interface_submit_summary.json"
    export_json(result, out_path)
    print(f"lab_interface_submit_summary={out_path}")
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
    if args.command == "submit-lab-interface":
        return _cmd_submit_lab_interface(args)
    if args.command == "submit-lab-interface-batch":
        return _cmd_submit_lab_interface_batch(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
