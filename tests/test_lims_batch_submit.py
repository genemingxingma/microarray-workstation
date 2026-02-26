from __future__ import annotations

import json
from pathlib import Path

from microarray_workstation.cli import _cmd_submit_lims_batch, build_parser
from microarray_workstation.integration.lims_client import LIMSClient


def test_submit_lims_batch_writes_summary(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a_summary.json").write_text(json.dumps({"sample": "A"}), encoding="utf-8")
    (tmp_path / "b_summary.json").write_text(json.dumps({"sample": "B"}), encoding="utf-8")

    def fake_submit_batch_results(self, endpoint, payloads, retries=3):
        assert endpoint == "/api/results"
        assert len(payloads) == 2
        return {
            "total": 2,
            "success": 2,
            "failed": 0,
            "records": [{"source": payloads[0][0], "status": "ok"}, {"source": payloads[1][0], "status": "ok"}],
        }

    monkeypatch.setattr(LIMSClient, "submit_batch_results", fake_submit_batch_results)

    out = tmp_path / "lims_out.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "submit-lims-batch",
            "--base-url",
            "http://localhost:8000",
            "--endpoint",
            "/api/results",
            "--input-dir",
            str(tmp_path),
            "--output",
            str(out),
        ]
    )

    rc = _cmd_submit_lims_batch(args)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["success"] == 2
