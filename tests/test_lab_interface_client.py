from __future__ import annotations

import json
from pathlib import Path

from microarray_workstation.integration.lab_interface_client import LaboratoryManagementInterfaceClient
from microarray_workstation.workflows.analysis_workflow import build_lab_interface_jobs_from_summaries


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.content = b"x"

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_submit_inbound_ack_ok(monkeypatch) -> None:
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        captured["json"] = json or {}
        return _FakeResponse({"ok": True, "ack_code": "AA", "job_id": 123})

    import requests

    monkeypatch.setattr(requests, "post", fake_post)

    client = LaboratoryManagementInterfaceClient(
        base_url="http://127.0.0.1:8069",
        auth_type="api_key",
        api_key="demo-key",
    )
    resp = client.submit_inbound(
        endpoint_code="HIS-REST",
        message_type="result",
        payload={"accession": "ACC-1", "results": [{"service_code": "S1", "result": "POSITIVE"}]},
        external_uid="MW-ACC-1",
    )

    assert captured["url"].endswith("/lab/interface/inbound/HIS-REST")
    assert captured["headers"]["X-API-Key"] == "demo-key"
    assert captured["json"]["jsonrpc"] == "2.0"
    assert captured["json"]["method"] == "call"
    assert captured["json"]["params"]["message_type"] == "result"
    assert captured["json"]["params"]["payload"]["accession"] == "ACC-1"
    assert resp["ack_code"] == "AA"


def test_build_lab_interface_jobs(tmp_path: Path) -> None:
    p = tmp_path / "chip_a_summary.json"
    p.write_text(
        json.dumps(
            {
                "accession": "ACC-XYZ",
                "results": [{"service_code": "GENE_A", "result": "POSITIVE", "note": "n"}],
            }
        ),
        encoding="utf-8",
    )

    jobs = build_lab_interface_jobs_from_summaries(str(tmp_path))
    assert len(jobs) == 1
    source, message_type, payload, external_uid = jobs[0]
    assert source.endswith("chip_a_summary.json")
    assert message_type == "result"
    assert payload["accession"] == "ACC-XYZ"
    assert payload["results"][0]["service_code"] == "GENE_A"
    assert external_uid == "MW-ACC-XYZ"


def test_submit_result_auto_prefers_external_api(monkeypatch) -> None:
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json or {}
        return _FakeResponse({"ok": True, "ack_code": "AA", "job_id": 456})

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    client = LaboratoryManagementInterfaceClient(base_url="http://127.0.0.1:8069")
    resp = client.submit_result_auto(
        endpoint_code="MICROARRAY_REST",
        payload={"accession": "ACC-2", "results": [{"service_code": "S2", "result": "NEGATIVE"}]},
        external_uid="MW-ACC-2",
    )
    assert "/lab/api/v1/MICROARRAY_REST/results" in captured["url"]
    assert captured["json"]["accession"] == "ACC-2"
    assert resp["ack_code"] == "AA"


def test_submit_result_auto_fallbacks_to_inbound(monkeypatch) -> None:
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        if "/lab/api/v1/" in url:
            raise RuntimeError("external_api_not_available")
        return _FakeResponse({"result": {"ok": True, "ack_code": "AA", "job_id": 789}})

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    client = LaboratoryManagementInterfaceClient(base_url="http://127.0.0.1:8069")
    resp = client.submit_result_auto(
        endpoint_code="MICROARRAY_REST",
        payload={"accession": "ACC-3", "results": [{"service_code": "S3", "result": "POSITIVE"}]},
        external_uid="MW-ACC-3",
    )
    assert calls["n"] >= 2
    assert resp["ack_code"] == "AA"
