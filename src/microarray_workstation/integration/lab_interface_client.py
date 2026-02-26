from __future__ import annotations

import base64
import json
import time
from typing import Any

import requests


class LaboratoryManagementInterfaceClient:
    def __init__(
        self,
        base_url: str,
        *,
        auth_type: str = "none",
        token: str | None = None,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout_sec: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_type = auth_type
        self.token = token
        self.api_key = api_key
        self.username = username
        self.password = password
        self.timeout_sec = timeout_sec

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_type == "bearer" and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.auth_type == "api_key" and self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.auth_type == "basic" and self.username is not None and self.password is not None:
            raw = f"{self.username}:{self.password}".encode("utf-8")
            headers["Authorization"] = f"Basic {base64.b64encode(raw).decode('ascii')}"
        return headers

    @staticmethod
    def _unwrap_response(data: Any) -> dict[str, Any]:
        if isinstance(data, dict) and "result" in data and isinstance(data["result"], dict):
            return data["result"]
        if isinstance(data, dict):
            return data
        return {"ok": False, "ack_code": "AR", "error": f"unexpected_response: {type(data)}"}

    def submit_inbound(
        self,
        endpoint_code: str,
        *,
        message_type: str,
        payload: dict[str, Any],
        external_uid: str | None = None,
        raw_message: str | None = None,
        retries: int = 3,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/lab/interface/inbound/{endpoint_code}"
        params: dict[str, Any] = {"message_type": message_type, "payload": payload}
        if external_uid:
            params["external_uid"] = external_uid
        if raw_message:
            params["raw_message"] = raw_message
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": "call", "params": params}

        headers = self._headers()
        last_err: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=self.timeout_sec)
                resp.raise_for_status()
                payload_json = resp.json()
                if isinstance(payload_json, dict) and payload_json.get("error"):
                    raise RuntimeError(f"Endpoint error: {payload_json['error']}")
                parsed = self._unwrap_response(payload_json)
                ack = str(parsed.get("ack_code") or "")
                ok = bool(parsed.get("ok", ack == "AA"))
                if not ok or ack in {"AE", "AR"}:
                    raise RuntimeError(f"Rejected by endpoint: ack={ack or 'NA'} error={parsed.get('error') or ''}")
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < retries:
                    time.sleep(1.2 * attempt)
                continue

        raise RuntimeError(f"Failed to submit inbound interface after {retries} attempts: {last_err}")

    def submit_batch_inbound(
        self,
        endpoint_code: str,
        jobs: list[tuple[str, str, dict[str, Any], str | None]],
        retries: int = 3,
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        success = 0
        failed = 0

        for source, message_type, payload, external_uid in jobs:
            try:
                resp = self.submit_inbound(
                    endpoint_code=endpoint_code,
                    message_type=message_type,
                    payload=payload,
                    external_uid=external_uid,
                    retries=retries,
                )
                records.append({"source": source, "status": "ok", "response": resp})
                success += 1
            except Exception as exc:  # noqa: BLE001
                records.append({"source": source, "status": "failed", "error": str(exc)})
                failed += 1

        return {"total": len(jobs), "success": success, "failed": failed, "records": records}
