from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import requests


class LIMSClient:
    def __init__(self, base_url: str, token: str | None = None, timeout_sec: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_sec = timeout_sec

    def _headers(self, payload: dict[str, Any]) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        # Idempotency key from payload to avoid duplicate submissions.
        idem = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        headers["Idempotency-Key"] = idem
        return headers

    def submit_result(self, endpoint: str, payload: dict[str, Any], retries: int = 3) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._headers(payload)

        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
                resp.raise_for_status()
                return resp.json() if resp.content else {"status": "ok"}
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < retries:
                    time.sleep(1.2 * attempt)
                continue

        raise RuntimeError(f"Failed to submit LIMS result after {retries} attempts: {last_err}")

    def submit_batch_results(
        self,
        endpoint: str,
        payloads: list[tuple[str, dict[str, Any]]],
        retries: int = 3,
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        success = 0
        failed = 0

        for source, payload in payloads:
            try:
                resp = self.submit_result(endpoint=endpoint, payload=payload, retries=retries)
                records.append({"source": source, "status": "ok", "response": resp})
                success += 1
            except Exception as exc:  # noqa: BLE001
                records.append({"source": source, "status": "failed", "error": str(exc)})
                failed += 1

        return {
            "total": len(payloads),
            "success": success,
            "failed": failed,
            "records": records,
        }
