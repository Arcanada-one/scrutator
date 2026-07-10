"""Minimal synchronous Model Connector client for golden-set candidate-generation tooling.

Mirrors `src/scrutator/ltm/llm.py`'s call shape (`POST /connectors/{connector}/execute`) but
stays stdlib-only (urllib) and synchronous, matching this directory's batch-script style.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class ModelConnectorError(Exception):
    """Raised when Model Connector is unreachable or returns a non-success status."""


class ModelConnectorClient:
    """Thin wrapper around `POST {mc_url}/connectors/{connector}/execute`.

    `connector` MUST be passed explicitly by the caller — no silent provider-default
    fallback (CLAUDE.md risk row: unset/default `coworker`/Model-Connector provider
    resolution has previously failed closed with `unknown provider 'none'`).
    """

    def __init__(self, mc_url: str, connector: str, model: str, api_key: str, timeout_s: float = 120.0):
        if not connector:
            raise ValueError("connector must be explicit — no silent provider-default fallback")
        self.mc_url = mc_url.rstrip("/")
        self.connector = connector
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def call(self, prompt: str, system: str | None = None) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        url = f"{self.mc_url}/connectors/{self.connector}/execute"
        body = {"prompt": full_prompt, "model": self.model, "maxTurns": 1}
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            # nosec B310 — mc_url is operator/config-supplied (default: the production Model
            # Connector https:// URL), never attacker-controlled input.
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310
                out = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            raise ModelConnectorError(f"Model Connector request failed: {exc}") from exc

        if out.get("status") != "success":
            raise ModelConnectorError(f"Model Connector error: {out.get('error', out)}")
        return out["result"]

    def call_json(self, prompt: str, system: str | None = None) -> dict | list:
        """Call and parse JSON out of the response, tolerating markdown code fences."""
        text = self.call(prompt, system).strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                start_idx = text.find(start_char)
                end_idx = text.rfind(end_char)
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        return json.loads(text[start_idx : end_idx + 1])
                    except json.JSONDecodeError:
                        continue
            raise ModelConnectorError(f"could not parse JSON from Model Connector response: {text[:200]!r}") from None
