"""LTM LLM client — sequential calls to Model Connector with permissive JSON parsing."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Callable
from typing import Any

import httpx

log = logging.getLogger("scrutator.ltm.llm")


def parse_json_permissive(text: str) -> dict | list:
    """Parse JSON from LLM output. Tries strict, then extracts from markdown/text.

    Returns {"raw": text} if no JSON found.
    """
    text = text.strip()
    if not text:
        return {"raw": ""}

    # 1. Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Try extracting from markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Try finding JSON array in text (non-greedy to pick first)
    arr_match = re.search(r"(\[.*?\])", text, re.DOTALL)
    if arr_match:
        try:
            return json.loads(arr_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 4. Try finding JSON object in text (non-greedy)
    obj_match = re.search(r"(\{.*?\})", text, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    return {"raw": text}


class LtmLlmError(Exception):
    """Raised when Model Connector returns an error."""


class LtmLlmClient:
    """Sequential LLM client via Model Connector.

    Sends one request at a time to a CLI connector (Cursor by default).
    """

    def __init__(
        self,
        mc_url: str,
        connector: str = "cursor",
        model: str = "auto",
        api_key: str = "",
        timeout_s: int = 300,
        usage_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.mc_url = mc_url.rstrip("/")
        self.connector = connector
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.usage_sink = usage_sink

    def _emit_usage(self, data: dict[str, Any], status: str) -> None:
        """Emit billing metadata without prompts, results, credentials, or errors."""
        if self.usage_sink is None:
            return
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        request_id = data.get("id")
        request_id_hash = None
        if isinstance(request_id, str) and len(request_id) <= 256:
            request_id_hash = hashlib.sha256(request_id.encode()).hexdigest()
        self.usage_sink(
            {
                "request_id_sha256": request_id_hash,
                "connector": self.connector,
                "model": self.model,
                "status": status,
                "input_tokens": self._usage_number(usage.get("inputTokens")),
                "output_tokens": self._usage_number(usage.get("outputTokens")),
                "total_tokens": self._usage_number(usage.get("totalTokens")),
                "cost_usd": self._usage_number(usage.get("costUsd")),
            }
        )

    @staticmethod
    def _usage_number(value: Any) -> int | float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        if not math.isfinite(value) or value < 0:
            return 0
        return value

    @staticmethod
    def _response_data(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except (TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    async def call(self, prompt: str, system: str | None = None) -> str:
        """Single LLM call via MC. Returns raw text response."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        url = f"{self.mc_url}/connectors/{self.connector}/execute"
        body: dict = {
            "prompt": full_prompt,
            "model": self.model,
            "maxTurns": 1,
            "responseFormat": {"type": "json_object"},
        }
        headers: dict = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            self._emit_usage({}, "transport_error")
            raise LtmLlmError("MC transport error") from exc

        data = self._response_data(response)
        if response.status_code >= 400:
            self._emit_usage(data, "http_error")
            log.error("MC returned HTTP %d", response.status_code)
            raise LtmLlmError(f"MC returned {response.status_code}")

        if data.get("status") != "success":
            self._emit_usage(data, "connector_error")
            raise LtmLlmError("MC connector error")

        result = data.get("result")
        if not isinstance(result, str):
            self._emit_usage(data, "response_contract_error")
            raise LtmLlmError("MC response contract error")
        self._emit_usage(data, "success")
        return result

    async def extract_json(self, prompt: str, system: str | None = None) -> dict | list:
        """LLM call + permissive JSON parse."""
        raw = await self.call(prompt, system)
        return parse_json_permissive(raw)
