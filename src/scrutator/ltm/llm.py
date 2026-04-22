"""LTM LLM client — sequential calls to Model Connector with permissive JSON parsing."""

from __future__ import annotations

import json
import logging
import re

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
    ):
        self.mc_url = mc_url.rstrip("/")
        self.connector = connector
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

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

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(url, json=body, headers=headers)

        if response.status_code >= 400:
            log.error("MC returned %d: %s", response.status_code, response.text[:200])
            raise LtmLlmError(f"MC returned {response.status_code}")

        data = response.json()
        if data.get("status") != "success":
            raise LtmLlmError(f"MC error: {data.get('error', data)}")

        return data["result"]

    async def extract_json(self, prompt: str, system: str | None = None) -> dict | list:
        """LLM call + permissive JSON parse."""
        raw = await self.call(prompt, system)
        return parse_json_permissive(raw)
